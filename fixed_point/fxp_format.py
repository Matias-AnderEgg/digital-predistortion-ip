"""Configuración central de punto fijo para la simulación del DPD.

Modela el datapath de una FPGA: define formatos (ancho de palabra / bits
fraccionarios) separados para las señales, los coeficientes y el acumulador,
y provee un helper `quantize` para cuantizar arrays de numpy con fxpmath.

El objetivo es poder correr EXACTAMENTE el mismo código del DPD en punto
flotante (referencia) o en punto fijo, según el flag `enabled` del config.
Así se compara la degradación de EVM/ACPR/BER al bajar el ancho de palabra.

Escala de la señal: el TX se normaliza a RMS=1 y la OFDM tiene PAPR ~12 dB,
así que los picos llegan a ~4. Por eso el formato de señal necesita ~3 bits
enteros (signo + magnitud hasta 4); el resto son bits fraccionarios.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from fxpmath import Fxp


@dataclass(frozen=True)
class FxpFormat:
    """Formato de un registro de punto fijo Q(n_int).(n_frac)."""

    n_word: int
    n_frac: int
    signed: bool = True

    @property
    def n_int(self) -> int:
        """Bits enteros (sin contar el signo)."""
        return self.n_word - self.n_frac - (1 if self.signed else 0)

    @property
    def dtype(self) -> str:
        s = "s" if self.signed else "u"
        return f"fxp-{s}{self.n_word}/{self.n_frac}"


@dataclass(frozen=True)
class FxpConfig:
    """Formatos del datapath del DPD.

    Todos los formatos aplican a datos complejos (parte real e imaginaria
    comparten ancho de palabra, como en una FPGA con dos caminos idénticos).

    - data:  muestras de señal (x, u) y términos de la base MP.
    - coeff: coeficientes del Memory Polynomial (w del DPD, h del modelo PA).
    - accum: acumulador del producto interno coef*base (más ancho para no
             perder precisión al sumar muchos términos).

    Si `enabled` es False, `quantize` es la identidad: el código corre en
    punto flotante de doble precisión (camino de referencia).
    """

    data: FxpFormat
    coeff: FxpFormat
    accum: FxpFormat
    rounding: str = "around"   # redondeo al más cercano (vs 'trunc' = piso)
    overflow: str = "saturate"  # satura en los límites (vs 'wrap')
    enabled: bool = True
    name: str = ""
    # Convención normalizada del supervisor: la señal x_phys entra al bloque
    # ya dividida por `signal_scale` (G, idealmente potencia de 2 = shift),
    # se procesa en el rango [-1, 1), y la salida u_norm sale multiplicada por
    # G para volver a la escala física. Por orden p, después de coef*basis se
    # aplica un shift de G^(p-1) para compensar (también gratis en HW).
    # Con signal_scale=1.0 el comportamiento es el actual (sin normalizar).
    signal_scale: float = 1.0
    # Acumulador interno del LMS para los coeficientes durante el entrenamiento.
    # En FPGA real es típico: storage angosto (coeff, p.ej. 16 bits) para que el
    # DSP48 multiplique en 1 ciclo y para que el host lea/escriba en 32 bits I+Q,
    # pero el lazo LMS acumula los micro-updates en un registro MÁS ANCHO para
    # evitar que coefs chicos (~1e-4) se queden trabados en 0 por el step de
    # cuantización del storage. Solo afecta al training: las predicciones siguen
    # usando el storage (coeff). Si es None, el comportamiento es el anterior
    # (acumulador = storage, sin separación).
    coef_accum: FxpFormat | None = None
    # Escala de la salida. El DPD expande los picos y la salida necesita más
    # rango que la entrada: con la misma escala recorta una de cada 236
    # muestras a -9 dB (Apéndice B de la tesis). Como es potencia de dos,
    # cambiarla es un corrimiento de bits. None usa signal_scale.
    output_scale: float | None = None
    # Como se calcula |x|^(p-1) en el datapath.
    #   "sqrt"    : ax = sqrt(I^2+Q^2) cuantizada al formato de dato, y de ahí
    #               las potencias. Es lo que hacia el modelo historicamente.
    #   "squared" : a2 = I^2+Q^2 en el acumulador ancho, y las potencias por
    #               multiplicaciones sucesivas por a2. Como todos los órdenes
    #               son impares, los exponentes de la magnitud son pares y la
    #               raíz no hace falta: sale igual de preciso (0.02 dB de
    #               diferencia) y ahorra la unidad de raíz cuadrada en HW.
    magnitude_mode: str = "sqrt"

    @property
    def out_scale(self) -> float:
        """Escala efectiva del registro de salida."""
        return self.signal_scale if self.output_scale is None else self.output_scale

    def label(self) -> str:
        if self.name:
            return self.name
        if not self.enabled:
            return "float"
        return f"data={self.data.dtype} coeff={self.coeff.dtype}"


# Formato de referencia: bypass de cuantización (doble precisión).
FLOAT = FxpConfig(
    data=FxpFormat(64, 52),
    coeff=FxpFormat(64, 52),
    accum=FxpFormat(64, 52),
    enabled=False,
    name="float",
)

# Default "explore": 16 bits, 3 bits enteros para la señal (picos OFDM ~4),
# acumulador de 40 bits para el producto interno del MP.
DEFAULT = FxpConfig(
    data=FxpFormat(n_word=16, n_frac=13),
    coeff=FxpFormat(n_word=16, n_frac=13),
    accum=FxpFormat(n_word=40, n_frac=26),
    name="s16/13",
)


def make_config(
    n_word: int,
    n_frac: int,
    *,
    coeff_n_word: int | None = None,
    coeff_n_frac: int | None = None,
    accum_extra_int: int = 8,
    accum_extra_frac: int = 13,
    rounding: str = "around",
    overflow: str = "saturate",
    output_scale: float | None = None,
    magnitude_mode: str = "sqrt",
    name: str | None = None,
) -> FxpConfig:
    """Construye un FxpConfig a partir del formato de señal.

    Por defecto los coeficientes usan el mismo formato que la señal, pero se
    pueden desacoplar con `coeff_n_word`/`coeff_n_frac`. Esto importa porque
    los coeficientes de orden alto del MP son ~1e-4 y necesitan mas bits
    fraccionarios que la señal para no redondearse a cero (lo que colapsaria
    el DPD a la identidad).

    El acumulador agrega bits de guarda enteros y fraccionarios sobre el
    formato de la señal, para modelar el ancho extendido del DSP/MAC.
    """
    data = FxpFormat(n_word=n_word, n_frac=n_frac)
    coeff = FxpFormat(
        n_word=coeff_n_word if coeff_n_word is not None else n_word,
        n_frac=coeff_n_frac if coeff_n_frac is not None else n_frac,
    )
    accum = FxpFormat(
        n_word=n_word + accum_extra_int + accum_extra_frac,
        n_frac=n_frac + accum_extra_frac,
    )
    if name is None:
        name = f"s{n_word}/{n_frac}"
        if (coeff.n_word, coeff.n_frac) != (data.n_word, data.n_frac):
            name += f" c:s{coeff.n_word}/{coeff.n_frac}"
    return FxpConfig(
        data=data,
        coeff=coeff,
        accum=accum,
        rounding=rounding,
        overflow=overflow,
        output_scale=output_scale,
        magnitude_mode=magnitude_mode,
        name=name,
    )


def make_normalized_config(
    n_word_data: int = 16,
    n_word_coef: int = 16,
    n_word_coef_accum: int | None = 24,
    coef_n_int: int = 1,
    signal_scale: float = 8.0,
    accum_extra_int: int = 8,
    accum_extra_frac: int = 13,
    rounding: str = "around",
    overflow: str = "saturate",
    output_scale: float | None = None,
    magnitude_mode: str = "sqrt",
    name: str | None = None,
) -> FxpConfig:
    """Convención normalizada del supervisor (FPGA estándar).

    - Señal: all-fractional (1 sign + (n_word_data-1) frac, rango [-1, 1)).
      La señal física se divide por `signal_scale` (G) antes de entrar al
      bloque (shift externo, gratis en HW si G es potencia de 2).
    - Coef: 1 sign + coef_n_int int + (n_word_coef-1-coef_n_int) frac.
      Almacenado en su escala física natural (max ~1 para nuestros DPDs).
    - Por orden p: después de la multiplicación coef*basis se aplica un
      shift de G^(p-1) (también gratis en HW: solo alineación de bits en el
      acumulador). Esto compensa la normalización para los términos no-lineales.
    - Salida: u_norm se multiplica por G externamente para recuperar u_phys.

    Con esta convención, todas las multiplicaciones intermedias caben en 1
    DSP48 (coef de 16 bits entra en B=18 bits, señal en A=25 bits).
    """
    # Señal: 1 sign + 0 enteros + (N-1) fraccionales
    data = FxpFormat(n_word=n_word_data, n_frac=n_word_data - 1)
    # Coef storage: 1 sign + coef_n_int int + restante fraccional
    coef = FxpFormat(n_word=n_word_coef, n_frac=n_word_coef - 1 - coef_n_int)
    # Coef accumulator interno del LMS: mismo n_int, más bits fraccionarios para
    # poder acumular updates chicos sin que se redondeen a 0 en el storage.
    # None -> mismo formato que coef (sin separación storage/internal).
    if n_word_coef_accum is not None:
        coef_accum = FxpFormat(n_word=n_word_coef_accum,
                               n_frac=n_word_coef_accum - 1 - coef_n_int)
    else:
        coef_accum = None
    # Acumulador del datapath (MAC): ancho con headroom para shifts por orden.
    accum = FxpFormat(
        n_word=n_word_data + accum_extra_int + accum_extra_frac,
        n_frac=(n_word_data - 1) + accum_extra_frac,
    )
    if name is None:
        name = f"norm G={signal_scale:g} d=s{n_word_data}/{data.n_frac} c=s{n_word_coef}/{coef.n_frac}"
        if coef_accum is not None and coef_accum.n_word != coef.n_word:
            name += f" ca=s{coef_accum.n_word}/{coef_accum.n_frac}"
    return FxpConfig(
        data=data,
        coeff=coef,
        accum=accum,
        coef_accum=coef_accum,
        signal_scale=signal_scale,
        rounding=rounding,
        overflow=overflow,
        output_scale=output_scale,
        magnitude_mode=magnitude_mode,
        name=name,
    )


def quantize(x, fmt: FxpFormat, cfg: FxpConfig):
    """Cuantiza un escalar o array (real o complejo) al formato `fmt`.
    Devuelve un numpy array (o escalar) en punto flotante con el valor ya
    cuantizado, de modo que el resto del código numpy lo procesa normalmente.
    Si `cfg.enabled` es False, devuelve la entrada sin tocar (camino float).
    """
    if not cfg.enabled:
        return x
    arr = np.asarray(x)
    q = Fxp(
        arr,
        signed=fmt.signed,
        n_word=fmt.n_word,
        n_frac=fmt.n_frac,
        rounding=cfg.rounding,
        overflow=cfg.overflow,
    )
    out = q.astype(complex) if np.iscomplexobj(arr) else q.astype(float)
    return out
