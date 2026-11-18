"""Generador de vectores de referencia (golden vectors) para verificar el
datapath de aplicación del DPD en RTL contra el modelo de punto fijo de Python.

Alcance: SOLO el bloque de aplicación (`apply_memory_poly_fxp`), es decir el
Memory Polynomial ya entrenado, corriendo con coeficientes congelados. Es el
bloque que primero se implementa en RTL (el lazo de entrenamiento GD es un
bloque aparte, con su propia verificación más adelante).

Convención:
    - Señal:       s16/15 (Q0.15), all-fractional, rango [-1, 1).
    - Coeficiente: s16/15 (Q0.15), el mismo formato que la señal. w[1,0]
      (orden 1, tap 0) está fijado por HW en 1.0 en el modelo matemático, pero en Q0.15 el máximo representable es 32767/32768 =
      0.999969482..., así que ese coeficiente queda saturado a ese valor
      (no a 1.0 exacto). Esto es intencional, no un bug.
    - Acumulador interno de entrenamiento: s24/22 (wide accumulator), no
      afecta estos vectores (son de INFERENCIA, no de entrenamiento).
    - Shift externo de ENTRADA G=8=2^3 (potencia de 2 -> gratis en HW).
    - Shift externo de SALIDA G_out=128=2^7, DISTINTO al de entrada. El DPD
      expande los picos: con la señal LTE y G_out=8 recortaría una de cada
      236 muestras, y el error del núcleo contra el flotante pasaría de
      -54.9 a -14.7 dB. El pico NO converge (ver config.md): crece unos
      10 dB cada dos décadas de muestras, porque el termino de orden 7
      amplifica la cola de una señal OFDM, que es casi gaussiana.
    - Magnitud SIN raíz cuadrada: |x|^2 = I^2+Q^2 en el acumulador ancho, y
      las potencias por multiplicaciones sucesivas. Como todos los órdenes
      son impares, los exponentes de la magnitud son pares y la raíz no hace
      falta. Cuesta 0.02 dB de precisión y ahorra la unidad de raíz en HW.
    - Pmax=7 (órdenes impares 1,3,5,7), M=3 taps de memoria -> 12
      coeficientes complejos por bloque (w para el DPD).

Salida: fixed_point/golden_vectors/out/
    coeffs_w_re.hex, coeffs_w_im.hex          (12 líneas c/u, hex 16-bit)
    functional_input_i.hex / _q.hex           (N líneas, hex 16-bit)
    functional_output_i.hex / _q.hex          (N líneas, hex 16-bit)
    directed_input_i.hex / _q.hex             (idem, casos de esquina)
    directed_output_i.hex / _q.hex
    golden_vectors_functional.csv             (vista humana: float + hex)
    golden_vectors_directed.csv               (idem, con etiqueta por caso)
    config.md                                 (documentación de convención)

Uso:
    python fixed_point/golden_vectors/generate.py
"""
from __future__ import annotations

import os
import sys
import csv

import numpy as np
from fxpmath import Fxp

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fixed_point import fxp_format as fxp
from fixed_point.dpd import train_dpd_gd, apply_memory_poly_fxp
from fixed_point import lte
from fixed_point.tx import qam16_gray_mod

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT_DIR, exist_ok=True)

# Config del datapath: Q0.15 para señal Y coeficientes, G=8, acc wide=24 bit.
PMAX = 7
M_TAPS = 3
ORDERS = list(range(1, PMAX + 1, 2))  # [1, 3, 5, 7]
BO_DB = -9.0  # peor caso: el mismo corner usado para justificar M=3/P=7.
G = 8.0        # shift externo de entrada
G_OUT = 128.0  # shift externo de salida (distinto: el DPD expande los picos)

CFG = fxp.make_normalized_config(
    n_word_data=16,
    n_word_coef=16,
    n_word_coef_accum=24,
    coef_n_int=0,              # Q0.15 para el coeficiente
    signal_scale=G,
    output_scale=G_OUT,
    magnitude_mode="squared",  # sin raíz cuadrada
)

N_WORD = 16
N_FRAC = 15  # Q0.15 tanto para dato como para coeficiente


def to_code(values: np.ndarray) -> np.ndarray:
    """Complejo (ya cuantizado a Q0.15) -> códigos enteros de 16 bits (I, Q)."""
    values = np.asarray(values).reshape(-1)
    re = Fxp(values.real, signed=True, n_word=N_WORD, n_frac=N_FRAC,
              rounding=CFG.rounding, overflow=CFG.overflow)
    im = Fxp(values.imag, signed=True, n_word=N_WORD, n_frac=N_FRAC,
              rounding=CFG.rounding, overflow=CFG.overflow)
    return re.val.astype(np.int32), im.val.astype(np.int32)


def hex16(code: int) -> str:
    return format(int(code) & 0xFFFF, "04X")


def write_hex(path: str, codes: np.ndarray) -> None:
    with open(path, "w") as f:
        for c in codes:
            f.write(hex16(c) + "\n")


def write_csv(path: str, rows: list[dict], fieldnames: list[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


# La señal LTE de la tesis. build_params arma la OFDM de 52 portadoras, que
# los vectores no usan y queda porque la usan otros scripts.
N_SIM_LTE = 364   # largo de las corridas de la tesis
N_TRAIN = 16000


def senal_lte(semilla):
    """La señal de la tesis entera (364 símbolos), filtrada y con RMS 1. La
    misma que sw/hw_test/lte_placa.py."""
    bits = np.random.default_rng(semilla).integers(0, 2, N_SIM_LTE * lte.NUSED * 4)
    x, grilla = lte.modular(qam16_gray_mod(bits), N_SIM_LTE)
    return lte.filtrar(x, lte.filtro_tx()), grilla


# 1) Entrenar el DPD (float, mismo pipeline validado) y congelar coeficientes
#    en Q0.15 -> estos son los valores que se escriben en W_COEFF_RE/IM.
def build_params():
    """La OFDM de 52 portadoras. Los vectores usan senal_lte."""
    Nfft, Nused, cp, osf, df = 64, 52, 16, 4, 15e3
    fs = Nfft * df * osf
    return dict(
        Nfft=Nfft, Nused=Nused, cp_len=cp, os_factor=osf,
        n_ofdm_symbols=1000, fs=fs, fs_pa=fs,
        main_bw=Nused * df, adj_offset=Nfft * df, adj_bw=Nused * df,
        R0=50.0, pa_mode="arctan_mem", pa_nl="arctan", use_memory=True,
        sat_amp=1.0, p_soft=3.0, plot_time_debug=False,
    )


CACHE_PATH = os.path.join(os.path.dirname(__file__), "coeffs_cache.npz")


def _train_signature(train_gd: dict) -> str:
    """Firma de todo lo que afecta al entrenamiento.

    Deliberadamente NO incluye `output_scale` ni `magnitude_mode`: esas dos
    solo afectan la inferencia (`apply_memory_poly_fxp`), no los coeficientes
    que devuelve el GD, así que cambiarlas no tiene por que invalidar el cache.
    """
    campos = dict(
        bo_db=BO_DB, pmax=PMAX, m=M_TAPS,
        data=(CFG.data.n_word, CFG.data.n_frac),
        coeff=(CFG.coeff.n_word, CFG.coeff.n_frac),
        accum=(CFG.accum.n_word, CFG.accum.n_frac),
        coef_accum=None if CFG.coef_accum is None
        else (CFG.coef_accum.n_word, CFG.coef_accum.n_frac),
        signal_scale=CFG.signal_scale,
        rounding=CFG.rounding, overflow=CFG.overflow,
        senal="lte", seed_train=123, train_gd=sorted(train_gd.items()),
    )
    return repr(campos)


def train_coeffs(usar_cache: bool = True):
    params = lte.params()
    train_common = dict(
        pa_mode=params["pa_mode"], pa_nl=params["pa_nl"], use_memory=params["use_memory"],
        fs_in=params["fs"], fs_pa=params["fs_pa"], R0=params["R0"],
        Pmax=PMAX, M=M_TAPS, sat_amp=params["sat_amp"], p_soft=params["p_soft"],
    )
    # Entrenamiento en flotante, como en el ARM del SoC, y recién al final a
    # Q0.15. Modelo del PA con mu_h=0.001, sin NLMS, y respaldo con NLMS si
    # diverge. Ver train_dpd_gd.
    train_gd = dict(mu_h=0.001, use_nlms_h=False, mu_h_respaldo=0.05, train_fxp=False,
                    mu_w=0.5, ep_pa=40, ep_dpd=5, h_ganancia_max=1.0,
                    N_train_max=N_TRAIN, w_reg_lambda=1e-4, w_reg_pmin=3)

    firma = _train_signature(train_gd)
    if usar_cache and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=False)
        if str(z["firma"]) == firma:
            coeffs_float = z["coeffs_float"]
            print("Coeficientes leidos del cache (mismos hiperparametros).", flush=True)
            return params, coeffs_float, fxp.quantize(coeffs_float, CFG.coeff, CFG)
        print("Cache invalido (cambiaron los hiperparametros), reentrenando.", flush=True)

    tx_train = senal_lte(123)[0]

    print("Entrenando DPD (GD, convención Q0.15, BO=%.0f dB, LTE)..." % BO_DB, flush=True)
    train_out = train_dpd_gd(
        tx_train[:train_gd["N_train_max"]],
        bo_db=BO_DB, **train_common, normalize_gain=False,
        fxp_cfg=CFG, **train_gd,
    )
    coeffs_float = np.asarray(train_out["dpd_coeffs"], dtype=complex)
    np.savez(CACHE_PATH, coeffs_float=coeffs_float, firma=np.array(firma))

    # Congelar en Q0.15: estos son EXACTAMENTE los códigos que se escriben
    # por AXI-Lite en W_COEFF_RE[12] / W_COEFF_IM[12].
    coeffs_q = fxp.quantize(coeffs_float, CFG.coeff, CFG)
    return params, coeffs_float, coeffs_q


def order_tap_labels():
    labels = []
    for q in range(M_TAPS):
        for p in ORDERS:
            labels.append(f"w[{p},{q}]")
    return labels


def dump_coeffs(coeffs_q: np.ndarray, coeffs_float: np.ndarray):
    re_codes, im_codes = to_code(coeffs_q)
    write_hex(os.path.join(OUT_DIR, "coeffs_w_re.hex"), re_codes)
    write_hex(os.path.join(OUT_DIR, "coeffs_w_im.hex"), im_codes)

    labels = order_tap_labels()
    rows = []
    for i, lbl in enumerate(labels):
        rows.append(dict(
            idx=i, label=lbl,
            w_float_re=f"{coeffs_float[i].real:.8f}", w_float_im=f"{coeffs_float[i].imag:.8f}",
            w_q015_re=f"{coeffs_q[i].real:.8f}", w_q015_im=f"{coeffs_q[i].imag:.8f}",
            code_re=int(re_codes[i]), code_im=int(im_codes[i]),
            hex_re=hex16(re_codes[i]), hex_im=hex16(im_codes[i]),
        ))
    write_csv(os.path.join(OUT_DIR, "coeffs_w.csv"), rows,
              ["idx", "label", "w_float_re", "w_float_im", "w_q015_re", "w_q015_im",
               "code_re", "code_im", "hex_re", "hex_im"])
    return rows


# 2) Vector funcional: el comienzo de la señal LTE de evaluación (semilla
#    456, ya con RMS 1), aplicado al MP congelado. Son las primeras muestras
#    de la subtrama de sw/hw_test/lte_placa.py, sin renormalizar.
def gen_functional_vectors(params, coeffs_q, n_samples=4096):
    x_dpd = senal_lte(456)[0][:n_samples]

    x_norm = fxp.quantize(x_dpd / G, CFG.data, CFG)          # lo que ve el datapath
    u_phys = apply_memory_poly_fxp(x_dpd, coeffs_q, Pmax=PMAX, M=M_TAPS, cfg=CFG)
    u_norm = fxp.quantize(u_phys / G_OUT, CFG.data, CFG)      # dominio interno, pre-shift de salida

    xi, xq = to_code(x_norm)
    ui, uq = to_code(u_norm)

    write_hex(os.path.join(OUT_DIR, "functional_input_i.hex"), xi)
    write_hex(os.path.join(OUT_DIR, "functional_input_q.hex"), xq)
    write_hex(os.path.join(OUT_DIR, "functional_output_i.hex"), ui)
    write_hex(os.path.join(OUT_DIR, "functional_output_q.hex"), uq)

    rows = []
    for n in range(n_samples):
        rows.append(dict(
            n=n,
            x_phys_re=f"{x_dpd[n].real:.6f}", x_phys_im=f"{x_dpd[n].imag:.6f}",
            x_code_i=int(xi[n]), x_code_q=int(xq[n]),
            x_hex_i=hex16(xi[n]), x_hex_q=hex16(xq[n]),
            u_phys_re=f"{u_phys[n].real:.6f}", u_phys_im=f"{u_phys[n].imag:.6f}",
            u_code_i=int(ui[n]), u_code_q=int(uq[n]),
            u_hex_i=hex16(ui[n]), u_hex_q=hex16(uq[n]),
        ))
    write_csv(os.path.join(OUT_DIR, "golden_vectors_functional.csv"), rows,
              ["n", "x_phys_re", "x_phys_im", "x_code_i", "x_code_q", "x_hex_i", "x_hex_q",
               "u_phys_re", "u_phys_im", "u_code_i", "u_code_q", "u_hex_i", "u_hex_q"])

    # Pico real de |x_norm| en código. Los casos dirigidos se arman alrededor
    # de este valor y no del fondo de escala del formato, que es ~3x mas
    # grande que cualquier cosa que el bloque vaya a ver en operacion.
    pico = int(np.max(np.hypot(xi.astype(float), xq.astype(float))))
    print(f"  pico de |x_norm| en el vector funcional: {pico} "
          f"({pico/32768:.4f} de fondo de escala)", flush=True)
    return n_samples, pico


# 3) Vectores dirigidos: casos de esquina del datapath (cero, LSB, full-scale,
#    saturación de magnitud, ráfaga alternada para ejercitar los M=3 taps).
def build_directed_cases(pico_op: int):
    """Casos de esquina del datapath, como SECUENCIA (no muestras sueltas).

    La línea de retardo de M taps arrastra estado de una muestra a la
    siguiente, así que el orden importa: cada grupo arranca con la línea
    vaciada por M ceros y termina vaciandola de nuevo. Sin eso, la salida de
    un caso depende de los casos anteriores y deja de ser interpretable.

    Los casos están armados alrededor del pico REAL de la señal
    (`pico_op`), no del fondo de escala del formato. Con fondo de escala la
    salida satura casi siempre, y un vector que satura no distingue nada: un
    RTL con un signo dado vuelta o una columna de coeficientes mal indexada
    satura al mismo valor y pasa igual. La saturación se prueba aparte, al
    final, con unos pocos casos hechos a propósito para eso.
    """
    FS, NEG_FS = 32767, -32768
    OP = int(pico_op)

    def flush(tag):
        """M ceros: deja la linea de retardo en un estado conocido."""
        return [(f"flush_{tag}_{k}", 0, 0) for k in range(M_TAPS)]

    cases = []

    # 1) Básicos, con la línea de retardo ya limpia (arranca en cero).
    cases += [("zero", 0, 0),
              ("lsb_pos_re", 1, 0), ("lsb_neg_re", -1, 0),
              ("lsb_pos_im", 0, 1), ("lsb_neg_im", 0, -1)]
    cases += flush("basicos")

    # 2) Impulsos: una muestra sola seguida de M ceros. La respuesta al
    #    impulso recorre los taps uno por uno, así que estos casos verifican
    #    directamente el indexado de taps y el orden de las columnas de
    #    coeficientes, que es uno de los errores gruesos que queremos cazar.
    diag = int(round(OP / np.sqrt(2)))
    for nombre, re_, im_ in [("imp_re_pos", OP, 0), ("imp_re_neg", -OP, 0),
                             ("imp_im_pos", 0, OP), ("imp_im_neg", 0, -OP),
                             ("imp_diag", diag, diag)]:
        cases.append((nombre, re_, im_))
        cases += [(f"{nombre}_tap{k}", 0, 0) for k in range(1, M_TAPS + 1)]

    # 3) Escalera de amplitud dentro del rango de operacion. La contribucion
    #    de cada orden crece como |x|^p, así que barrer la amplitud separa
    #    los cuatro órdenes: si uno esta mal, el error crece con el escalon.
    for k in range(1, 9):
        cases.append((f"ramp_{k}_de_8", (OP * k) // 8, 0))
    cases += flush("ramp")

    # 4) Rafaga alternada: ejercita los M taps a la vez con valores opuestos.
    for k in range(4):
        cases.append((f"alt_burst_{k}", OP if k % 2 == 0 else -OP, 0))
    cases += flush("burst")

    # 5) Saturación deliberada, al final para no contaminar lo anterior.
    cases += [("sat_max_re", FS, 0), ("sat_min_re", NEG_FS, 0),
              ("sat_corner_re_im", FS, FS)]
    cases += flush("sat")
    return cases


def gen_directed_vectors(coeffs_q, pico_op: int):
    cases = build_directed_cases(pico_op)
    labels = [c[0] for c in cases]
    x_norm_complex = np.array(
        [(re / 32768.0) + 1j * (im / 32768.0) for _, re, im in cases], dtype=complex
    )
    # Estos códigos YA están en el dominio interno (post shift /G). Para
    # reusar apply_memory_poly_fxp (que hace x_dpd/G internamente) hay que
    # pasarle x_dpd = x_norm * G.
    x_dpd = x_norm_complex * G

    x_norm_q = fxp.quantize(x_dpd / G, CFG.data, CFG)
    u_phys = apply_memory_poly_fxp(x_dpd, coeffs_q, Pmax=PMAX, M=M_TAPS, cfg=CFG)
    u_norm = fxp.quantize(u_phys / G_OUT, CFG.data, CFG)

    xi, xq = to_code(x_norm_q)
    ui, uq = to_code(u_norm)

    write_hex(os.path.join(OUT_DIR, "directed_input_i.hex"), xi)
    write_hex(os.path.join(OUT_DIR, "directed_input_q.hex"), xq)
    write_hex(os.path.join(OUT_DIR, "directed_output_i.hex"), ui)
    write_hex(os.path.join(OUT_DIR, "directed_output_q.hex"), uq)

    rows = []
    for n, lbl in enumerate(labels):
        rows.append(dict(
            n=n, label=lbl,
            x_code_i=int(xi[n]), x_code_q=int(xq[n]),
            x_hex_i=hex16(xi[n]), x_hex_q=hex16(xq[n]),
            u_phys_re=f"{u_phys[n].real:.6f}", u_phys_im=f"{u_phys[n].imag:.6f}",
            u_code_i=int(ui[n]), u_code_q=int(uq[n]),
            u_hex_i=hex16(ui[n]), u_hex_q=hex16(uq[n]),
        ))
    write_csv(os.path.join(OUT_DIR, "golden_vectors_directed.csv"), rows,
              ["n", "label", "x_code_i", "x_code_q", "x_hex_i", "x_hex_q",
               "u_phys_re", "u_phys_im", "u_code_i", "u_code_q", "u_hex_i", "u_hex_q"])
    return len(cases)


def write_config_doc(n_functional, n_directed, coeff_rows):
    w10 = coeff_rows[0]  # idx 0 = w[1,0]
    doc = f"""# Golden vectors — bloque de aplicación del DPD (Memory Polynomial)

Generado por `generate.py`. Verifica el datapath de INFERENCIA del DPD
(coeficientes ya entrenados y congelados), NO el lazo de entrenamiento GD.

## Convención de punto fijo

| Campo        | Formato | Rango                | Nota |
|--------------|---------|-----------------------|------|
| Señal de entrada | s16/15 (Q0.15) | [-1, 1)     | shift externo G={G:g} (x_phys = x_norm * {G:g}) |
| Señal de salida  | s16/15 (Q0.15) | [-1, 1)     | shift externo G_out={G_OUT:g} (u_phys = u_norm * {G_OUT:g}) |
| Coeficiente  | s16/15 (Q0.15) | [-1, 0.999969482...] | antes Q1.14; w[1,0] satura a 0.999969482 (no llega a 1.0 exacto) |
| Redondeo / overflow | {CFG.rounding} / {CFG.overflow} | | |

### Por qué la salida usa otra escala que la entrada

No tienen por qué ser iguales. El DPD expande los picos, y con G_out = G = 8
la salida recortaría. Como el corrimiento es potencia de dos, cambiarlo son
bits de alineación: cero compuertas.

Los números de esta sección son de la señal LTE, con el predistorsionador
del barrido de la tesis (M = 3, P = 7, BO -9 dB, entrenado en Python con el
paso del modelo limitado a 1 y 16000 muestras), no con los 12 códigos de
estos vectores. Los da `experimentos/analisis_salida_zf.py`, con la corrida copiada en su
encabezado, y son los del Apéndice B de la tesis. Se miden sobre la señal de evaluación (semilla 456,
364 símbolos, 1597440 muestras):

| `G_out` | recortan | error del núcleo contra el flotante | margen sobre el pico |
|---|---|---|---|
| 8 | 1 de cada 236 | -14.7 dB | -17.0 dB |
| **128** | **ninguna** | **-54.9 dB** | **7.1 dB** |

Con 8 el recorte domina el error del núcleo por unos 40 dB.

**El pico no converge.** Con 10 señales de 364 símbolos (semillas 456 a
465), el pico de la componente más grande de `u_phys` crece con la cantidad
de muestras observadas y no se estabiliza:

| muestras | pico | `u_norm` con G_out = 128 | margen |
|---|---|---|---|
| 16000 | 18.9 | 0.148 | 16.6 dB |
| 160000 | 33.4 | 0.261 | 11.7 dB |
| 1597440 | 56.7 | 0.443 | 7.1 dB |
| 15974400 (260 ms) | 110.8 | 0.866 | 1.3 dB |

Sube unos 10 dB cada dos décadas de muestras. El motivo es que el término
de orden 7 amplifica la cola de una señal OFDM, que es casi gaussiana y no
tiene pico máximo. **No existe un formato que garantice cero recorte**: lo
que se elige es una probabilidad de recorte, no un margen seguro. Con 128,
en los 16 millones de muestras no recorta ninguna.

Por eso el bloque **satura** en vez de dar la vuelta. Un pico raro que se
pase se recorta y degrada suave; con wrap aparecería con el signo cambiado,
que es catastrófico.

Con {G_OUT:g} el ruido de cuantización de la salida queda 61 dB por debajo
de la señal (-61.2 dB). La resolución sobra y el rango no, así que conviene
gastar bits de resolución en margen de pico.

### Margen de la entrada

Con `G = {G:g}`, el pico de `x_norm` sobre 320k muestras de una OFDM de 52
portadoras llegó a 0.444, o sea 7.0 dB de margen, y baja con el tiempo de
observación. Con la señal LTE no se midió. La entrada viene de un conversor
y está acotada por él.

### Cómo se calcula la magnitud (sin raíz cuadrada)

`|x|^(p-1)` se calcula a partir de `|x|^2 = I^2 + Q^2`, guardado en el
acumulador ancho, y las potencias salen por multiplicaciones sucesivas por
`|x|^2`. Como todos los órdenes son impares, los exponentes de la magnitud
son pares y la raíz cuadrada no hace falta en ningún lado. Cuesta 0.02 dB
de precisión contra el camino con raíz y ahorra una unidad de raíz
cuadrada (CORDIC o tabla más Newton-Raphson) en el hardware.

En el RTL son, por tap: dos multiplicadores para `I^2` y `Q^2`, un sumador,
y dos multiplicadores más para `|x|^4` y `|x|^6`.

- Pmax = {PMAX} (órdenes impares {ORDERS}), M = {M_TAPS} taps -> 12 coeficientes complejos.
- BO de diseño usado para entrenar: {BO_DB:.0f} dB (peor caso, mismo corner que justifica M=3/P=7).
- Orden de columnas de coeficientes (para w y para h): `for q in range(M): for p in orders:`
  i.e. col 0..3 = tap q=0 (p=1,3,5,7), col 4..7 = tap q=1, col 8..11 = tap q=2.
- w[1,0] (orden 1, tap 0) = {w10['w_q015_re']} + {w10['w_q015_im']}j -> código {w10['hex_re']} (I) — el
  1.0 matemático satura al máximo representable de Q0.15.

## Codificación de los .hex

Un valor por línea, entero de 16 bits en complemento a 2, 4 dígitos hex
mayúsculas (compatible con `$readmemh` de SystemVerilog). I y Q en archivos
separados, mismo índice de línea = misma muestra.

## Archivos

- `coeffs_w_re.hex` / `coeffs_w_im.hex` (12 líneas): coeficientes del DPD
  ya congelados en Q0.15 — los valores exactos que se escriben en
  W_COEFF_RE[12] / W_COEFF_IM[12] del mapa de registros.
- `coeffs_w.csv`: misma info en texto, con el float de entrenamiento al lado
  del valor cuantizado, para ver el error de cuantización por coeficiente.
- `functional_input_i/q.hex`, `functional_output_i/q.hex` ({n_functional} líneas):
  el comienzo de la subtrama LTE de evaluación de la tesis (semilla 456,
  la de `sw/hw_test/lte_placa.py`) aplicado al MP congelado. Caso de uso
  representativo, para simulación de forma de onda / regresión.
- `golden_vectors_functional.csv`: misma info en texto (float físico + hex).
- `directed_input_i/q.hex`, `directed_output_i/q.hex` ({n_directed} líneas):
  casos de esquina. **Son una SECUENCIA, no muestras sueltas:** la línea de
  retardo de M={M_TAPS} taps arrastra estado, así que el orden importa. Cada
  grupo empieza y termina vaciando la línea con {M_TAPS} ceros (`flush_*`),
  para que la salida de cada caso sea interpretable por sí sola.

  Los casos están armados alrededor del pico REAL de la señal, no del fondo
  de escala del formato. Con fondo de escala la salida satura casi siempre,
  y un vector que satura no distingue nada: un RTL con un signo dado vuelta
  o una columna de coeficientes mal indexada satura al mismo valor y pasa
  igual. Los grupos son:

  | grupo | qué verifica |
  |---|---|
  | `zero`, `lsb_*` | que el cero da cero y que el bit menos significativo no se pierde |
  | `imp_*` | respuesta al impulso: una muestra sola y luego ceros, así la salida recorre los taps uno por uno. Verifica el indexado de taps y el orden de las columnas de coeficientes |
  | `ramp_*_de_8` | escalera de amplitud. La contribución de cada orden crece como una potencia distinta de la magnitud, así que barrer la amplitud separa los cuatro órdenes |
  | `alt_burst_*` | ráfaga alternada: ejercita los {M_TAPS} taps a la vez con valores opuestos |
  | `sat_*` | saturación deliberada, al final para no contaminar lo anterior |
- `golden_vectors_directed.csv`: misma info, con etiqueta de caso por fila.

## Cómo se generó `u` (salida esperada)

`u_phys = apply_memory_poly_fxp(x_dpd, coeffs_q, Pmax={PMAX}, M={M_TAPS}, cfg)`,
que hace internamente: `x_norm = quantize(x_dpd/{G:g})`, evalúa el MP con
acumulador ancho, y devuelve `u_norm * {G_OUT:g}`. El valor que ve el registro
interno de salida del RTL (antes del shift externo) es
`u_norm = quantize(u_phys/{G_OUT:g})`, que es lo que está en los .hex `*_output_*`.

Los `*_input_*` están en la escala de entrada y los `*_output_*` en la de
salida. El RTL no necesita saber ninguna de las dos: trabaja siempre con
códigos de 16 bits, y el cambio de escala entre la entrada y la salida es
el corrimiento fijo de `G_out/G = {G_OUT/G:g}` que aplica dentro del bloque
antes de redondear al registro de salida.

## Reproducibilidad

Señal LTE de la tesis (`fixed_point/lte.py`, 16-QAM, FFT de 1024,
sobremuestreo 4, filtro de transmisión), con semillas fijas: las primeras
{N_TRAIN} muestras de la de semilla 123 para entrenar y la de semilla 456
para el vector funcional. El entrenamiento corre en
flotante, como en el ARM del SoC, y los coeficientes se pasan a Q0.15 al
final. Hiperparámetros GD: mu_h=0.001 sin NLMS, con respaldo NLMS
mu_h=0.05 si el modelo del PA diverge,
mu_w=0.5, ep_pa=40, ep_dpd=5, N_train_max=16000, w_reg_lambda=1e-4,
w_reg_pmin=3, y el paso efectivo del modelo del PA limitado a 1
(h_ganancia_max=1). Son los mismos 12 códigos que da el C con
`dpd_train_cfg_default` (prueba 3 de `sw/dpd_train/test/test_dpd_train.py`).

Correr `generate.py` de nuevo debería reproducir bit-exacto estos mismos
archivos.
"""
    with open(os.path.join(OUT_DIR, "config.md"), "w") as f:
        f.write(doc)


def main():
    params, coeffs_float, coeffs_q = train_coeffs()
    coeff_rows = dump_coeffs(coeffs_q, coeffs_float)
    n_func, pico_op = gen_functional_vectors(params, coeffs_q)
    n_dir = gen_directed_vectors(coeffs_q, pico_op)
    write_config_doc(n_func, n_dir, coeff_rows)
    print(f"OK -> {OUT_DIR}")
    print("  coeficientes: 12 (RE) + 12 (IM)")
    print(f"  vector funcional: {n_func} muestras")
    print(f"  vector dirigido:  {n_dir} muestras")
    print(f"  w[1,0] = {coeff_rows[0]['w_q015_re']} (código {coeff_rows[0]['hex_re']})")


if __name__ == "__main__":
    main()
