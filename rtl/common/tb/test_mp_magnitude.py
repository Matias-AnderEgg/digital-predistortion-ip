"""Test cocotb del bloque de magnitud.

Además del espejo y de fxpmath, hay un tercer test que compara contra el
modelo de punto fijo del proyecto corriendo sobre las muestras reales de
los golden vectors. Ese es el que de verdad ata el bloque al modelo: los
otros dos verifican que el RTL hace lo que dice el espejo, y este verifica
que el espejo dice lo que hace el modelo.
"""
import os
import random
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from fxp_ref import to_signed, limits, round_sat_fxp, mp_magnitude

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# (sufijo, X_W, X_FRAC, A_W, A_FRAC, NPOW)
CONFIGS = [
    ("a", 16, 15, 37, 28, 3),   # el uso real en el DPD
    ("b", 16, 15, 37, 28, 1),   # una sola potencia (Pmax=3)
    ("c", 16, 15, 20, 18, 3),   # acumulador angosto: la saturación se alcanza
    ("d", 12, 11, 32, 20, 3),   # entrada mas angosta
]

# Igual que "a" pero con pipeline: (config, PIPE). La latencia es NPOW*PIPE.
# La "f" es la que usa el datapath.
CONFIGS_PIPE = [
    (("e", 16, 15, 37, 28, 3), 1),
    (("f", 16, 15, 37, 28, 3), 3),
]


async def arrancar_reloj(dut):
    """Reloj y reset. Los modos combinacionales lo ignoran."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def aplicar(dut, suf, x_re, x_im, a_w, npow):
    getattr(dut, f"x_re_{suf}").value = x_re
    getattr(dut, f"x_im_{suf}").value = x_im
    await Timer(1, unit="ns")
    plano = int(getattr(dut, f"pot_{suf}").value)
    mascara = (1 << a_w) - 1
    return [to_signed((plano >> (k * a_w)) & mascara, a_w) for k in range(npow)]


def valores(x_w, semilla, n=250):
    lo, hi = limits(x_w)
    v = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
         (hi, 0), (lo, 0), (0, hi), (0, lo),
         (hi, hi), (lo, lo), (hi, lo)]          # |x| = sqrt(2), el máximo
    rng = random.Random(semilla)
    v += [(rng.randint(lo, hi), rng.randint(lo, hi)) for _ in range(n)]
    v += [(rng.randint(-64, 64), rng.randint(-64, 64)) for _ in range(n)]
    return v


@cocotb.test()
async def contra_espejo(dut):
    """RTL contra el espejo en Python, en las cuatro configuraciones."""
    for suf, xw, xf, aw, af, npow in CONFIGS:
        for x_re, x_im in valores(xw, semilla=ord(suf)):
            got = await aplicar(dut, suf, x_re, x_im, aw, npow)
            exp = mp_magnitude(x_re, x_im, xf, aw, af, npow)
            assert got == exp, (
                f"config {suf}: x=({x_re},{x_im}) RTL={got} espejo={exp}"
            )


@cocotb.test()
async def contra_fxpmath(dut):
    """Las reducciones de formato de la cadena, verificadas con fxpmath."""
    for suf, xw, xf, aw, af, npow in CONFIGS:
        for x_re, x_im in valores(xw, semilla=11, n=25):
            got = await aplicar(dut, suf, x_re, x_im, aw, npow)
            a = [round_sat_fxp(x_re*x_re + x_im*x_im, 2*xf, aw, af)]
            for k in range(1, npow):
                a.append(round_sat_fxp(a[k-1]*a[0], 2*af, aw, af))
            assert got == a, (
                f"config {suf}: x=({x_re},{x_im}) RTL={got} fxpmath={a}"
            )


@cocotb.test()
async def contra_el_modelo(dut):
    """RTL contra el modelo del proyecto, sobre las muestras de los golden.

    Reproduce el camino de la magnitud de `apply_memory_poly_fxp` con
    magnitude_mode='squared', llamando a la misma función `quantize` que usa
    el modelo, y lo compara contra el RTL muestra por muestra.
    """
    import numpy as np
    from fixed_point import fxp_format as fxp

    cfg = fxp.make_normalized_config(
        n_word_data=16, n_word_coef=16, n_word_coef_accum=24, coef_n_int=0,
        signal_scale=8.0, output_scale=128.0, magnitude_mode="squared",
    )
    aw, af = cfg.accum.n_word, cfg.accum.n_frac
    assert (aw, af) == (37, 28), "la config A del testbench asume s37/28"

    ruta = os.path.join(ROOT, "fixed_point", "golden_vectors", "out")

    def leer(nombre):
        vals = []
        with open(os.path.join(ruta, nombre)) as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    n = int(ln, 16)
                    vals.append(n - (1 << 16) if n & 0x8000 else n)
        return vals

    xi = leer("functional_input_i.hex")[:400]
    xq = leer("functional_input_q.hex")[:400]

    for x_re, x_im in zip(xi, xq):
        got = await aplicar(dut, "a", x_re, x_im, aw, 3)

        # El camino de la magnitud, tal cual lo hace el modelo.
        x = (x_re + 1j*x_im) / 2**15
        a2 = fxp.quantize(np.array([x.real**2 + x.imag**2]), cfg.accum, cfg)
        a4 = fxp.quantize(a2*a2, cfg.accum, cfg)
        a6 = fxp.quantize(a4*a2, cfg.accum, cfg)
        esperado = [int(round(float(v[0])*2**af)) for v in (a2, a4, a6)]

        assert got == esperado, (
            f"x=({x_re},{x_im}) RTL={got} modelo={esperado}"
        )


@cocotb.test()
async def con_pipeline_da_lo_mismo(dut):
    """Con pipeline tiene que dar exactamente los mismos valores que sin el.

    Si difieren, la alineación entre potencias esta mal: las etapas
    profundas se calculan mas tarde y hay que retrasar las de arriba para
    que salgan juntas. Es el tipo de error que no se ve mirando el código.
    """
    await arrancar_reloj(dut)
    for (suf, xw, xf, aw, af, npow), pipe in CONFIGS_PIPE:
        for x_re, x_im in valores(xw, semilla=42, n=120):
            getattr(dut, f"x_re_{suf}").value = x_re
            getattr(dut, f"x_im_{suf}").value = x_im
            # Mantener la entrada durante toda la latencia deja al pipeline
            # entero con valores derivados de la misma muestra.
            for _ in range(npow * pipe):
                await RisingEdge(dut.clk)
            await Timer(1, unit="ns")

            got = await aplicar(dut, suf, x_re, x_im, aw, npow)
            exp = mp_magnitude(x_re, x_im, xf, aw, af, npow)
            assert got == exp, (
                f"PIPE={pipe} x=({x_re},{x_im}) RTL={got} espejo={exp}")


@cocotb.test()
async def pipeline_en_flujo(dut):
    """Una secuencia continua, leyendo con la latencia correspondiente.

    Verifica que el bloque procesa una muestra por ciclo y no solo que da
    bien cuando la entrada se queda quieta.
    """
    await arrancar_reloj(dut)
    for (suf, xw, xf, aw, af, npow), pipe in CONFIGS_PIPE:
        lat = npow * pipe
        casos = valores(xw, semilla=7, n=200)

        esperados = []
        for n, (x_re, x_im) in enumerate(casos):
            getattr(dut, f"x_re_{suf}").value = x_re
            getattr(dut, f"x_im_{suf}").value = x_im
            esperados.append(mp_magnitude(x_re, x_im, xf, aw, af, npow))
            await RisingEdge(dut.clk)
            await Timer(1, unit="ns")
            if n >= lat - 1:
                got = await aplicar(dut, suf, x_re, x_im, aw, npow)
                k = n - (lat - 1)
                assert got == esperados[k], (
                    f"PIPE={pipe} muestra {k}: RTL={got} espejo={esperados[k]}"
                )
