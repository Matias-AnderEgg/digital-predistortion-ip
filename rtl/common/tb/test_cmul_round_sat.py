"""Test cocotb del multiplicador complejo.

Cuatro combinaciones de anchos, una de ellas la que usa el DPD de verdad.
Compara contra el espejo en Python y contra fxpmath.
"""
import random

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from fxp_ref import to_signed, limits, cmul_round_sat, cmul_round_sat_fxp

# (sufijo, A_W, A_FRAC, B_W, B_FRAC, O_W, O_FRAC)
CONFIGS = [
    ("a", 16, 15, 37, 28, 37, 28),   # el uso real: coef x base -> acumulador
    ("b", 16, 15, 16, 15, 16, 15),   # todo Q0.15, el que mas satura
    ("c", 16, 14, 16, 15, 24, 22),   # anchos y comas distintos en los 3 lados
    ("d", 16, 15, 37, 28, 16, 15),   # salida angosta: redondea y satura
]


def valores(a_w, b_w, semilla, n=250):
    """Pares (a, b) complejos: bordes, casos chicos y azar."""
    a_lo, a_hi = limits(a_w)
    b_lo, b_hi = limits(b_w)
    v = [
        (0, 0, 0, 0),
        (a_hi, 0, b_hi, 0), (a_lo, 0, b_lo, 0),
        (0, a_hi, 0, b_hi), (0, a_lo, 0, b_lo),
        (a_hi, a_hi, b_hi, b_hi),          # el peor caso de módulo
        (a_lo, a_lo, b_lo, b_lo),
        (a_hi, a_lo, b_lo, b_hi),
        (1, 0, 1, 0), (1, 1, 1, 1), (-1, -1, -1, -1),
        (a_hi, 0, 1, 0), (1, 0, b_hi, 0),
    ]
    rng = random.Random(semilla)
    for _ in range(n):
        v.append((rng.randint(a_lo, a_hi), rng.randint(a_lo, a_hi),
                  rng.randint(b_lo, b_hi), rng.randint(b_lo, b_hi)))
    # Valores chicos: el producto no satura y todo lo decide el redondeo.
    for _ in range(n):
        v.append((rng.randint(-1000, 1000), rng.randint(-1000, 1000),
                  rng.randint(-1000, 1000), rng.randint(-1000, 1000)))
    return v


async def aplicar(dut, suf, a_re, a_im, b_re, b_im, o_w):
    getattr(dut, f"a_re_{suf}").value = a_re
    getattr(dut, f"a_im_{suf}").value = a_im
    getattr(dut, f"b_re_{suf}").value = b_re
    getattr(dut, f"b_im_{suf}").value = b_im
    await Timer(1, unit="ns")
    return (to_signed(int(getattr(dut, f"o_re_{suf}").value), o_w),
            to_signed(int(getattr(dut, f"o_im_{suf}").value), o_w))


@cocotb.test()
async def contra_espejo(dut):
    """RTL contra el espejo en Python, en las cuatro configuraciones."""
    for suf, aw, af, bw, bf, ow, of in CONFIGS:
        for a_re, a_im, b_re, b_im in valores(aw, bw, semilla=ord(suf)):
            got = await aplicar(dut, suf, a_re, a_im, b_re, b_im, ow)
            exp = cmul_round_sat(a_re, a_im, b_re, b_im, af, bf, ow, of)
            assert got == exp, (
                f"config {suf}: a=({a_re},{a_im}) b=({b_re},{b_im}) "
                f"RTL={got} espejo={exp}"
            )


@cocotb.test()
async def contra_fxpmath(dut):
    """RTL contra el espejo contra fxpmath, los tres iguales."""
    for suf, aw, af, bw, bf, ow, of in CONFIGS:
        for a_re, a_im, b_re, b_im in valores(aw, bw, semilla=3, n=25):
            got = await aplicar(dut, suf, a_re, a_im, b_re, b_im, ow)
            esp = cmul_round_sat(a_re, a_im, b_re, b_im, af, bf, ow, of)
            fxp = cmul_round_sat_fxp(a_re, a_im, b_re, b_im, af, bf, ow, of)
            assert got == esp == fxp, (
                f"config {suf}: a=({a_re},{a_im}) b=({b_re},{b_im}) "
                f"RTL={got} espejo={esp} fxpmath={fxp}"
            )


@cocotb.test()
async def con_pipeline_en_flujo(dut):
    """Con pipeline, una muestra por ciclo y cada resultado PIPE ciclos después.

    Es la configuración "a" con los registros antes del redondeo con que va
    en el datapath. Tiene que dar lo mismo que el espejo, corrido PIPE ciclos.
    """
    PIPE = 2
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 1
    await RisingEdge(dut.clk)
    dut.rst.value = 0

    _, aw, af, bw, bf, ow, of = CONFIGS[0]
    esperados = []
    for n, (a_re, a_im, b_re, b_im) in enumerate(valores(aw, bw, semilla=11, n=150)):
        dut.a_re_e.value = a_re
        dut.a_im_e.value = a_im
        dut.b_re_e.value = b_re
        dut.b_im_e.value = b_im
        esperados.append(cmul_round_sat(a_re, a_im, b_re, b_im, af, bf, ow, of))
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if n >= PIPE - 1:
            got = (to_signed(int(dut.o_re_e.value), ow),
                   to_signed(int(dut.o_im_e.value), ow))
            k = n - (PIPE - 1)
            assert got == esperados[k], (
                f"muestra {k}: RTL={got} espejo={esperados[k]}")
