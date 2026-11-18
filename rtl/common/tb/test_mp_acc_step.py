"""Test cocotb del paso de acumulacion del Memory Polynomial."""
import random

import cocotb
from cocotb.triggers import Timer

from fxp_ref import to_signed, limits, acc_step

# (sufijo, A_W, A_FRAC, MAX_SHIFT)
CONFIGS = [
    ("a", 37, 28, 18),   # el uso real
    ("b", 20, 15,  6),   # angosto: la saturación se alcanza seguido
    ("c", 37, 28,  0),   # sin corrimiento (caso G=1)
]


async def aplicar(dut, suf, acc_re, acc_im, t_re, t_im, sh, a_w):
    getattr(dut, f"acc_in_re_{suf}").value = acc_re
    getattr(dut, f"acc_in_im_{suf}").value = acc_im
    getattr(dut, f"term_re_{suf}").value = t_re
    getattr(dut, f"term_im_{suf}").value = t_im
    getattr(dut, f"shift_{suf}").value = sh
    await Timer(1, unit="ns")
    return (to_signed(int(getattr(dut, f"acc_out_re_{suf}").value), a_w),
            to_signed(int(getattr(dut, f"acc_out_im_{suf}").value), a_w))


def casos(a_w, max_shift, semilla, n=200):
    lo, hi = limits(a_w)
    v = []
    # Bordes: el acumulador y el termino en los extremos, con y sin
    # corrimiento. Son los que fuerzan la saturación en los dos sentidos.
    for sh in {0, max_shift // 2, max_shift}:
        v += [(0, 0, 0, 0, sh), (hi, hi, hi, hi, sh), (lo, lo, lo, lo, sh),
              (hi, lo, 1, -1, sh), (lo, hi, -1, 1, sh),
              (0, 0, hi, lo, sh), (hi, hi, lo, lo, sh)]
    rng = random.Random(semilla)
    for _ in range(n):
        v.append((rng.randint(lo, hi), rng.randint(lo, hi),
                  rng.randint(lo, hi), rng.randint(lo, hi),
                  rng.randint(0, max_shift)))
    # Valores chicos: nada satura y se verifica la suma pura.
    for _ in range(n):
        v.append((rng.randint(-10000, 10000), rng.randint(-10000, 10000),
                  rng.randint(-10000, 10000), rng.randint(-10000, 10000),
                  rng.randint(0, max_shift)))
    return v


@cocotb.test()
async def contra_espejo(dut):
    """RTL contra el espejo, en las tres configuraciones."""
    for suf, aw, af, msh in CONFIGS:
        for acc_re, acc_im, t_re, t_im, sh in casos(aw, msh, semilla=ord(suf)):
            got = await aplicar(dut, suf, acc_re, acc_im, t_re, t_im, sh, aw)
            exp = (acc_step(acc_re, t_re, sh, aw, af),
                   acc_step(acc_im, t_im, sh, aw, af))
            assert got == exp, (
                f"config {suf}: acc=({acc_re},{acc_im}) term=({t_re},{t_im}) "
                f"shift={sh} RTL={got} espejo={exp}"
            )


@cocotb.test()
async def satura_en_los_dos_sentidos(dut):
    """Que la saturacion exista de verdad y no sea un camino muerto."""
    aw, msh = 20, 6
    hi = limits(aw)[1]
    lo = limits(aw)[0]

    arriba = await aplicar(dut, "b", hi, hi, hi, hi, msh, aw)
    assert arriba == (hi, hi), f"deberia saturar arriba, dio {arriba}"

    abajo = await aplicar(dut, "b", lo, lo, lo, lo, msh, aw)
    assert abajo == (lo, lo), f"deberia saturar abajo, dio {abajo}"

    # Y que sin saturar sume bien: el corrimiento de verdad multiplica.
    justo = await aplicar(dut, "b", 0, 0, 3, -3, 4, aw)
    assert justo == (48, -48), f"3<<4 deberia dar 48, dio {justo}"
