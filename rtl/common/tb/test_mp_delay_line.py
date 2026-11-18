"""Test cocotb de la línea de retardo.

Es el primer bloque con reloj del diseno, así que además del orden de los
taps hay que verificar el reset y el enable.
"""
import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from fxp_ref import to_signed, limits

# (sufijo, W, M)
CONFIGS = [
    ("a", 16, 3),   # el uso real
    ("b", 16, 1),   # un solo tap: el lazo interno no corre
    ("c",  8, 5),   # mas taps y otro ancho
]


def leer_taps(dut, suf, w, m):
    plano_re = int(getattr(dut, f"tap_re_{suf}").value)
    plano_im = int(getattr(dut, f"tap_im_{suf}").value)
    mask = (1 << w) - 1
    return [(to_signed((plano_re >> (k*w)) & mask, w),
             to_signed((plano_im >> (k*w)) & mask, w)) for k in range(m)]


async def arrancar(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 0
    for suf, w, _ in CONFIGS:
        getattr(dut, f"x_re_{suf}").value = 0
        getattr(dut, f"x_im_{suf}").value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def reset_deja_todo_en_cero(dut):
    """Despues del reset todos los taps valen cero, en las tres config."""
    await arrancar(dut)
    for suf, w, m in CONFIGS:
        taps = leer_taps(dut, suf, w, m)
        assert taps == [(0, 0)]*m, f"config {suf}: tras reset {taps}"


@cocotb.test()
async def orden_de_los_taps(dut):
    """Una muestra sola y después ceros: tiene que recorrer los taps.

    Es el mismo patron que los casos `imp_*` de los golden vectors. Si los
    taps están al revés o sobra uno, salta acá.
    """
    await arrancar(dut)
    dut.en.value = 1
    marca = 1234

    # Un impulso...
    dut.x_re_a.value = marca
    dut.x_im_a.value = -marca
    await RisingEdge(dut.clk)
    dut.x_re_a.value = 0
    dut.x_im_a.value = 0

    # ...y después ceros. El impulso tiene que ir apareciendo en un tap
    # distinto en cada ciclo, y salirse al cabo de M.
    for esperado in range(3):
        await Timer(1, unit="ns")
        taps = leer_taps(dut, "a", 16, 3)
        for k, (re, im) in enumerate(taps):
            if k == esperado:
                assert (re, im) == (marca, -marca), \
                    f"ciclo {esperado}: el impulso deberia estar en el tap {k}, hay {(re, im)}"
            else:
                assert (re, im) == (0, 0), \
                    f"ciclo {esperado}: el tap {k} deberia estar en cero, hay {(re, im)}"
        await RisingEdge(dut.clk)

    await Timer(1, unit="ns")
    assert leer_taps(dut, "a", 16, 3) == [(0, 0)]*3, "el impulso no se fue"


@cocotb.test()
async def enable_congela_la_linea(dut):
    """Con en en cero la linea no avanza aunque cambie la entrada."""
    await arrancar(dut)
    dut.en.value = 1
    dut.x_re_a.value = 111
    dut.x_im_a.value = 222
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")
    antes = leer_taps(dut, "a", 16, 3)

    dut.en.value = 0
    for v in (333, 444, 555):
        dut.x_re_a.value = v
        dut.x_im_a.value = v
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert leer_taps(dut, "a", 16, 3) == antes, "la linea avanzo con en en cero"


@cocotb.test()
async def contra_espejo(dut):
    """Secuencia larga al azar contra una cola en Python."""
    await arrancar(dut)
    dut.en.value = 1
    rng = random.Random(5)
    colas = {suf: deque([(0, 0)]*m, maxlen=m) for suf, _, m in CONFIGS}

    for _ in range(300):
        muestras = {}
        for suf, w, m in CONFIGS:
            lo, hi = limits(w)
            re, im = rng.randint(lo, hi), rng.randint(lo, hi)
            muestras[suf] = (re, im)
            getattr(dut, f"x_re_{suf}").value = re
            getattr(dut, f"x_im_{suf}").value = im

        await RisingEdge(dut.clk)
        for suf, w, m in CONFIGS:
            colas[suf].appendleft(muestras[suf])
        await Timer(1, unit="ns")

        for suf, w, m in CONFIGS:
            taps = leer_taps(dut, suf, w, m)
            esperado = list(colas[suf])
            assert taps == esperado, f"config {suf}: RTL={taps} espejo={esperado}"
