"""Test cocotb del retardo de N ciclos."""
import random
from collections import deque

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

# (sufijo, W, N)
CONFIGS = [("a", 16, 0), ("b", 16, 1), ("c", 16, 5), ("d", 74, 3)]


async def arrancar(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    dut.rst.value = 1
    dut.en.value = 1
    for suf, _, _ in CONFIGS:
        getattr(dut, f"d_{suf}").value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


@cocotb.test()
async def retarda_lo_que_dice(dut):
    """Cada configuracion entrega la muestra de N ciclos atras."""
    await arrancar(dut)
    rng = random.Random(1)
    # La cadena tiene N registros. Después del flanco, el primero guarda el
    # valor que había en la entrada y la salida es el del registro N-1, o
    # sea el valor de N-1 flancos atrás. Con N=1 la salida es el valor que
    # se acaba de meter, no cero.
    colas = {suf: deque([0]*n, maxlen=n) for suf, _, n in CONFIGS if n > 0}

    for _ in range(200):
        vals = {}
        for suf, w, n in CONFIGS:
            v = rng.randrange(1 << w)
            vals[suf] = v
            getattr(dut, f"d_{suf}").value = v

        await Timer(1, unit="ns")
        for suf, w, n in CONFIGS:
            if n == 0:
                assert int(getattr(dut, f"q_{suf}").value) == vals[suf], \
                    f"config {suf}: con N=0 tiene que ser un cable"

        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        for suf, w, n in CONFIGS:
            if n == 0:
                continue
            colas[suf].appendleft(vals[suf])
            esperado = colas[suf][n-1]
            got = int(getattr(dut, f"q_{suf}").value)
            assert got == esperado, \
                f"config {suf} (N={n}): salio {got}, se esperaba {esperado}"


@cocotb.test()
async def enable_congela(dut):
    """Con en en cero la cadena no avanza."""
    await arrancar(dut)
    dut.d_c.value = 0xABCD
    await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(1, unit="ns")
    antes = int(dut.q_c.value)
    for v in (1, 2, 3, 4, 5, 6):
        dut.d_c.value = v
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        assert int(dut.q_c.value) == antes, "la cadena avanzo con en en cero"
