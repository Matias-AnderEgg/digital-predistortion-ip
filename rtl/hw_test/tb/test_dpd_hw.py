"""Lo que va a la placa: IP mas reproductor, contra los golden vectors.

Hace lo mismo que va a hacer el programa del ARM: carga los coeficientes por
AXI-Lite, carga la BRAM de entrada, arranca el reproductor por el GPIO,
espera `done` y compara la BRAM de salida contra el golden.
"""
import os
import sys

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "..", "dpd_top", "tb"))

from axil import AxiLite, MUESTREO_NS  # noqa: E402
from test_dpd_apply_top import (  # noqa: E402
    ADDR_CTRL, BIT_COMMIT, MASK_C, cargar, coeficientes_golden, leer_hex,
)

START = 1 << 31
LATENCY = 29          # mp_pkg::mp_latency(3, 4, 1)


async def arrancar(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    bus = AxiLite(dut)
    bus.reposo()
    dut.ctrl.value = 0
    dut.in_a_we.value = 0
    dut.in_a_addr.value = 0
    dut.in_a_din.value = 0
    dut.out_a_addr.value = 0
    dut.aresetn.value = 0
    for _ in range(3):
        await RisingEdge(dut.clk)
    dut.aresetn.value = 1
    await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")
    return bus


async def cargar_entrada(dut, xi, xq):
    """Lo que hace el ARM escribiendo la BRAM de entrada por el puerto A."""
    for k, (re, im) in enumerate(zip(xi, xq)):
        dut.in_a_addr.value = k
        dut.in_a_din.value = ((im & MASK_C) << 16) | (re & MASK_C)
        dut.in_a_we.value = 1
        await RisingEdge(dut.clk)
    dut.in_a_we.value = 0


async def leer_salida(dut, n):
    out = []
    for k in range(n):
        dut.out_a_addr.value = k
        await RisingEdge(dut.clk)
        await Timer(MUESTREO_NS, unit="ns")
        w = int(dut.out_a_dout.value)
        re, im = w & 0xFFFF, w >> 16
        out.append((re - 0x10000 if re & 0x8000 else re, im - 0x10000 if im & 0x8000 else im))
    return out


async def correr(dut, n, max_ciclos=20000):
    """Arranca, espera done y lo baja. Devuelve los ciclos que tardo."""
    dut.ctrl.value = START | n
    ciclos = 0
    while True:
        await RisingEdge(dut.clk)
        await Timer(MUESTREO_NS, unit="ns")
        ciclos += 1
        if int(dut.estado.value) & 1:
            break
        assert ciclos < max_ciclos, "el reproductor nunca termino"
    dut.ctrl.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")
    assert int(dut.estado.value) & 0b11 == 0, "done no bajo al soltar start"
    return ciclos


def comparar(nombre, obtenido, esperado):
    malas = [(k, o, e) for k, (o, e) in enumerate(zip(obtenido, esperado)) if o != e]
    assert not malas, (f"{nombre}: {len(malas)} de {len(esperado)} muestras distintas, "
                       f"primeras: {malas[:5]}")


def vectores(tipo):
    xi = leer_hex(f"{tipo}_input_i.hex")
    xq = leer_hex(f"{tipo}_input_q.hex")
    esp = list(zip(leer_hex(f"{tipo}_output_i.hex"), leer_hex(f"{tipo}_output_q.hex")))
    return xi, xq, esp


@cocotb.test()
async def test_golden_en_corridas_seguidas(dut):
    """Funcional, dirigido y funcional otra vez, sin reset en el medio.

    La segunda y la tercera corrida arrancan con el pipeline lleno de lo que
    dejó la anterior, que es justo lo que el reproductor tiene que manejar
    contando pulsos en vez de mirar `valid`.
    """
    bus = await arrancar(dut)
    await cargar(bus, coeficientes_golden())
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)

    for tipo in ["functional", "directed", "functional"]:
        xi, xq, esp = vectores(tipo)
        await cargar_entrada(dut, xi, xq)
        ciclos = await correr(dut, len(xi))
        comparar(tipo, await leer_salida(dut, len(xi)), esp)
        dut._log.info(f"{tipo}: {len(xi)} muestras iguales al golden, {ciclos} ciclos")


@cocotb.test()
async def test_tarda_n_mas_latencia(dut):
    """Una muestra por ciclo: N + LATENCY y un par de ciclos de registro."""
    bus = await arrancar(dut)
    await cargar(bus, coeficientes_golden())
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    xi, xq, _ = vectores("functional")
    await cargar_entrada(dut, xi, xq)
    ciclos = await correr(dut, len(xi))
    assert len(xi) + LATENCY - 1 <= ciclos <= len(xi) + LATENCY + 3, \
        f"tardo {ciclos} ciclos para {len(xi)} muestras"


@cocotb.test()
async def test_cero_muestras_termina_enseguida(dut):
    await arrancar(dut)
    ciclos = await correr(dut, 0, max_ciclos=5)
    assert ciclos <= 2
