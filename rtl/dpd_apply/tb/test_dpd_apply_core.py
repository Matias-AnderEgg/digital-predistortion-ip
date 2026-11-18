"""Verificacion del bloque de aplicación contra los golden vectors.

Esto es lo que todo lo anterior venia preparando: alimentar el RTL con las
mismas muestras que el modelo y exigir que la salida coincida bit a bit,
las 4096 veces.

Los vectores se generan con fixed_point/golden_vectors/generate.py y su
convención esta documentada en out/config.md.
"""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(ROOT, "fixed_point", "golden_vectors", "out")

X_W = 16
C_W = 16
NTERM = 12


def leer_hex(nombre, ancho=16):
    """Un valor por linea, entero con signo en complemento a dos."""
    vals = []
    with open(os.path.join(GOLDEN, nombre)) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                n = int(ln, 16)
                vals.append(n - (1 << ancho) if n & (1 << (ancho - 1)) else n)
    return vals


def to_signed(val, width):
    val &= (1 << width) - 1
    return val - (1 << width) if val & (1 << (width - 1)) else val


def empaquetar(codigos, ancho):
    """Lista de enteros con signo -> un vector plano, el indice 0 abajo."""
    plano = 0
    for k, v in enumerate(codigos):
        plano |= (v & ((1 << ancho) - 1)) << (k * ancho)
    return plano


async def preparar(dut):
    """Reloj, coeficientes y reset. Deja la linea de retardo en cero."""
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())

    w_re = leer_hex("coeffs_w_re.hex")
    w_im = leer_hex("coeffs_w_im.hex")
    assert len(w_re) == len(w_im) == NTERM, "se esperaban 12 coeficientes"
    dut.w_re.value = empaquetar(w_re, C_W)
    dut.w_im.value = empaquetar(w_im, C_W)

    dut.rst.value = 1
    dut.en.value = 0
    dut.x_re.value = 0
    dut.x_im.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(1, unit="ns")


async def correr(dut, xi, xq, ui, uq, etiquetas=None):
    """Alimenta la secuencia y compara muestra por muestra.

    Los vectores son una SECUENCIA: la línea de retardo arrastra estado, así
    que hay que alimentarlos en orden y sin saltear ninguno.

    La comparacion NO depende de la latencia del pipeline: se alimenta todo,
    se sigue el reloj un rato mas para vaciarlo, y se toman las salidas en
    los ciclos con `valid` en alto. Así cambiar la profundidad del pipeline
    (por ejemplo ACC_PER_STAGE) no obliga a tocar el testbench.
    """
    VACIADO = 64   # de sobra para cualquier profundidad razonable

    salidas = []
    dut.en.value = 1
    for n in range(len(xi) + VACIADO):
        if n < len(xi):
            dut.x_re.value = xi[n]
            dut.x_im.value = xq[n]
        else:
            dut.x_re.value = 0
            dut.x_im.value = 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")   # deja propagar la parte combinacional
        if int(dut.valid.value):
            salidas.append((to_signed(int(dut.u_re.value), X_W),
                            to_signed(int(dut.u_im.value), X_W)))
    dut.en.value = 0

    assert len(salidas) >= len(xi), (
        f"salieron {len(salidas)} muestras validas y se esperaban al menos "
        f"{len(xi)}: el pipeline no se vacio o `valid` no se levanta")

    fallas = []
    for n in range(len(xi)):
        got = salidas[n]
        esp = (ui[n], uq[n])
        if got != esp:
            quien = f" ({etiquetas[n]})" if etiquetas else ""
            fallas.append(f"  n={n}{quien}: x=({xi[n]},{xq[n]}) "
                          f"RTL={got} golden={esp}")
            if len(fallas) >= 10:
                break
    assert not fallas, (
        f"{len(fallas)} muestras distintas del golden (se muestran las primeras):\n"
        + "\n".join(fallas))


@cocotb.test()
async def vectores_dirigidos(dut):
    """52 casos de esquina: impulsos, escalera, rafaga y saturacion."""
    import csv
    await preparar(dut)
    with open(os.path.join(GOLDEN, "golden_vectors_directed.csv")) as f:
        etiquetas = [r["label"] for r in csv.DictReader(f)]
    await correr(dut,
                 leer_hex("directed_input_i.hex"), leer_hex("directed_input_q.hex"),
                 leer_hex("directed_output_i.hex"), leer_hex("directed_output_q.hex"),
                 etiquetas)


@cocotb.test()
async def vectores_funcionales(dut):
    """4096 muestras de una senal OFDM real."""
    await preparar(dut)
    await correr(dut,
                 leer_hex("functional_input_i.hex"), leer_hex("functional_input_q.hex"),
                 leer_hex("functional_output_i.hex"), leer_hex("functional_output_q.hex"))


@cocotb.test()
async def valid_con_en_espaciado(dut):
    """Con `en` espaciado, `valid` marca una muestra nueva y una sola vez.

    `en` es habilitación de reloj, así que entre pulso y pulso la salida
    repite el valor anterior. Si `valid` fuera solo la bandera de pipeline
    lleno, quedaría en alto en esos ciclos y el que instancia el bloque
    tendría que hacer `valid & en` por su cuenta.

    El test alimenta los mismos vectores dirigidos pero con `en` uno de cada
    tres ciclos. Si `valid` volviera a ser un nivel, saldrian del orden de
    tres veces mas muestras y la comparacion se correría enseguida.
    """
    HUECO = 3    # un pulso de `en` cada tres ciclos
    VACIADO = 64  # ciclos de sobra para vaciar el pipeline

    await preparar(dut)
    xi = leer_hex("directed_input_i.hex")
    xq = leer_hex("directed_input_q.hex")
    ui = leer_hex("directed_output_i.hex")
    uq = leer_hex("directed_output_q.hex")

    salidas = []
    n = 0
    for ciclo in range((len(xi) + VACIADO) * HUECO):
        pulso = (ciclo % HUECO == 0)
        if pulso:
            dut.x_re.value = xi[n] if n < len(xi) else 0
            dut.x_im.value = xq[n] if n < len(xi) else 0
            n += 1
        dut.en.value = 1 if pulso else 0
        await RisingEdge(dut.clk)
        await Timer(1, unit="ns")
        if int(dut.valid.value):
            salidas.append((to_signed(int(dut.u_re.value), X_W),
                            to_signed(int(dut.u_im.value), X_W)))
    dut.en.value = 0

    assert len(salidas) >= len(xi), (
        f"salieron {len(salidas)} muestras validas y se esperaban al menos "
        f"{len(xi)}: `valid` no esta marcando todas")

    fallas = []
    for k in range(len(xi)):
        if salidas[k] != (ui[k], uq[k]):
            fallas.append(f"  n={k}: x=({xi[k]},{xq[k]}) "
                          f"RTL={salidas[k]} golden={(ui[k], uq[k])}")
            if len(fallas) >= 10:
                break
    assert not fallas, (
        f"{len(fallas)} muestras distintas del golden con `en` espaciado "
        f"(se muestran las primeras):\n" + "\n".join(fallas))
