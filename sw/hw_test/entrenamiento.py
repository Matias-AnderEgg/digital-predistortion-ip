"""Funciones comunes para pasar datos entre la Mac y el IP de la placa.

Empaquetan y desempaquetan las muestras en palabras de 32 bits, {im, re},
calculan la salida que tiene que dar el IP según el modelo de punto fijo y
escriben arreglos de C. Las usan lte_placa.py, sw/lazo_simulado/lazo_lte.py
y las figuras de la placa. Trae además los módulos de la prueba del C
(T, GV, dpd y fxp).
"""
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "sw", "dpd_train", "test"))

import test_dpd_train as T  # noqa: E402  (trae generate.py, dpd y el PA)

GV, dpd, fxp = T.GV, T.dpd, T.fxp


def empaquetar(re, im):
    return ((np.asarray(im, np.int64) & 0xFFFF) << 16) | (np.asarray(re, np.int64) & 0xFFFF)


def desempaquetar(w):
    w = np.asarray(w, np.int64)
    re, im = w & 0xFFFF, (w >> 16) & 0xFFFF
    re = np.where(re & 0x8000, re - 0x10000, re)
    im = np.where(im & 0x8000, im - 0x10000, im)
    return re, im


def salida_esperada(x, w_q):
    """Codigos de salida del IP, como en generate.gen_functional_vectors."""
    u = dpd.apply_memory_poly_fxp(x, w_q, Pmax=GV.PMAX, M=GV.M_TAPS, cfg=GV.CFG)
    return GV.to_code(fxp.quantize(u / GV.G_OUT, GV.CFG.data, GV.CFG))


def arreglo(fh, tipo, nombre, valores, fmt):
    fh.write(f"static const {tipo} {nombre}[{len(valores)}] = {{\n")
    for k in range(0, len(valores), 6):
        fh.write("    " + ", ".join(fmt(v) for v in valores[k:k + 6]) + ",\n")
    fh.write("};\n\n")
