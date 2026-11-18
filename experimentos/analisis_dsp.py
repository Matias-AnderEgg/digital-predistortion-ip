"""Cuenta de los DSP48 del camino de datos, según el ancho interno.

Enumera cada multiplicación del camino de datos con sus anchos y cuenta en
cuántos DSP48E1 se parte: el multiplicador del primitivo es de 25 por 18
bits con signo, y un operando más ancho se divide en trozos. No reemplaza a
la síntesis, que da 162 con el ancho de 37 bits (rtl/synth/README.md).

Respalda la estimación de la sección de compromisos entre recursos y
desempeño: 174 DSP48 con 37 bits, 84 con 25 y 78 con 18.

Uso:
    python experimentos/analisis_dsp.py
"""
from math import ceil

# Multiplicador del DSP48E1, con signo.
PUERTO_A, PUERTO_B = 25, 18

# Parámetros de la instancia, los de dpd_apply_core.
X_W, C_W, A_W = 16, 16, 37
M, NORD = 3, 4
NPOW = NORD - 1


def trozos(ancho, ancho_puerto):
    """En cuántos trozos se parte un operando con signo: cada trozo aporta
    ancho_puerto - 1 bits, porque uno es el de signo."""
    return max(1, ceil((ancho - 1) / (ancho_puerto - 1)))


def dsps(a, b):
    """DSP48 de una multiplicación de a por b bits, con la mejor asignación
    de los operandos a los dos puertos."""
    return min(trozos(a, PUERTO_A) * trozos(b, PUERTO_B),
               trozos(b, PUERTO_A) * trozos(a, PUERTO_B))


def multiplicaciones(a_w):
    """Las multiplicaciones de una rama (un retardo), con sus anchos. a_w es
    el ancho del camino de la magnitud y de la base."""
    m = [("cuadrador de I", X_W, X_W), ("cuadrador de Q", X_W, X_W)]
    m += [(f"potencia {2 * (k + 2)}", a_w, a_w) for k in range(NPOW - 1)]
    for j in range(1, NORD):
        p = 2 * j + 1
        m += [(f"base orden {p}, I", X_W, a_w), (f"base orden {p}, Q", X_W, a_w)]
    for j in range(NORD):
        p = 2 * j + 1
        m += [(f"coeficiente orden {p}, parcial {k + 1}", C_W, a_w) for k in range(4)]
    return m


def total(a_w):
    return M * sum(dsps(a, b) for _, a, b in multiplicaciones(a_w))


def main():
    print(f"M = {M}, {NORD} órdenes, muestra de {X_W} bits, coeficiente de {C_W}, "
          f"camino interno de {A_W}, DSP48E1 de {PUERTO_A} x {PUERTO_B}\n")
    print(f"  {'multiplicación':34} {'anchos':>10} {'DSP48':>6}")
    rama = multiplicaciones(A_W)
    for nombre, a, b in rama:
        print(f"  {nombre:34} {a:3} x {b:<4} {dsps(a, b):6}")
    print(f"\n  por rama: {len(rama)} multiplicaciones, {total(A_W) // M} DSP48")
    print(f"  {M} ramas: {len(rama) * M} multiplicaciones, {total(A_W)} DSP48\n")

    print("  ancho interno   DSP48")
    for a_w in (37, 32, 27, 25, 22, 18):
        print(f"  {a_w:13}   {total(a_w):5}")


if __name__ == "__main__":
    main()
