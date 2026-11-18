"""Test cocotb del bloque de redondeo y saturación.

Barre cinco combinaciones de anchos, una por cada uso previsto en el
datapath del DPD, y compara el RTL contra el espejo en Python y contra
fxpmath.
"""
import random

import cocotb
from cocotb.triggers import Timer

from fxp_ref import to_signed, limits, round_sat, round_sat_fxp

# (sufijo del puerto, IN_W, IN_FRAC, OUT_W, OUT_FRAC)
CONFIGS = [
    ("a", 32, 30, 16, 15),   # producto Q0.30 -> Q0.15
    ("b", 37, 28, 16, 15),   # acumulador s37/28 -> Q0.15
    ("c", 20, 15, 16, 15),   # sin bits que descartar, solo satura
    ("d", 24, 22, 16, 14),   # accum del LMS s24/22 -> coef s16/14
    ("e", 17, 16, 16, 15),   # un solo bit descartado, sin sticky
]


def valores(in_w: int, in_frac: int, out_w: int, out_frac: int, semilla: int):
    """Casos de prueba para una configuracion: bordes, empates y azar."""
    lo, hi = limits(in_w)
    shift = in_frac - out_frac
    v = [0, 1, -1, lo, hi, lo + 1, hi - 1]

    if shift >= 1:
        mitad = 1 << (shift - 1)
        for base in range(-4, 5):
            empate = base * (1 << shift) + mitad
            # El empate exacto y sus dos vecinos inmediatos, que son los
            # que caen justo de un lado y del otro de la mitad.
            for d in (-1, 0, 1):
                if lo <= empate + d <= hi:
                    v.append(empate + d)

    rng = random.Random(semilla)
    # Azar sobre todo el rango de entrada: ejercita la saturación.
    v += [rng.randint(lo, hi) for _ in range(300)]
    # Azar sobre el rango que si entra en la salida: ejercita el redondeo.
    dentro = 1 << (out_w - 1 + shift)
    v += [rng.randint(max(lo, -dentro), min(hi, dentro - 1)) for _ in range(300)]
    return v


@cocotb.test()
async def contra_espejo(dut):
    """RTL contra el espejo en Python, en las cinco configuraciones."""
    for sufijo, in_w, in_frac, out_w, out_frac in CONFIGS:
        din = getattr(dut, f"din_{sufijo}")
        dout = getattr(dut, f"dout_{sufijo}")

        for v in valores(in_w, in_frac, out_w, out_frac, semilla=hash(sufijo) & 0xFFFF):
            din.value = v
            await Timer(1, unit="ns")
            got = to_signed(int(dout.value), out_w)
            exp = round_sat(v, in_frac, out_w, out_frac)
            assert got == exp, (
                f"config {sufijo} (s{in_w}/{in_frac} -> s{out_w}/{out_frac}): "
                f"din={v} RTL={got} espejo={exp}"
            )


@cocotb.test()
async def contra_fxpmath(dut):
    """RTL contra el espejo contra fxpmath, los tres iguales.

    Sobre un subconjunto de valores, porque fxpmath construye un objeto por
    valor y es lento comparado con la aritmética de enteros.
    """
    for sufijo, in_w, in_frac, out_w, out_frac in CONFIGS:
        din = getattr(dut, f"din_{sufijo}")
        dout = getattr(dut, f"dout_{sufijo}")

        muestras = valores(in_w, in_frac, out_w, out_frac, semilla=7)[:60]
        for v in muestras:
            din.value = v
            await Timer(1, unit="ns")
            rtl = to_signed(int(dout.value), out_w)
            espejo = round_sat(v, in_frac, out_w, out_frac)
            fxp = round_sat_fxp(v, in_frac, out_w, out_frac)
            assert rtl == espejo == fxp, (
                f"config {sufijo}: din={v} RTL={rtl} espejo={espejo} fxpmath={fxp}"
            )
