"""Margen del formato interno Q8.28 del núcleo, con la señal LTE de la tesis.

Las cuentas internas del camino de datos usan s37/28 (Q8.28): los 16 bits de
la señal más 8 bits enteros y 13 fraccionarios de guarda. El script mide qué
pide cada parte:

  1. Comprueba que la copia del camino de datos de este script, con el
     formato interno variable, da bit a bit lo mismo que el modelo de punto
     fijo (fixed_point/dpd.py con generate.CFG) y que la salida esperada del
     RTL, sobre los vectores funcionales.
  2. A -9 y +3 dB entrena el predistorsionador como el barrido
     (sys_with_acpr/barrido_mp.py, M = 3, P = 7) y anota el pico de cada
     etapa antes de redondear, y el error de los registros de 16 bits y del
     formato interno contra el predistorsionador en punto flotante.
  3. Barre los bits de guarda enteros (con 13 fraccionarios) y los
     fraccionarios (con 8 enteros) y mide, sobre la cadena de
     figuras/cuantizacion_zf.py, las saturaciones, el error interno, la EVM
     con el receptor escalar y el peor ACPR adyacente.

El error interno es el de la salida del predistorsionador con el formato
dado, contra la misma cuenta sin redondear las cuentas internas (sólo la
entrada, los coeficientes y la salida en Q0.15), relativo a la potencia de
la salida.

Respalda la Sec. 7.2.3: los 28 bits fraccionarios hacen falta por el
corrimiento de 18 bits del término de orden 7, y de los 8 enteros alcanza
con 3 para esta señal.

Uso: PYTHONDONTWRITEBYTECODE=1 python experimentos/formato_interno.py
(un minuto; deja experimentos/resultados/formato_interno.txt)
"""
import contextlib
import csv
import io
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "fixed_point", "golden_vectors"))
sys.path.insert(0, RAIZ)
os.environ.setdefault("MPLBACKEND", "Agg")

import generate as GV  # noqa: E402
from fixed_point.dpd import apply_memory_poly_fxp  # noqa: E402
from sys_with_acpr import dpd as D  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, _bits  # noqa: E402
from sys_with_acpr.main import eval_chain, generate_ofdm_signal, simulate_with_dpd  # noqa: E402

M, P, ORDENES = 3, 7, (1, 3, 5, 7)
G, G_OUT = 8.0, 128.0
RESULTADO = os.path.join(AQUI, "resultados", "formato_interno.txt")


def q(v, w, f, sat=None):
    """Redondeo al más cercano, con empate al par, y saturación a s(w)/(f),
    como fxp_format.quantize con rounding="around"."""
    if w is None:
        return v
    if np.iscomplexobj(v):
        return q(v.real, w, f, sat) + 1j * q(v.imag, w, f, sat)
    c = np.rint(np.asarray(v) * 2.0 ** f)
    lo, hi = -2.0 ** (w - 1), 2.0 ** (w - 1) - 1
    if sat is not None:
        sat[0] += int(np.count_nonzero((c < lo) | (c > hi)))
    return np.clip(c, lo, hi) / 2.0 ** f


def pico(v):
    v = np.asarray(v)
    if np.iscomplexobj(v):
        return float(max(np.max(np.abs(v.real)), np.max(np.abs(v.imag))))
    return float(np.max(np.abs(v)))


def nucleo(x, w, guarda_ent=8, guarda_frac=13, interno=True, picos=None):
    """El camino de datos de apply_memory_poly_fxp con GV.CFG, con el
    formato interno s(16 + guarda_ent + guarda_frac)/(15 + guarda_frac).
    interno=False no redondea las cuentas internas. picos: dict donde se
    guarda el pico de cada etapa antes de redondear. Devuelve la salida y
    cuántos valores saturaron."""
    if interno:
        aw, af = 16 + guarda_ent + guarda_frac, 15 + guarda_frac
    else:
        aw, af = None, None
    sat = [0]

    def ancho(v, etapa):
        if picos is not None:
            picos[etapa] = max(picos.get(etapa, 0.0), pico(v))
        return q(v, aw, af, sat)

    x = q(np.asarray(x, complex).reshape(-1) / G, 16, 15)
    w = q(np.asarray(w, complex).reshape(-1), 16, 15)
    y = np.zeros(len(x), complex)
    col = 0
    for m in range(M):
        xm = np.zeros(len(x), complex)
        xm[m:] = x[:len(x) - m]
        a2 = ancho(xm.real ** 2 + xm.imag ** 2, "|x|^2")
        pot, k = np.ones(len(x)), 0
        for p in ORDENES:
            while k < (p - 1) // 2:
                k += 1
                pot = ancho(pot * a2, f"|x|^{2 * k}")
            base = ancho(xm * pot, f"base, orden {p}")
            prod = ancho(w[col] * base, f"coeficiente por base, orden {p}")
            y = ancho(y + prod * G ** (p - 1), "acumulador")
            col += 1
    return q(y * (G / G_OUT), 16, 15) * G_OUT, sat[0]


def db(e, ref):
    return 10 * np.log10(np.mean(np.abs(e) ** 2) / np.mean(np.abs(ref) ** 2))


def leer_hex(nombre):
    with open(os.path.join(RAIZ, "fixed_point", "golden_vectors", "out", nombre)) as f:
        v = np.array([int(ln, 16) for ln in f if ln.strip()])
    return np.where(v & 0x8000, v - (1 << 16), v) / 2 ** 15


def comprobar():
    x = (leer_hex("functional_input_i.hex") + 1j * leer_hex("functional_input_q.hex")) * G
    with open(os.path.join(RAIZ, "fixed_point", "golden_vectors", "out", "coeffs_w.csv")) as f:
        w = np.array([complex(float(r["w_float_re"]), float(r["w_float_im"])) for r in csv.DictReader(f)])
    rtl = (leer_hex("functional_output_i.hex") + 1j * leer_hex("functional_output_q.hex")) * G_OUT
    u_modelo = apply_memory_poly_fxp(x, w, Pmax=P, M=M, cfg=GV.CFG)
    u, _ = nucleo(x, w)
    return len(x), int(np.count_nonzero(u != u_modelo)), int(np.count_nonzero(u != rtl))


def cadena(bo, w, tx, grilla, bits, **kw):
    """La cadena de cuantizacion_zf.py con el núcleo de este script."""
    salida = {}

    def aplicar(x, c, Pmax, M):
        salida["u"], salida["sat"] = nucleo(x, c, **kw)
        return salida["u"]

    flotante = D.apply_memory_poly
    D.apply_memory_poly = aplicar
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            _, y, info = D.apply_dpd_and_pa(
                tx, w, pa_mode=PARAMS["pa_mode"], pa_nl=PARAMS["pa_nl"],
                use_memory=PARAMS["use_memory"], bo_db=bo, fs_in=PARAMS["fs"],
                fs_pa=PARAMS["fs_pa"], R0=PARAMS["R0"], Pmax=P, M=M,
                sat_amp=PARAMS["sat_amp"], p_soft=PARAMS["p_soft"],
                normalize_dpd_input=True, normalize_dpd_output=True,
                normalize_pa_input=False)
            r = eval_chain(y, grilla, bits, PARAMS, bo, info)
    finally:
        D.apply_memory_poly = flotante
    return r, salida["u"], salida["sat"]


def main():
    lineas = []

    def out(s=""):
        print(s, flush=True)
        lineas.append(s)

    n, d_modelo, d_rtl = comprobar()
    out(f"Comprobación sobre los {n} vectores funcionales: {d_modelo} muestras distintas del "
        f"modelo de punto fijo y {d_rtl} de la salida esperada del RTL.")

    bits_tr, bits = _bits()
    tx, grilla = generate_ofdm_signal(bits, PARAMS)
    x_rms = tx / np.sqrt(np.mean(np.abs(tx) ** 2))    # como normalize_dpd_input
    casos = [(e, 13) for e in (8, 4, 3, 2, 1, 0)] + [(8, f) for f in (10, 8, 6, 4, 2, 0)]

    for bo in (-9.0, 3.0):
        with contextlib.redirect_stdout(io.StringIO()):
            r_flot = simulate_with_dpd(bits_tr, bits, PARAMS, bo, method="gd",
                                       dpd_cfg=dict(DPD_CFG, M=M, Pmax=P))
        w = r_flot["train_out"]["dpd_coeffs"]
        out(f"\n=== Back-off {bo:+.0f} dB")
        out(f"Punto flotante: EVM {r_flot['evm_escalar_percent']:.3f} %, "
            f"ACPR {max(r_flot['acpr_l_db'], r_flot['acpr_u_db']):.2f} dB")

        u_flot = D.apply_memory_poly(x_rms, w, Pmax=P, M=M)
        u_16, _ = nucleo(x_rms, w, interno=False)
        picos = {}
        u_n, sat = nucleo(x_rms, w, picos=picos)
        out(f"Error contra punto flotante, relativo a la salida: registros de 16 bits "
            f"{db(u_16 - u_flot, u_flot):.1f} dB, formato interno Q8.28 {db(u_n - u_16, u_16):.1f} dB, "
            f"núcleo entero {db(u_n - u_flot, u_flot):.1f} dB. Saturaciones con Q8.28: {sat}.")
        out("\nPico de cada etapa antes de redondear, y bits enteros que necesita:")
        for etapa, v in picos.items():
            bits_ent = max(0, int(np.floor(np.log2(v))) + 1) if v > 0 else 0
            out(f"   {etapa:34s} {v:10.2e} {bits_ent:3d}")

        _, u_ref, _ = cadena(bo, w, tx, grilla, bits, interno=False)
        out(f"\n{'guarda ent.':>11} {'guarda frac.':>12} {'formato':>8} {'satur.':>7} "
            f"{'error interno [dB]':>19} {'EVM [%]':>8} {'ACPR [dB]':>9}")
        for ge, gf in casos:
            r, u, s = cadena(bo, w, tx, grilla, bits, guarda_ent=ge, guarda_frac=gf)
            err = db(u - u_ref, u_ref)
            out(f"{ge:11d} {gf:12d} {'Q%d.%d' % (ge, 15 + gf):>8} {s:7d} {err:19.1f} "
                f"{r['evm_escalar_percent']:8.3f} {max(r['acpr_l_db'], r['acpr_u_db']):9.2f}")

    os.makedirs(os.path.dirname(RESULTADO), exist_ok=True)
    with open(RESULTADO, "w") as f:
        f.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    main()
