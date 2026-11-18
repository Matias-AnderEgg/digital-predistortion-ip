"""Optimo del costo del DLA con el modelo del PA saturado, PA de la tesis.

Pregunta: si el DPD se entrena a través de un modelo del PA que satura en vez
de extrapolar, el optimo de ese costo (least_squares, solo diagnostico) se
acerca al DPD ajustado contra el PA real? Si no se acerca, no vale la pena
probar el GD con el modelo saturado.

Modelos (h identificado con el C, sin saturación):
  sal      yhat = mp(u, h) recortado en amplitud a A = max|y_ent|, fase de yhat
  sal+1dB  igual con A un dB arriba
  ent      u recortado a U = max|x_ent| (fase de u) antes de pasar por el modelo
Objetivos:
  h0x      d = h0 x, el del DLA
  lim      d = h0 x recortado en amplitud a A, fase de x (el limitador)
  lib      ganancia libre: g yhat contra x, con g el que mejor ajusta
Modelo pol: el polinomio sin tocar. norm: u escalado a RMS 1 antes del modelo.
Misma estructura que sys_with_acpr/piso_grilla.py (M=3, P=7, w[0] = 1). Arranca del mejor de
w_C y paso directo según su propio costo. Importa otro_pa sin modificarlo.
"""
import ctypes
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import otro_pa as O  # noqa: E402

L, T, GV, dpd = O.L, O.T, O.GV, O.dpd
BOS = [-9, -7.5, -6, -4.5, -3, -1.5, 0, 1.5, 3, 9]
if len(sys.argv) > 1:
    BOS = [float(b) for b in sys.argv[1:]]
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados", "meseta_optimo.txt")

x_ent = L.senal(123)[0][:L.N_ENT]
x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
x, grilla = L.senal(456)
mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
completo = lambda r: np.concatenate(([1.0 + 0j], r[:11] + 1j * r[11:]))  # noqa: E731
reales = lambda w: np.concatenate((w[1:].real, w[1:].imag))  # noqa: E731


def recorte(v, a):
    m = np.abs(v)
    return np.where(m > a, a * v / np.maximum(m, 1e-300), v)


lineas = [f"{'BO':>5s}  {'DPD':14s} {'EVM':>7s} {'ACPR':>7s}  {'pico u':>6s}  {'fuera':>6s}  {'rms u':>5s}"]
print(lineas[0], flush=True)
t0 = time.time()
for bo in BOS:
    y_ent = np.ascontiguousarray(O.pa(x_ent, "tesis", bo, False))
    lib = T.cargar_lib()
    cfg = T.Cfg()
    lib.dpd_train_cfg_default(ctypes.byref(cfg))
    h = np.zeros(12, np.complex128)
    w_c = np.zeros(12, np.complex128)
    lib.dpd_train_pa_model(x_ent, y_ent, len(x_ent), ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
    lib.dpd_train_dla(x_ent, len(x_ent), h, ctypes.byref(cfg), w_c, np.zeros(cfg.ep_dpd))
    A, U = np.max(np.abs(y_ent)), np.max(np.abs(x_ent))
    d_h0 = h[0] * x_ent
    d_lim = recorte(d_h0, A)

    modelos = {
        "pol": lambda u: mp(u, h),
        "sal": lambda u: recorte(mp(u, h), A),
        "sal+1dB": lambda u: recorte(mp(u, h), A * 10 ** (1 / 20)),
        "ent": lambda u: mp(recorte(u, U), h),
        # Como el PA de la simulación (fixed_bo): u escalado a RMS 1 antes del PA.
        "norm": lambda u: mp(u / np.sqrt(np.mean(np.abs(u) ** 2)), h),
    }
    sols = {"C": w_c}
    for nm, f in modelos.items():
        for nd, d in (("h0x", d_h0), ("lim", d_lim), ("lib", None)):
            def res(r, f=f, d=d):
                yh = f(mp(x_ent, completo(r)))
                # lib: ganancia libre, como el ajuste contra el PA real de sys_with_acpr/piso_grilla.py
                e = yh - d if d is not None else np.vdot(yh, x_ent) / np.vdot(yh, yh) * yh - x_ent
                return np.concatenate((e.real, e.imag))
            ini = min((w_c, completo(np.zeros(22))), key=lambda w: np.sum(res(reales(w)) ** 2))
            sols[f"{nm} {nd}"] = completo(least_squares(res, reales(ini), x_scale="jac").x)

    for etiqueta, w in sols.items():
        u = mp(x, w)
        _, evm, acpr = L.metricas(x, grilla, O.pa(u, "tesis", bo, True))[:3]
        rx = np.sqrt(np.mean(np.abs(x) ** 2))
        pico = np.max(np.abs(u)) / rx
        rms = np.sqrt(np.mean(np.abs(u) ** 2)) / rx
        # Fuera del rango identificado, como en limitador(): el PA lleva u a
        # RMS 1, igual que la entrada de identificacion.
        a = np.abs(u) / np.sqrt(np.mean(np.abs(u) ** 2))
        fuera = 100 * np.mean(a > U)
        lineas.append(f"{bo:+5.1f}  {etiqueta:14s} {evm:6.2f}% {acpr:7.1f}  {pico:6.2f}  {fuera:5.2f}%  {rms:5.3f}")
        print(lineas[-1], flush=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

with open(SALIDA, "w") as fh:
    fh.write("\n".join(lineas) + "\n")
print("guardado en", SALIDA)
