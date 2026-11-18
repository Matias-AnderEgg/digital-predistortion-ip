"""Modo piso de sw/otro_pa/otro_pa.py en la grilla de -9 a +3 dB, PA de la tesis.

Pregunta: la meseta de la EVM con DPD entre -9 y -3 dB, viene del modelo del
PA (el DPD se entrena a través de un polinomio que no satura) o del PA real
(saturación)? Mismos tres DPD que piso(), con la misma estructura (M=3, P=7,
w[0] = 1), mas uno:
  C           el del C (modelo por GD, DPD por GD a través del modelo)
  optimo      minimiza el costo del DLA a través del modelo (least_squares)
  PA real     minimiza el NMSE a través del PA verdadero, ganancia libre
  PA real h0  igual, pero contra la ganancia del modelo h0 x, el objetivo del DLA
Los dos de PA real arrancan del mejor de C y optimo según su propio costo.
Lo cita la tesis en Convergencia del entrenamiento (fila "PA real").

Importa sw/otro_pa/otro_pa.py sin modificarlo, que a su vez usa sw/hw_test/
y el C compilado de sw/dpd_train/ (make -C sw/dpd_train).

Uso:
    python sys_with_acpr/piso_grilla.py
Escribe sys_with_acpr/resultados_identificacion/piso_grilla.txt (unos 70 s).
"""
import ctypes
import os
import sys
import time

import numpy as np
from scipy.optimize import least_squares

OTRO_PA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sw", "otro_pa")
sys.path.insert(0, OTRO_PA_DIR)
import otro_pa as O  # noqa: E402

L, T, GV, dpd = O.L, O.T, O.GV, O.dpd
BOS = [-9, -7.5, -6, -4.5, -3, -1.5, 0, 1.5, 3]
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_identificacion", "piso_grilla.txt")

x_ent = L.senal(123)[0][:L.N_ENT]
x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
x, grilla = L.senal(456)
mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
completo = lambda r: np.concatenate(([1.0 + 0j], r[:11] + 1j * r[11:]))  # noqa: E731
reales = lambda w: np.concatenate((w[1:].real, w[1:].imag))  # noqa: E731

lineas = [f"{'BO':>5s}  {'DPD':10s} {'EVM':>7s} {'ACPR':>7s}  {'pico u':>6s}"]
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

    def r_dla(r):
        e = mp(mp(x_ent, completo(r)), h) - h[0] * x_ent
        return np.concatenate((e.real, e.imag))

    def r_pa(r):
        y = O.pa(mp(x_ent, completo(r)), "tesis", bo, False)
        e = np.vdot(y, x_ent) / np.vdot(y, y) * y - x_ent
        return np.concatenate((e.real, e.imag))

    def r_pa_h0(r):
        e = O.pa(mp(x_ent, completo(r)), "tesis", bo, False) - h[0] * x_ent
        return np.concatenate((e.real, e.imag))

    w_opt = completo(least_squares(r_dla, reales(w_c), x_scale="jac").x)
    sol = {}
    for nombre, f in (("PA real", r_pa), ("PA real h0", r_pa_h0)):
        ini = min((w_c, w_opt), key=lambda w: np.sum(f(reales(w)) ** 2))
        sol[nombre] = completo(least_squares(f, reales(ini), x_scale="jac").x)
    # pico u: pico de la entrada al PA con DPD, en veces el RMS de x.
    for etiqueta, w in (("C", w_c), ("optimo", w_opt), ("PA real", sol["PA real"]),
                        ("PA real h0", sol["PA real h0"])):
        u = mp(x, w)
        nm, evm, acpr = L.metricas(x, grilla, O.pa(u, "tesis", bo, True))[:3]
        pico = np.max(np.abs(u)) / np.sqrt(np.mean(np.abs(x) ** 2))
        lineas.append(f"{bo:+5.1f}  {etiqueta:10s} {evm:6.2f}% {acpr:7.1f}  {pico:6.2f}")
        print(lineas[-1], flush=True)
    print(f"  ({time.time() - t0:.0f} s)", flush=True)

with open(SALIDA, "w") as fh:
    fh.write("\n".join(lineas) + "\n")
print("guardado en", SALIDA)
