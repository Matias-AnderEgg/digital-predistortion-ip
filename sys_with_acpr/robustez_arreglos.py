"""Dos arreglos posibles para la dependencia con el registro de entrenamiento
(Robustez del entrenamiento).

Con 200 registros (robustez.py --registros 200), el ajuste del modelo del PA
diverge en 34, y entre los que convergen la EVM a -9 dB va de 3,0 a 14,7 %,
con mediana 5,5 %. Las dos cosas se ordenan con el pico de la señal de
entrenamiento. La lectura es que en la muestra del pico el paso del LMS sale
muy grande y desplaza el modelo. Se prueba:

    limite   el paso efectivo de cada muestra no pasa de g_max (el
             h_ganancia_max del C), con 8000 muestras
    muestras 16000 y 32000 muestras, sin límite

Cada configuración corre la cadena completa (guarda y respaldo, como en la
tesis) con los 200 registros de robustez.py, a -9 dB, y el límite también a
+9 dB, para ver si cuesta algo donde el PA es casi lineal. La línea de base,
8000 muestras sin límite, está en robustez_200.pkl.

Uso:
    python sys_with_acpr/robustez_arreglos.py [--jobs 10] [--solo lim1,n16000]
Escribe resultados_robustez/arreglos_<config>.pkl, uno por configuración.
"""
import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr.main import simulate_with_dpd  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, _bits  # noqa: E402

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_robustez")
SEMILLAS = [123] + list(range(1, 200))
M, P = 3, 7

# nombre: (g_max, N, back-off[, cambios a DPD_CFG])
CONFIGS = {
    "lim1": (1.0, 8000, (-9.0, 9.0)),
    "lim2": (2.0, 8000, (-9.0, 9.0)),
    "n16000": (None, 16000, (-9.0,)),
    "n32000": (None, 32000, (-9.0,)),
    "base9": (None, 8000, (9.0,)),      # la base a +9 dB, para comparar el límite
    "lim1_n16000": (1.0, 16000, (-9.0, 9.0)),   # las dos juntas
    # El NLMS del respaldo como esquema principal, para ver si en general es
    # peor que el LMS precondicionado o sólo con la semilla 123.
    "nlms": (None, 8000, (-9.0, 9.0), {"mu_h": 0.05, "use_nlms_h": True}),
}


def cadena(tarea):
    semilla, g_max, n, bo, extra = tarea
    n_bits = PARAMS["n_ofdm_symbols"] * PARAMS["Nused"] * 4
    bits_train = np.random.default_rng(semilla).integers(0, 2, n_bits)
    _, bits_eval = _bits()
    # g_max y N explícitos: DPD_CFG ya trae el límite y 16000 muestras, y la
    # base de esta comparación es el entrenamiento sin límite y con 8000.
    cfg = dict(DPD_CFG, M=M, Pmax=P, N_train_max=n, h_ganancia_max=g_max, **extra)
    with contextlib.redirect_stdout(io.StringIO()):
        r = simulate_with_dpd(bits_train, bits_eval, PARAMS, bo, method="gd", dpd_cfg=cfg)
    t = r["train_out"]
    return (semilla, bo), {"evm": float(r["evm_percent"]),
                           "evm_escalar": float(r["evm_escalar_percent"]), "acpr_l": float(r["acpr_l_db"]),
                           "acpr_u": float(r["acpr_u_db"]), "mse_pa": np.asarray(t["mse_pa"]),
                           "pa_respaldo": bool(t.get("pa_respaldo", False))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--solo", default=",".join(CONFIGS))
    args = ap.parse_args()
    for nombre in args.solo.split(","):
        g_max, n, bos, *resto = CONFIGS[nombre]
        extra = resto[0] if resto else {}
        t0 = time.time()
        tareas = [(s, g_max, n, bo, extra) for bo in bos for s in SEMILLAS]
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            res = dict(ex.map(cadena, tareas))
        print(f"{nombre}: g_max {g_max}, {n} muestras, {extra or ''}, {len(SEMILLAS)} registros")
        for bo in bos:
            evm = np.array([res[(s, bo)]["evm"] for s in SEMILLAS])
            resp = np.array([res[(s, bo)]["pa_respaldo"] for s in SEMILLAS])
            print(f"  {bo:+.0f} dB: EVM mediana {np.median(evm):.2f} %, de {evm.min():.2f} a "
                  f"{evm.max():.2f} %, percentiles 5 y 95 {np.percentile(evm, 5):.2f} y "
                  f"{np.percentile(evm, 95):.2f} %, respaldo en {resp.sum()}, "
                  f"semilla 123 {res[(123, bo)]['evm']:.2f} %")
            esc = np.array([res[(s, bo)]["evm_escalar"] for s in SEMILLAS])
            print(f"        escalar: mediana {np.median(esc):.2f} %, de {esc.min():.2f} a "
                  f"{esc.max():.2f} %, percentiles 5 y 95 {np.percentile(esc, 5):.2f} y "
                  f"{np.percentile(esc, 95):.2f} %, semilla 123 {res[(123, bo)]['evm_escalar']:.2f} %")
        with open(os.path.join(DIR, f"arreglos_{nombre}.pkl"), "wb") as f:
            pickle.dump({"meta": {"SEMILLAS": SEMILLAS, "g_max": g_max, "N": n, "BO": bos, "extra": extra,
                                  "M": M, "P": P}, "res": res}, f)
        print(f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    main()
