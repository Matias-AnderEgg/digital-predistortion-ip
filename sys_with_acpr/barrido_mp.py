"""Barrido de M, P, método y back-off para la elección de la configuración.

Regenera el barrido del 2do semestre (informe "Barrido Final MyP") con el
código actual, y guarda todo lo que necesitan las figuras de una sola corrida.

Grilla, la misma del informe:
    corte de memoria  M = 1, 3, 5, 7 con P = 7
    corte de orden    P = 3, 5, 7, 9, 11 con M = 3
    back-off          -9 a +9 dB cada 3 dB
    método            LS (aprendizaje indirecto) y GD (aprendizaje directo)
más la referencia sin DPD en cada back-off y la señal ideal.

Señal, PA y configuración de entrenamiento son las de main.py (las copia
PARAMS y DPD_CFG, mantenerlas iguales). Cada punto se entrena con las mismas
semillas que main.py, 123 para entrenar y 456 para evaluar.

El receptor ecualiza por subportadora (rx_eq), y la EVM con el receptor
escalar se guarda también.

Por punto se guarda: EVM (con el receptor por subportadora y con el escalar), BER, ACPR_L/U del adyacente y del alterno, coeficientes del DPD (y de h en GD),
operaciones, curvas de MSE por época, si entró el respaldo NLMS, la PSD a la
salida del PA y una muestra de la constelación alineada, con los dos receptores.

Uso:
    python sys_with_acpr/barrido_mp.py            # corre todo, en paralelo
    python sys_with_acpr/barrido_mp.py --jobs 4
Escribe sys_with_acpr/resultados_barrido/barrido_mp.pkl e imprime las tablas
de EVM con el mismo formato que el informe.
"""
import argparse
import contextlib
import io
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr.main import simulate_ideal, simulate_no_dpd, simulate_with_dpd  # noqa: E402
from sys_with_acpr import lte  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_barrido", "barrido_mp.pkl")

BO_LIST = [-9, -6, -3, 0, 3, 6, 9]
M_CORTE = [1, 3, 5, 7]           # con P = 7
P_CORTE = [3, 5, 7, 9, 11]       # con M = 3
METODOS = ["ls", "gd"]
N_CONST = 3000                   # símbolos de constelación que se guardan

# Copia de main.py. Bajada LTE de 10 MHz (lte.py).
_NFFT, _NUSED, _OS, _NSYM, _DF = lte.NFFT, lte.NUSED, 4, 364, lte.DF
_FS = _NFFT * _DF * _OS
PARAMS = {
    "senal": "lte", "Nfft": _NFFT, "Nused": _NUSED, "os_factor": _OS, "n_ofdm_symbols": _NSYM,
    "fs": _FS, "fs_pa": _FS, "main_bw": _NUSED * _DF, "adj_offset": 10e6,
    "adj_bw": _NUSED * _DF, "R0": 50.0, "pa_mode": "arctan_mem", "pa_nl": "arctan",
    "use_memory": True, "sat_amp": 1.0, "p_soft": 3.0, "plot_time_debug": False,
    "rx_eq": "subportadora",
}
DPD_CFG = {
    "rcond": 1e-4, "eval_mode": "fixed_bo",
    "mu_h": 0.001, "use_nlms_h": False, "mu_h_respaldo": 0.05, "mu_w": 0.5,
    # Paso efectivo del modelo del PA limitado a 1 y 16000 muestras.
    "h_ganancia_max": 1.0,
    "ep_pa": 40, "ep_dpd": 5, "N_train_max": 16000,
    "w_reg_lambda": 1e-4, "w_reg_pmin": 3,
}


def _bits():
    n_bits = _NSYM * _NUSED * 4
    bits_train = np.random.default_rng(123).integers(0, 2, n_bits)
    bits_eval = np.random.default_rng(456).integers(0, 2, n_bits)
    return bits_train, bits_eval


def _resumen(r):
    """Lo que se guarda de un resultado de main.py, sin las señales enteras."""
    out = {
        "evm": float(r["evm_percent"]), "evm_escalar": float(r["evm_escalar_percent"]),
        "ber": float(r["ber"]),
        "acpr_l": float(r["acpr_l_db"]), "acpr_u": float(r["acpr_u_db"]),
        "acpr_alt_l": float(r["acpr_alt_l_db"]), "acpr_alt_u": float(r["acpr_alt_u_db"]),
        "Pxx_dB": np.asarray(r["Pxx_dB"], dtype=np.float32),
        "const": np.asarray(r["qam_rx_aligned"][:N_CONST], dtype=np.complex64),
        # Con el receptor escalar.
        "const_escalar": np.asarray(r["qam_rx_escalar"][:N_CONST], dtype=np.complex64),
    }
    t = r.get("train_out")
    if t is not None:
        out.update({
            "w": np.asarray(t["dpd_coeffs"]), "h": t.get("h_model"),
            "ops": float(t.get("ops_total", np.nan)),
            "mse_pa": np.asarray(t.get("mse_pa", [])), "mse_dpd": np.asarray(t.get("mse_dpd", [])),
            "pa_respaldo": bool(t.get("pa_respaldo", False)),
            "nmse_directo": float(t.get("direct_nmse_gain_aligned", np.nan)),
        })
    return out


def correr_punto(tarea):
    """Una tarea: ('ideal',), ('no_dpd', bo) o (metodo, M, P, bo)."""
    bits_train, bits_eval = _bits()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        if tarea[0] == "ideal":
            r = simulate_ideal(bits_eval, PARAMS)
        elif tarea[0] == "no_dpd":
            r = simulate_no_dpd(bits_eval, PARAMS, tarea[1])
        else:
            metodo, M, P, bo = tarea
            cfg = dict(DPD_CFG, M=M, Pmax=P)
            r = simulate_with_dpd(bits_train, bits_eval, PARAMS, bo, method=metodo, dpd_cfg=cfg)
    res = _resumen(r)
    res["seg"] = time.time() - t0
    if tarea[0] == "ideal":
        res["f"] = np.asarray(r["f"])
    return tarea, res


def tareas():
    configs = sorted({(M, 7) for M in M_CORTE} | {(3, P) for P in P_CORTE})
    ts = [("ideal",)] + [("no_dpd", bo) for bo in BO_LIST]
    # GD primero, que es lo que más tarda
    for metodo in ["gd", "ls"]:
        ts += [(metodo, M, P, bo) for (M, P) in configs for bo in BO_LIST]
    return ts


def tabla(res, metodo, corte):
    if corte == "M":
        cols, clave, tit = M_CORTE, (lambda c: (metodo, c, 7)), f"EVM [%] {metodo.upper()}, P = 7"
    else:
        cols, clave, tit = P_CORTE, (lambda c: (metodo, 3, c)), f"EVM [%] {metodo.upper()}, M = 3"
    print(f"\n{tit}")
    print("  BO  " + "".join(f"{corte}={c:<6}" for c in cols) + "  sin DPD")
    for bo in BO_LIST:
        fila = "".join(f"{res[clave(c) + (bo,)]['evm']:<8.2f}" for c in cols)
        print(f"{bo:+4d}  {fila}  {res[('no_dpd', bo)]['evm']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()

    ts = tareas()
    print(f"{len(ts)} puntos, {args.jobs} procesos", flush=True)
    res, t0 = {}, time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(correr_punto, t) for t in ts]
        for i, fu in enumerate(as_completed(futs), 1):
            tarea, r = fu.result()
            res[tarea] = r
            extra = " respaldo NLMS" if r.get("pa_respaldo") else ""
            print(f"[{i}/{len(ts)}] {tarea} EVM={r['evm']:.2f} % ({r['seg']:.0f} s){extra}", flush=True)

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    meta = {"BO_LIST": BO_LIST, "M_CORTE": M_CORTE, "P_CORTE": P_CORTE, "PARAMS": PARAMS,
            "DPD_CFG": DPD_CFG, "N_CONST": N_CONST, "fecha": time.strftime("%Y-%m-%d %H:%M"),
            "minutos": (time.time() - t0) / 60}
    with open(SALIDA, "wb") as f:
        pickle.dump({"meta": meta, "res": res}, f)
    print(f"\nGuardado en {SALIDA} ({meta['minutos']:.1f} min)")

    for metodo in METODOS:
        tabla(res, metodo, "M")
    for metodo in METODOS:
        tabla(res, metodo, "P")


if __name__ == "__main__":
    main()
