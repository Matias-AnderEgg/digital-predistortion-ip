"""Cómo identificar el modelo del PA en el DLA: GD, NLMS o LS.

Dos partes:

1. Memoria del PA vista por un memory polynomial: NMSE de un modelo del PA
   ajustado por LS, P = 7, M = 1, 3, 5, 7, 9, sobre muestras distintas de las
   de ajuste, sin memoria y con PA_MEMORIA_K = 1, 2, 4.
2. DLA por GD con el modelo del PA identificado de tres maneras, M = 1, 3, 5,
   en los siete back-off del barrido:
       gd    el esquema de DPD_CFG (mu_h = 1e-3, respaldo NLMS)
       nlms  NLMS con mu_h = 0,05 (la configuración del respaldo)
       ls    mínimos cuadrados (reemplaza _train_pa_model_gd en el proceso)
   Mide el NMSE del modelo del PA sobre el entrenamiento, ACPR, EVM con el
   receptor por subportadora y con el escalar, y el NMSE de la salida del PA
   contra la entrada con una ganancia compleja (sin receptor).

Uso:
    python sys_with_acpr/identificacion_pa.py
Escribe sys_with_acpr/resultados_identificacion/identificacion_pa.pkl.
"""
import contextlib
import io
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr import pa, dpd, barrido_mp as b  # noqa: E402
from sys_with_acpr.main import simulate_no_dpd, simulate_with_dpd  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_identificacion",
                      "identificacion_pa.pkl")
BOS = [-9, -6, -3, 0, 3, 6, 9]
_orig = dpd._train_pa_model_gd
ESTADO = {}


def _nmse_h(x, y, h, Pmax, M):
    Phi, v, _ = dpd.build_memory_poly_regressor(x, Pmax, M)
    e = y[v] - Phi @ h
    return 10 * np.log10(np.sum(abs(e) ** 2) / np.sum(abs(y[v]) ** 2))


def _pa_model(x, y, *, Pmax, M, mu, n_epochs=5, N_train_max=None, **kw):
    N = min(len(x), N_train_max or len(x))
    x, y = x[:N], y[:N]
    if ESTADO["modo"] == "ls":
        Phi, v, orders = dpd.build_memory_poly_regressor(x, Pmax, M)
        h = np.linalg.lstsq(Phi, y[v], rcond=None)[0]
        mse = np.full(n_epochs, np.mean(abs(y[v] - Phi @ h) ** 2))
    else:
        h, orders, mse = _orig(x, y, Pmax=Pmax, M=M, mu=mu, n_epochs=n_epochs, N_train_max=N_train_max, **kw)
    ESTADO["nmse_h"] = _nmse_h(x, y, h, Pmax, M)
    return h, orders, mse


def memoria_vs_M(bo):
    """Parte 1: NMSE del modelo LS del PA en función de M."""
    bits, _ = b._bits()
    N0, NT = 20000, 120000
    out = {}
    for nombre, k, mem in (("sin memoria", 1, False), ("k=1", 1, True), ("k=2", 2, True), ("k=4", 4, True)):
        pa.PA_MEMORIA_K = k
        p = dict(b.PARAMS, use_memory=mem, pa_mode="arctan_mem" if mem else "arctan_nomem")
        with contextlib.redirect_stdout(io.StringIO()):
            r = simulate_no_dpd(bits, p, bo)
        x, y = r["tx_signal"], r["y_pa"]
        for M in (1, 3, 5, 7, 9):
            Phi, v, _ = dpd.build_memory_poly_regressor(x, 7, M)
            yv = y[v]
            tr, te = slice(N0, N0 + NT), slice(N0 + NT, N0 + 2 * NT)
            c = np.linalg.lstsq(Phi[tr], yv[tr], rcond=None)[0]
            e = yv[te] - Phi[te] @ c
            out[(nombre, M)] = 10 * np.log10(np.sum(abs(e) ** 2) / np.sum(abs(yv[te]) ** 2))
    pa.PA_MEMORIA_K = 1
    return bo, out


def dla(t):
    """Parte 2: ('no_dpd', bo) o (modo, M, bo)."""
    dpd._train_pa_model_gd = _pa_model
    bits_train, bits_eval = b._bits()
    p = dict(b.PARAMS)
    with contextlib.redirect_stdout(io.StringIO()):
        if t[0] == "no_dpd":
            r = simulate_no_dpd(bits_eval, p, t[1])
        else:
            modo, M, bo = t
            ESTADO["modo"] = modo
            cfg = dict(b.DPD_CFG, M=M, Pmax=7)
            if modo == "nlms":
                cfg.update(use_nlms_h=True, mu_h=0.05)
            r = simulate_with_dpd(bits_train, bits_eval, p, bo, method="gd", dpd_cfg=cfg)
        x, y = r["tx_signal"], r["y_pa"]
        g = np.vdot(x, y) / np.vdot(x, x)
        nmse = 10 * np.log10(np.sum(abs(y - g * x) ** 2) / np.sum(abs(y) ** 2))
    out = {"evm": r["evm_percent"], "evm_esc": r["evm_escalar_percent"], "nmse": nmse,
           "acpr": max(r["acpr_l_db"], r["acpr_u_db"]), "Pxx_dB": r["Pxx_dB"], "f": r["f"]}
    if "train_out" in r:
        out["respaldo"] = bool(r["train_out"].get("pa_respaldo", False))
        out["nmse_h"] = ESTADO["nmse_h"]
        out["w"] = r["train_out"]["dpd_coeffs"]
    return t, out


def main():
    t0 = time.time()
    with Pool(8) as pool:
        mem = dict(pool.map(memoria_vs_M, (-9, 0, 3)))
        tareas = [("no_dpd", bo) for bo in BOS] + [(m, M, bo) for m in ("gd", "nlms", "ls")
                                                    for M in (1, 3, 5) for bo in BOS]
        res = dict(pool.map(dla, tareas))
    with open(SALIDA, "wb") as fh:
        pickle.dump({"memoria_vs_M": mem, "dla": res}, fh)
    print(f"{time.time() - t0:.0f} s, guardado en {SALIDA}")

    print("\nNMSE del modelo LS del PA [dB], M = 1, 3, 5, 7, 9")
    for bo, d in mem.items():
        for nombre in ("sin memoria", "k=1", "k=2", "k=4"):
            print(f"  BO {bo:+d}  {nombre:12s}" + "".join(f"{d[(nombre, M)]:7.1f}" for M in (1, 3, 5, 7, 9)))
    for met, tit, fmt in (("nmse_h", "NMSE modelo PA [dB]", "{:7.1f}"), ("acpr", "ACPR [dB]", "{:7.1f}"),
                          ("evm", "EVM subportadora [%]", "{:7.2f}"), ("evm_esc", "EVM escalar [%]", "{:7.2f}"),
                          ("nmse", "NMSE sin receptor [dB]", "{:7.1f}")):
        print(f"\n{tit:24s}" + "".join(f"{bo:+7d}" for bo in BOS))
        if met != "nmse_h":
            print(f"{'sin DPD':24s}" + "".join(fmt.format(res[('no_dpd', bo)][met]) for bo in BOS))
        for modo in ("gd", "nlms", "ls"):
            for M in (1, 3, 5):
                print(f"{'PA ' + modo + ' M=' + str(M):24s}" + "".join(fmt.format(res[(modo, M, bo)][met]) for bo in BOS))
    print("\nrespaldo:", sorted({k[:2] for k, v in res.items() if v.get("respaldo")}))


if __name__ == "__main__":
    main()
