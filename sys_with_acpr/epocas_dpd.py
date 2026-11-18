"""¿Sirven las épocas del DLA? El DPD de cada época evaluado con el PA real.

El error de cada época se mide a través del modelo identificado del PA, un
polinomio que no satura, así que no sirve para elegir época. Acá se guarda w
al final de cada época y cada uno se evalúa con la misma cadena que
el barrido (simulate_with_dpd, PA real, mismos bits), para M = 1 y 3, P = 7,
en cinco back-off. Por este experimento el DLA devuelve el w de la última
época. Este script fuerza 40 épocas (EP_DPD) para ver
la curva completa, y la columna "elegida" es la que devuelve el entrenamiento.

Para no tocar dpd.py, _train_dpd_gd_direct_dla se recompila desde su fuente
con una línea que guarda w por época, y main.train_dpd_gd se reemplaza al
evaluar por uno que devuelve el entrenamiento ya hecho con otro w.

Uso:
    python sys_with_acpr/epocas_dpd.py
Escribe sys_with_acpr/resultados_identificacion/epocas_dpd.pkl.
"""
import contextlib
import inspect
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

from sys_with_acpr import dpd, main, barrido_mp as b  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_identificacion", "epocas_dpd.pkl")
BOS = [-9, -3, 0, 3, 9]
MS = [1, 3]
P = 7
EP_DPD = 40

# w al final de cada época, en el orden en que se entrenan.
W_EPOCAS = []
_fuente = inspect.getsource(dpd._train_dpd_gd_direct_dla)
_ancla = "        mse_hist.append(mse)\n"
assert _fuente.count(_ancla) == 1
dpd.W_EPOCAS = W_EPOCAS
exec(compile(_fuente.replace(_ancla, _ancla + "        W_EPOCAS.append(w.copy())\n"), dpd.__file__, "exec"),
     dpd.__dict__)


def _medir(r):
    # NMSE de la Ec. (eq:nmse) entre la salida del PA real y la señal original
    # de evaluación, con la ganancia alineada (como direct_nmse_gain_aligned de
    # dpd.py y el lazo de la placa).
    x, y = np.asarray(r["tx_signal"]), np.asarray(r["y_pa"])
    a = np.vdot(y, x) / np.vdot(y, y)
    nmse = 10 * np.log10(np.sum(np.abs(x - a * y) ** 2) / np.sum(np.abs(x) ** 2))
    return {"evm": float(r["evm_percent"]), "evm_escalar": float(r["evm_escalar_percent"]),
            "acpr": float(max(r["acpr_l_db"], r["acpr_u_db"])), "nmse": float(nmse)}


def correr(t):
    M, bo = t
    bits_train, bits_eval = b._bits()
    p = dict(b.PARAMS)
    cfg = dict(b.DPD_CFG, M=M, Pmax=P, ep_dpd=EP_DPD)
    W_EPOCAS.clear()
    entrenar = main.train_dpd_gd
    with contextlib.redirect_stdout(io.StringIO()):
        r = main.simulate_with_dpd(bits_train, bits_eval, p, bo, method="gd", dpd_cfg=cfg)
        tr = r["train_out"]
        ws = [w.copy() for w in W_EPOCAS]
        identidad = np.zeros_like(ws[0])
        identidad[0] = 1.0
        por_epoca = []
        for w in [identidad] + ws:
            main.train_dpd_gd = lambda *a, _w=w, **k: dict(tr, dpd_coeffs=_w)
            por_epoca.append(_medir(main.simulate_with_dpd(bits_train, bits_eval, p, bo, method="gd", dpd_cfg=cfg)))
        main.train_dpd_gd = entrenar
    return t, {"elegido": _medir(r), "mse_dpd": np.asarray(tr["mse_dpd"]), "por_epoca": por_epoca,
               "respaldo": bool(tr.get("pa_respaldo", False))}


def main_():
    t0 = time.time()
    with Pool(len(MS) * len(BOS)) as pool:
        res = dict(pool.map(correr, [(M, bo) for M in MS for bo in BOS]))
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as fh:
        pickle.dump(res, fh)
    print(f"{time.time() - t0:.0f} s, guardado en {SALIDA}")
    ep = [0, 1, 2, 3, 5, 10, 20, 40]
    for met, fmt in [("evm", "{:7.2f}"), ("evm_escalar", "{:7.2f}"), ("acpr", "{:7.1f}"), ("nmse", "{:7.1f}")]:
        print(f"\n{met} por época (0 = identidad)  elegida" + "".join(f"{e:7d}" for e in ep) + "  mejor(época)")
        for (M, bo), d in res.items():
            v = [x[met] for x in d["por_epoca"]]
            k = int(np.argmin(v))
            print(f"  M={M} BO {bo:+3d}  {fmt.format(d['elegido'][met])}        "
                  + "".join(fmt.format(v[e]) for e in ep) + f"  {fmt.format(v[k])} ({k})")
    print("\népoca de menor error del DPD, el criterio anterior (1 = primera):",
          {k: int(np.argmin(d["mse_dpd"])) + 1 for k, d in res.items()})
    print("respaldo:", [k for k, d in res.items() if d["respaldo"]])


if __name__ == "__main__":
    main_()
