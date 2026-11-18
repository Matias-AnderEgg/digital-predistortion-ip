"""¿Mejora el DPD con más épocas del modelo del PA?

M = 3, P = 7, GD, misma cadena y bits que el barrido (barrido_mp), con
ep_pa = 10, 20, 30, 40 (el de la tesis), 60, 80 y 160, a -9, -3, +3 y +9 dB.
Con más épocas el DPD mejora a back-off alto
(+9 dB: 0,54 -> 0,36 %), y en compresión 40 es el mejor punto: con menos el
modelo no alcanza, y con más no mejora (el error del modelo deja de bajar en la
época 79 a -9 dB).

Uso:
    python sys_with_acpr/epocas_pa.py
Escribe sys_with_acpr/resultados_identificacion/epocas_pa.log y .pkl, este
con la historia del MSE del modelo por época (unos 30 s).
"""
import contextlib, io, os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_identificacion", "epocas_pa.log")
sys.path.insert(0, ROOT)
from sys_with_acpr import main, barrido_mp as b  # noqa: E402

BOS = [-9, -3, 3, 9]
EPS = [10, 20, 30, 40, 60, 80, 160]

def correr(t):
    ep, bo = t
    bits_train, bits_eval = b._bits()
    cfg = dict(b.DPD_CFG, M=3, Pmax=7, ep_pa=ep)
    with contextlib.redirect_stdout(io.StringIO()):
        r = main.simulate_with_dpd(bits_train, bits_eval, dict(b.PARAMS), bo, method="gd", dpd_cfg=cfg)
    tr = r["train_out"]
    m = np.asarray(tr["mse_pa"])
    return t, dict(mse_pa=m, evm=float(r["evm_percent"]), esc=float(r["evm_escalar_percent"]),
                   acpr=float(max(r["acpr_l_db"], r["acpr_u_db"])), mse_fin=float(m[-1]), mse_min=float(m.min()),
                   ep_min=int(np.argmin(m)) + 1, n=len(m), resp=bool(tr.get("pa_respaldo", False)))

if __name__ == "__main__":
    t0 = time.time()
    with Pool(len(BOS) * len(EPS)) as p:
        res = dict(p.map(correr, [(e, bo) for e in EPS for bo in BOS]))
    lineas = [f"{time.time()-t0:.0f} s",
              "ep_pa  BO     EVM   EVM esc   ACPR    MSE modelo (min, época)  respaldo"]
    for e in EPS:
        for bo in BOS:
            r = res[(e, bo)]
            lineas.append(f"{e:5d} {bo:+3d}  {r['evm']:6.2f}  {r['esc']:6.2f}  {r['acpr']:6.1f}   "
                          f"{r['mse_min']:.2e} ({r['ep_min']}/{r['n']})  {r['resp']}")
    print("\n".join(lineas))
    with open(SALIDA, "w") as fh:
        fh.write("\n".join(lineas) + "\n")
    # Historia completa del MSE del modelo por época, para la figura.
    with open(SALIDA.replace(".log", ".pkl"), "wb") as fh:
        pickle.dump({"BOS": BOS, "EPS": EPS, "res": res}, fh)
