"""Por qué el aprendizaje indirecto falla a back-off bajo y anda a back-off alto.

El post-inverso del ILA
se ajusta sobre la salida del PA, que a back-off bajo está comprimida, y
después se aplica a la entrada, cuyos picos van más allá del rango de amplitud
en el que se ajustó. Ahí el polinomio extrapola.

Para cada back-off del barrido y M = 3, P = 3, 5, 7, 9, 11, entrena el ILA
como lo hace la cadena (train_dpd_ls) y mide:
    nmse      error del post-inverso sobre el entrenamiento [dB]
    ymax      amplitud máxima de la salida del PA (rango de ajuste)
    xmax      amplitud máxima de la entrada (rango de uso)
    fuera     porcentaje de muestras de la entrada con |x| > ymax
    papr_u    pico sobre valor eficaz de la salida del predistorsionador

Uso:
    python sys_with_acpr/rango_ila.py
Escribe sys_with_acpr/resultados_identificacion/rango_ila.pkl.
"""
import contextlib
import io
import os
import pickle
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr import dpd, barrido_mp as b  # noqa: E402
from sys_with_acpr.main import generate_ofdm_signal  # noqa: E402
from sys_with_acpr.pa import apply_selected_pa  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_identificacion", "rango_ila.pkl")
BOS = [-9, -6, -3, 0, 3, 6, 9]
M = 3
PS = [3, 5, 7, 9, 11]


def medir(tx, p, bo, P):
    pa_kw = dict(pa_mode=p["pa_mode"], pa_nl=p["pa_nl"], use_memory=p["use_memory"], bo_db=bo, R0=p["R0"],
                 fs_in=p["fs"], fs_pa=p["fs_pa"], sat_amp=p["sat_amp"], p_soft=p["p_soft"])
    with contextlib.redirect_stdout(io.StringIO()):
        out = dpd.train_dpd_ls(tx, Pmax=P, M=M, normalize_gain=True, **pa_kw)
        x, _ = dpd.normalize_rms(tx, target_rms=1.0)
        y, _, _ = apply_selected_pa(x, normalize_input=True, normalize_output=True, **pa_kw)
    c = out["dpd_coeffs"]
    Phi, v, _ = dpd.build_memory_poly_regressor(y, P, M)
    # La ganancia lineal se normaliza a 1 después del ajuste, así que el error
    # se mide con la mejor ganancia escalar entre el post-inverso y la entrada.
    xh = Phi @ c
    g = np.vdot(xh, x[v]) / np.vdot(xh, xh)
    nmse = 10 * np.log10(np.sum(abs(x[v] - g * xh) ** 2) / np.sum(abs(x[v]) ** 2))
    u = dpd.apply_memory_poly(x, c, Pmax=P, M=M)
    ymax = float(np.max(abs(y)))
    return {"nmse": float(nmse), "ymax": ymax, "xmax": float(np.max(abs(x))),
            "fuera": float(np.mean(abs(x) > ymax) * 100),
            "papr_u": float(np.max(abs(u)) / np.sqrt(np.mean(abs(u) ** 2)))}


def main():
    p = dict(b.PARAMS)
    bits_train, _ = b._bits()
    tx, _ = generate_ofdm_signal(bits_train, p)
    # Las mismas muestras que el barrido, para que los números coincidan con
    # los suyos.
    tx = tx[:b.DPD_CFG["N_train_max"]]
    res = {(bo, P): medir(tx, p, bo, P) for bo in BOS for P in PS}
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as fh:
        pickle.dump(res, fh)

    print(f"{'':22s}" + "".join(f"{bo:+8d}" for bo in BOS))
    print(f"{'max|x|':22s}" + "".join(f"{res[(bo, 7)]['xmax']:8.2f}" for bo in BOS))
    print(f"{'max|y|':22s}" + "".join(f"{res[(bo, 7)]['ymax']:8.2f}" for bo in BOS))
    print(f"{'fuera de rango [%]':22s}" + "".join(f"{res[(bo, 7)]['fuera']:8.2f}" for bo in BOS))
    for P in PS:
        print(f"{'NMSE post-inv P=' + str(P):22s}" + "".join(f"{res[(bo, P)]['nmse']:8.1f}" for bo in BOS))
    for P in PS:
        print(f"{'pico/rms u P=' + str(P):22s}" + "".join(f"{res[(bo, P)]['papr_u']:8.1f}" for bo in BOS))
    print(f"guardado en {SALIDA}")


if __name__ == "__main__":
    main()
