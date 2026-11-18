"""Robustez del ajuste del modelo del PA por gradiente (Robustez del entrenamiento).

Para varias realizaciones del registro de entrenamiento, 8000 muestras, M = 3 y P = 7,
mide:

    pico      amplitud máxima de x, con valor eficaz 1
    g_media   ganancia de lazo media del ajuste, mu_h * sum_i |phi_i|^2 / P_i
    g_pico    la misma ganancia en la muestra donde es máxima
    mse       MSE del modelo por época, 40 épocas, sin la guarda de
              divergencia, en cada back-off del barrido

con el esquema de DPD_CFG, mu_h = 1e-3 y sin NLMS por muestra. La ganancia de
lazo sólo depende de la entrada, así que se calcula una vez por realización.
La semilla 123 es la del registro del barrido.

Para las realizaciones que divergen, corre además la cadena completa a -9 dB
con la misma señal de evaluación que el barrido, sin guarda ni respaldo (el
entrenamiento de septiembre) y con el esquema actual, guarda y respaldo por
NLMS, y mide EVM y ACPR de los dos adyacentes.

Uso:
    python sys_with_acpr/robustez.py [--jobs 8] [--registros 20] > sys_with_acpr/resultados_robustez/robustez.log
Escribe sys_with_acpr/resultados_robustez/robustez.pkl, o robustez_<n>.pkl si
--registros no es 20.
"""
import argparse
import contextlib
import functools
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

from sys_with_acpr import dpd  # noqa: E402
from sys_with_acpr.main import generate_ofdm_signal, simulate_with_dpd  # noqa: E402
from sys_with_acpr.pa import apply_selected_pa  # noqa: E402
from sys_with_acpr.dpd import (_train_pa_model_gd, _basis_column_power,  # noqa: E402
                               _basis_from_history, _odd_orders, normalize_rms)
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, BO_LIST, _bits  # noqa: E402

DIR_SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_robustez")
SEMILLAS = [123] + list(range(1, 20))   # los 20 de la tesis; --registros cambia la cantidad
M, P = 3, 7
# Se estudia el entrenamiento con 8000 muestras y sin límite del paso. El de
# la tesis (límite 1 y 16000) está en robustez_arreglos.py, configuración
# lim1_n16000.
N = 8000
VIEJO = {"N_train_max": N, "h_ganancia_max": None}
MU_H = DPD_CFG["mu_h"]
EP_PA = DPD_CFG["ep_pa"]
DELTA = 1e-3                    # precond_delta de dpd.py


def registro(semilla):
    n_bits = PARAMS["n_ofdm_symbols"] * PARAMS["Nused"] * 4
    bits = np.random.default_rng(semilla).integers(0, 2, n_bits)
    x, _ = generate_ofdm_signal(bits, PARAMS)
    x, _ = normalize_rms(x[:N], target_rms=1.0)
    return x


def ganancia_lazo(x):
    orders = _odd_orders(P)
    inv_p = 1.0 / (_basis_column_power(x, orders, M) + DELTA)
    hist = np.zeros(M, dtype=np.complex128)
    g = np.empty(len(x))
    for n in range(len(x)):
        hist[1:] = hist[:-1]
        hist[0] = x[n]
        phi = _basis_from_history(hist, orders, M)
        g[n] = MU_H * float(np.sum(np.abs(phi) ** 2 * inv_p))
    return g


def correr(tarea):
    semilla, bo = tarea
    x = registro(semilla)
    y, _, _ = apply_selected_pa(x, pa_mode=PARAMS["pa_mode"], pa_nl=PARAMS["pa_nl"],
                                use_memory=PARAMS["use_memory"], bo_db=bo, R0=PARAMS["R0"],
                                fs_in=PARAMS["fs"], fs_pa=PARAMS["fs_pa"], sat_amp=PARAMS["sat_amp"],
                                p_soft=PARAMS["p_soft"], normalize_input=True, normalize_output=False)
    _, _, mse = _train_pa_model_gd(x, y, Pmax=P, M=M, mu=MU_H, n_epochs=EP_PA, N_train_max=N,
                                   use_nlms=False, divergence_factor=np.inf)
    return tarea, np.asarray(mse)


def cadena(tarea):
    """Cadena completa a -9 dB con el registro de la semilla, sin guarda o con
    el esquema actual."""
    semilla, guarda = tarea
    n_bits = PARAMS["n_ofdm_symbols"] * PARAMS["Nused"] * 4
    bits_train = np.random.default_rng(semilla).integers(0, 2, n_bits)
    _, bits_eval = _bits()
    cfg = dict(DPD_CFG, M=M, Pmax=P, **VIEJO)
    # El parche se deshace al terminar: el proceso del pool se reusa, y si
    # queda puesto la tarea siguiente "con guarda" corre sin guarda.
    try:
        if not guarda:
            dpd._train_pa_model_gd = functools.partial(_train_pa_model_gd, divergence_factor=np.inf)
            cfg["mu_h_respaldo"] = None
        with contextlib.redirect_stdout(io.StringIO()):
            r = simulate_with_dpd(bits_train, bits_eval, PARAMS, -9.0, method="gd", dpd_cfg=cfg)
    finally:
        dpd._train_pa_model_gd = _train_pa_model_gd
    t = r["train_out"]
    return tarea, {"evm": float(r["evm_percent"]),
                   "evm_escalar": float(r["evm_escalar_percent"]), "acpr_l": float(r["acpr_l_db"]),
                   "acpr_u": float(r["acpr_u_db"]), "mse_pa": np.asarray(t["mse_pa"]),
                   "pa_respaldo": bool(t.get("pa_respaldo", False))}


def cadena_todos(salida, jobs, t0):
    """Cadena completa a -9 dB, con guarda y respaldo, en los registros que no
    la tienen. Con guarda es lo que se usa en la tesis, así que da la EVM que
    se obtendría con cualquiera de los registros."""
    with open(salida, "rb") as f:
        d = pickle.load(f)
    faltan = [(s, True) for s in d["meta"]["SEMILLAS"] if (s, True) not in d["cadena"]]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        d["cadena"].update(dict(ex.map(cadena, faltan)))
    evm = np.array([d["cadena"][(s, True)]["evm"] for s in d["meta"]["SEMILLAS"]])
    resp = np.array([d["cadena"][(s, True)]["pa_respaldo"] for s in d["meta"]["SEMILLAS"]])
    print(f"Cadena con guarda a -9 dB en {len(faltan)} registros más, {len(evm)} en total")
    for nombre, v in [("sin respaldo", evm[~resp]), ("con respaldo", evm[resp])]:
        print(f"  {nombre}: {len(v)} registros, EVM mediana {np.median(v):.2f} %, "
              f"de {v.min():.2f} a {v.max():.2f} %")
    for s in d["meta"]["SEMILLAS"]:
        r = d["cadena"][(s, True)]
        print(f"  semilla {s:3d}: EVM {r['evm']:6.2f} %, ACPR {r['acpr_l']:6.2f} / {r['acpr_u']:6.2f} dB, "
              f"respaldo {r['pa_respaldo']}")
    with open(salida, "wb") as f:
        pickle.dump(d, f)
    print(f"\nGuardado en {salida} ({(time.time() - t0) / 60:.1f} min)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--registros", type=int, default=20,
                    help="cantidad de registros, el del barrido y las semillas 1 a n-1")
    ap.add_argument("--cadena-todos", action="store_true",
                    help="no vuelve a entrenar: lee robustez_<n>.pkl y agrega la cadena con "
                         "guarda en los registros que faltan")
    args = ap.parse_args()
    global SEMILLAS
    SEMILLAS = [123] + list(range(1, args.registros))
    nombre = "robustez.pkl" if args.registros == 20 else f"robustez_{args.registros}.pkl"
    SALIDA = os.path.join(DIR_SALIDA, nombre)
    if args.cadena_todos:
        cadena_todos(SALIDA, args.jobs, time.time())
        return
    t0 = time.time()

    regs = {}
    for s in SEMILLAS:
        x = registro(s)
        g = ganancia_lazo(x)
        regs[s] = {"pico": float(np.max(np.abs(x))), "g_media": float(np.mean(g)),
                   "g_pico": float(np.max(g)), "n_pico": int(np.argmax(g))}
    tareas = [(s, bo) for s in SEMILLAS for bo in BO_LIST]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        mse = dict(ex.map(correr, tareas))

    print(f"M = {M}, P = {P}, {N} muestras, mu_h = {MU_H:g}, {EP_PA} épocas, sin guarda")
    print("semilla   pico  g_media  g_pico   " + "  ".join(f"{bo:+6.1f}" for bo in BO_LIST))
    for s in sorted(SEMILLAS, key=lambda s: -regs[s]["g_pico"]):
        r = regs[s]
        cols = []
        for bo in BO_LIST:
            m = mse[(s, bo)]
            fin = m[np.isfinite(m)]
            diverge = len(fin) < len(m) or m[-1] > 1e3 * np.min(fin)
            cols.append("  DIV " if diverge else "  ok  ")
        print(f"{s:7d}  {r['pico']:5.2f}  {r['g_media']:7.4f}  {r['g_pico']:6.2f}  " + "  ".join(cols))
    print("\nMSE del modelo, primera y última época, en las que divergen:")
    for (s, bo), m in sorted(mse.items()):
        fin = m[np.isfinite(m)]
        if len(fin) < len(m) or m[-1] > 1e3 * np.min(fin):
            print(f"  semilla {s:3d}, BO {bo:+5.1f} dB: {m[0]:.3g} -> {m[-1]:.3g}")

    divergen = sorted({s for (s, bo), m in mse.items()
                       if not np.all(np.isfinite(m)) or m[-1] > 1e3 * np.min(m[np.isfinite(m)])})
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        cad = dict(ex.map(cadena, [(s, g) for s in divergen + [123] for g in (False, True)]))
    print("\nCadena completa a -9 dB:")
    for (s, g), r in sorted(cad.items()):
        nombre = "con guarda y respaldo" if g else "sin guarda          "
        m = r["mse_pa"]
        print(f"  semilla {s:3d}, {nombre}: EVM {r['evm']:5.2f} %, ACPR {r['acpr_l']:6.2f} / "
              f"{r['acpr_u']:6.2f} dB, MSE del modelo {m[0]:.3g} -> {m[-1]:.3g} en {len(m)} épocas, "
              f"respaldo {r['pa_respaldo']}")

    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as f:
        pickle.dump({"meta": {"SEMILLAS": SEMILLAS, "BO_LIST": BO_LIST, "M": M, "P": P, "N": N,
                              "MU_H": MU_H, "EP_PA": EP_PA},
                     "registros": regs, "mse": mse, "cadena": cad}, f)
    print(f"\nGuardado en {SALIDA} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
