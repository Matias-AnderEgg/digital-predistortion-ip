"""El error del entrenamiento del predistorsionador con un objetivo que satura.

En el aprendizaje directo, el error del predistorsionador compara la salida
del modelo del amplificador con la señal deseada d(n) = G x(n), y el modelo
es un polinomio que no satura. En compresión ese error explota en los picos
aunque el predistorsionador mejore la señal, y por eso la Sec. 6.2.1 sigue la
convergencia con el amplificador real. Este script prueba si tener en cuenta
la saturación lo arregla.

Entrena como el barrido (sys_with_acpr/barrido_mp.py, M = 3, P = 7) pero con
40 épocas del predistorsionador, guarda los coeficientes de cada época (como
sys_with_acpr/epocas_dpd.py, sin tocar dpd.py) y mide por época, sobre el
registro de entrenamiento:

  modelo, d          el error del entrenamiento, contra d(n) = G x(n)
  modelo, d sat.     contra el limitador ideal de la Ec. 5.1, que recorta
                     |G x(n)| en la salida máxima del amplificador, A_sat
  modelo sat., d sat. además la salida del modelo recortada en A_sat
  PA real            el NMSE con el amplificador real y la ganancia alineada,
                     con la entrada del amplificador a la misma potencia
                     media, como la Fig. de convergencia

Los tres primeros, en dB relativos a la potencia de d. A_sat es la amplitud
que da el amplificador con una entrada constante muy grande.

Uso: PYTHONDONTWRITEBYTECODE=1 python experimentos/convergencia_saturacion.py
(un minuto, los cuatro back-off en paralelo; deja
experimentos/resultados/convergencia_saturacion.txt)
"""
import contextlib
import inspect
import io
import os
import sys
from multiprocessing import Pool

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, RAIZ)
os.environ.setdefault("MPLBACKEND", "Agg")

from sys_with_acpr import dpd, barrido_mp as b  # noqa: E402
from sys_with_acpr.main import generate_ofdm_signal  # noqa: E402

RESULTADO = os.path.join(AQUI, "resultados", "convergencia_saturacion.txt")
BOS = [-9.0, -3.0, 3.0, 9.0]
EPOCAS = 40
MOSTRAR = (0, 1, 2, 3, 5, 10, 20, 40)
M, P = 3, 7
PARAMS = b.PARAMS

# Coeficientes al final de cada época: se recompila el entrenamiento desde su
# fuente con una línea que los guarda.
W_EPOCAS = []
_fuente = inspect.getsource(dpd._train_dpd_gd_direct_dla)
_ancla = "        mse_hist.append(mse)\n"
assert _fuente.count(_ancla) == 1
dpd.W_EPOCAS = W_EPOCAS
exec(compile(_fuente.replace(_ancla, _ancla + "        W_EPOCAS.append(w.copy())\n"),
             dpd.__file__, "exec"), dpd.__dict__)


def amplificador(u, bo, normalizar_entrada):
    y, _, _ = dpd.apply_selected_pa(
        u, pa_mode=PARAMS["pa_mode"], pa_nl=PARAMS["pa_nl"], use_memory=PARAMS["use_memory"],
        bo_db=bo, R0=PARAMS["R0"], fs_in=PARAMS["fs"], fs_pa=PARAMS["fs_pa"],
        sat_amp=PARAMS["sat_amp"], p_soft=PARAMS["p_soft"],
        normalize_input=normalizar_entrada, normalize_output=False)
    return y


def recortar(v, a):
    return v * np.minimum(1.0, a / np.maximum(np.abs(v), 1e-300))


def db(e, ref):
    return 10 * np.log10(np.mean(np.abs(e) ** 2) / np.mean(np.abs(ref) ** 2))


def correr(bo):
    W_EPOCAS.clear()
    bits_tr, _ = b._bits()
    cfg = dict(b.DPD_CFG, M=M, Pmax=P)
    tx = generate_ofdm_signal(bits_tr, PARAMS)[0][:int(cfg["N_train_max"])]
    with contextlib.redirect_stdout(io.StringIO()):
        r = dpd.train_dpd_gd(
            tx, pa_mode=PARAMS["pa_mode"], pa_nl=PARAMS["pa_nl"], use_memory=PARAMS["use_memory"],
            bo_db=bo, fs_in=PARAMS["fs"], fs_pa=PARAMS["fs_pa"], R0=PARAMS["R0"], Pmax=P, M=M,
            sat_amp=PARAMS["sat_amp"], p_soft=PARAMS["p_soft"], normalize_gain=False,
            mu_h=cfg["mu_h"], mu_w=cfg["mu_w"], use_nlms_h=cfg.get("use_nlms_h", False),
            mu_h_respaldo=cfg.get("mu_h_respaldo", 0.05), h_ganancia_max=cfg.get("h_ganancia_max"),
            ep_pa=cfg["ep_pa"], ep_dpd=EPOCAS, N_train_max=cfg.get("N_train_max"),
            w_amp_clip=cfg.get("w_amp_clip"), w_reg_lambda=cfg.get("w_reg_lambda", 0.0),
            w_reg_pmin=cfg.get("w_reg_pmin", 1))
    x, _ = dpd.normalize_rms(tx, target_rms=1.0)
    h, G = r["h_model"], r["target_gain"]
    a_sat = float(np.median(np.abs(amplificador(1e3 * np.ones(4096, complex), bo, False)[1000:3000])))
    d = G * x
    d_sat = recortar(d, a_sat)
    identidad = np.zeros(M * (P + 1) // 2, complex)
    identidad[0] = 1.0
    filas = []
    for k, w in enumerate([identidad] + list(W_EPOCAS)):
        u = dpd.apply_memory_poly(x, w, Pmax=P, M=M)
        y_mod = dpd.apply_memory_poly(u, h, Pmax=P, M=M)
        y_pa = amplificador(u, bo, True)
        g = np.vdot(y_pa, x) / np.vdot(y_pa, y_pa)
        filas.append((k, db(d - y_mod, d), db(d_sat - y_mod, d),
                      db(d_sat - recortar(y_mod, a_sat), d), db(x - g * y_pa, x)))
    return bo, a_sat, float(np.mean(np.abs(d) > a_sat)), filas


def main():
    with Pool(len(BOS)) as pool:
        res = pool.map(correr, BOS)
    lineas = []
    for bo, a_sat, frac, filas in res:
        lineas.append(f"=== Back-off {bo:+.0f} dB: A_sat = {a_sat:.3f}, "
                      f"{100 * frac:.2f} % de las muestras con |G x| > A_sat")
        lineas.append(f"{'época':>5} {'modelo, d':>10} {'modelo, d sat.':>15} "
                      f"{'modelo sat., d sat.':>20} {'PA real':>8}")
        for k, e1, e2, e3, e4 in filas:
            if k in MOSTRAR:
                lineas.append(f"{k:5d} {e1:10.2f} {e2:15.2f} {e3:20.2f} {e4:8.2f}")
        lineas.append("")
    texto = "\n".join(lineas)
    print(texto)
    os.makedirs(os.path.dirname(RESULTADO), exist_ok=True)
    with open(RESULTADO, "w") as f:
        f.write("Época 0: sin predistorsión. Errores en dB.\n\n" + texto)


if __name__ == "__main__":
    main()
