"""Reidentificar el modelo del PA en el lazo LTE.

Compara "sin reidentificar" contra "reidentificando desde cero, GD", con
el lazo de lazo_lte.py. Todo en la Mac (EnMac de lazo_lte.py), PA fijo, a BO +3, 0 y -9 dB:

  - 26 vueltas, una subtrama nueva por vuelta (las 26 que tiene la señal de
    evaluación de la tesis), pasadas del DPD de 2000 muestras;
  - en la vuelta 0, sin DPD, el modelo del PA se identifica con las primeras
    16000 muestras de la subtrama, como en lazo_lte.py;
  - reID: en las vueltas 9, 14, 19 y 24, antes de entrenar, el modelo se
    identifica desde cero con las dos últimas vueltas, la salida del IP u y
    la del PA y: las primeras 8000 muestras de cada una, 16000 en total,
    como la primera identificacion. Configuración por defecto del C (40
    epocas, paso limitado a 1).

Con limitador: la deseada del DPD no es h0 x sino h0 x recortada en amplitud al techo, con la fase de x. El
techo es el pico de la salida del PA en la identificacion de la vuelta 0
(DPD apagado) y queda fijo aunque se reidentifique. El C no tiene esa opción,
así que esos casos entrenan con dla_py de otro_pa.py, que sin techo da lo
mismo que el C del lazo (w a 1e-16).

Imprime la EVM promedio de las vueltas 10 a 25 y deja el resumen en
experimentos/resultados/reid_lte.txt, y cada vuelta en
experimentos/resultados/lte_bo<BO>_<caso>.csv (no se versionan).

Uso: python experimentos/reid_lte.py
"""
import ctypes
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
R = os.path.abspath(os.path.join(AQUI, ".."))
LAZO = os.path.join(R, "sw", "lazo_simulado")
sys.path.insert(0, LAZO)
sys.path.insert(0, os.path.join(R, "sw", "otro_pa"))
SALIDA = os.path.join(AQUI, "resultados")

V, N_ENT, N_MOD = 26, 2000, 16000
REID = (9, 14, 19, 24)
DESDE = 10
BOS = (3.0, 0.0, -9.0)
CASOS = ("sin reID", "reID", "lim sin reID", "lim reID")


def correr(arg):
    bo, caso = arg
    import lazo_lte as L
    senal = L.subtramas(V)
    m = L.EnMac([L.codigos_x(x) for x, _ in senal])
    techo = None
    m.inicio()
    us, ys, filas = [], [], []
    for k in range(V):
        x, grilla = senal[k]
        u_cod = m.aplicar(k)
        y = L.pa(L.de_codigos(u_cod, L.GV.G_OUT), bo)
        y_cod = L.codigos_y(y)
        nm, evm, acpr, alt, _, _ = L.LP.metricas(x, grilla, y)
        filas.append((k, nm, evm, acpr, alt))
        us.append(u_cod)
        ys.append(y_cod)
        if k == 0:
            m.modelo(u_cod, y_cod, N_MOD)
            if caso.startswith("lim"):
                techo = np.max(np.abs(L.de_codigos(y_cod[:N_MOD], L.G_Y)))
        if k in REID and caso in ("reID", "lim reID"):
            mitad = N_MOD // 2
            m.modelo(np.concatenate([us[-2][:mitad], us[-1][:mitad]]),
                     np.concatenate([ys[-2][:mitad], ys[-1][:mitad]]), N_MOD)
        if techo is None:
            m.entrenar(k, N_ENT)
        else:
            entrenar_con_techo(m, k, techo, L)
    return bo, caso, filas


def entrenar_con_techo(m, k, techo, L):
    """Lo mismo que EnMac.entrenar (una época, warm start), con dla_py y la
    deseada recortada al techo."""
    import otro_pa as O
    c = m.cfg
    x = np.ascontiguousarray(m.x[k][:N_ENT])
    w = O.dla_py(x, m.h, mu=c.mu_w, ep=1, lam=c.w_reg_lambda, G=c.G, delta=c.precond_delta,
                 clip=c.grad_clip, techo=techo, w_ini=m.w.copy(),
                 delta_col=c.precond_delta / c.G ** 2)
    m.w[:] = w
    c.warm_start = 1
    c.ep_dpd = 1
    m.c = m.lib.dpd_rms_salida(np.ascontiguousarray(m.x[k]), L.N_BLOQUE, m.w, ctypes.byref(c))


def main():
    os.makedirs(SALIDA, exist_ok=True)
    trabajos = [(bo, caso) for bo in BOS for caso in CASOS]
    with ProcessPoolExecutor(len(trabajos)) as ex:
        res = list(ex.map(correr, trabajos))
    lineas = [f"Lazo LTE en la Mac, PA fijo, {V} vueltas, pasadas de {N_ENT}, modelo con {N_MOD}",
              f"reID en las vueltas {', '.join(map(str, REID))}; promedio de las vueltas {DESDE} a {V - 1}",
              "", f"{'BO':>5s}  {'caso':12s} {'EVM':>7s} {'NMSE':>9s} {'ACPR':>8s}"]
    for bo, caso, filas in res:
        f = np.array(filas)
        nombre = f"lte_bo{bo:g}_{caso.replace(' ', '_')}.csv"
        np.savetxt(os.path.join(SALIDA, nombre), f, delimiter=",", fmt="%.3f",
                   header="vuelta,nmse_db,evm_pct,acpr_db,alterno_db", comments="")
        ult = f[DESDE:]
        lineas.append(f"{bo:+5.1f}  {caso:12s} {ult[:, 2].mean():5.2f} % {ult[:, 1].mean():6.2f} dB "
                      f"{ult[:, 3].mean():6.1f} dB")
    texto = "\n".join(lineas)
    print(texto)
    open(os.path.join(SALIDA, "reid_lte.txt"), "w").write(texto + "\n")


if __name__ == "__main__":
    main()
