"""El modelo del PA en Q0.15 contra flotante, con la señal LTE de la tesis.

A BO -9 dB, PA con memoria, fixed_bo, tres casos:

  sin DPD
  h flotante    el C por defecto, como lte_placa.py (16000 muestras)
  h Q0.15       lo mismo con quantize_h = 1

El DPD se aplica en punto fijo, como el IP, y se mide sobre la señal de
evaluación entera (364 símbolos, semilla 456) con las métricas de
lte_placa.py: NMSE, EVM, peor ACPR adyacente y alterno. Además busca los
tramos fuera del canal donde la PSD con DPD queda por encima de la de sin
DPD, con la PSD de Welch de 8192 muestras suavizada en 100 kHz.

Uso: python experimentos/h_q15_lte.py
(deja experimentos/resultados/h_q15_lte.txt y salida/h_q15_lte.png)
"""
import ctypes
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, ".."))
sys.path.insert(0, os.path.join(RAIZ, "sw", "hw_test"))
import lte_placa as L  # noqa: E402

T, GV, dpd = L.T, L.GV, L.dpd
BO = -9.0
SUAVE = 100e3      # ancho del promedio móvil de la PSD
UMBRAL = 0.5       # dB por encima de sin DPD para contar un tramo


def entrenar(**cambios):
    """Como L.entrenar_c, con cambios en la configuracion."""
    x_ent = L.senal(123)[0][:L.N_ENT]
    x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
    y_ent = np.ascontiguousarray(L.pa(x_ent, BO, False))
    lib = T.cargar_lib()
    cfg = T.Cfg()
    lib.dpd_train_cfg_default(ctypes.byref(cfg))
    for campo, valor in cambios.items():
        setattr(cfg, campo, valor)
    h = np.zeros(12, np.complex128)
    w = np.zeros(12, np.complex128)
    lib.dpd_train_pa_model(x_ent, y_ent, L.N_ENT, ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
    lib.dpd_train_dla(x_ent, L.N_ENT, h, ctypes.byref(cfg), w, np.zeros(cfg.ep_dpd))
    re_ = np.zeros(12, np.int16)
    im_ = np.zeros(12, np.int16)
    lib.dpd_coef_to_q15(w, re_, im_)
    return (re_ + 1j * im_) / 32768.0


def suavizar(f, psd_db):
    k = max(1, int(round(SUAVE / (f[1] - f[0]))))
    p = np.convolve(10 ** (psd_db / 10), np.ones(k) / k, mode="same")
    return 10 * np.log10(p)


def tramos(f, exceso, fuera):
    """Tramos contiguos fuera del canal con exceso > UMBRAL: (desde, hasta, max)."""
    malo = (exceso > UMBRAL) & fuera
    res, i = [], 0
    while i < len(f):
        if malo[i]:
            j = i
            while j + 1 < len(f) and malo[j + 1]:
                j += 1
            res.append((f[i], f[j], exceso[i:j + 1].max()))
            i = j + 1
        else:
            i += 1
    return res


def main():
    x, grilla = L.senal(456)
    casos = [("sin DPD", x)]
    for nombre, cambios in (("h flotante", {}), ("h Q0.15", {"quantize_h": 1})):
        w = entrenar(**cambios)
        casos.append((nombre, dpd.apply_memory_poly_fxp(x, w, Pmax=GV.PMAX, M=GV.M_TAPS, cfg=GV.CFG)))
    res = {}
    for nombre, u in casos:
        nm, ev, ac, al, f, psd = L.metricas(x, grilla, L.pa(u, BO, True))
        res[nombre] = (nm, ev, ac, al, suavizar(f, psd))

    canal = L.PARAMS["main_bw"] / 2
    fuera = np.abs(f) > canal
    lineas = [f"BO {BO:g} dB, PA con memoria, fixed_bo, senal LTE, {grilla.shape[0]} simbolos de evaluacion",
              f"entrenamiento: C por defecto, {L.N_ENT} muestras; DPD aplicado en punto fijo", "",
              f"{'':12s} {'NMSE':>9s} {'EVM':>8s} {'ACPR':>8s} {'alterno':>8s}"]
    for nombre, (nm, ev, ac, al, _) in res.items():
        lineas.append(f"{nombre:12s} {nm:6.2f} dB {ev:6.2f} % {ac:5.1f} dB {al:5.1f} dB")
    lineas += ["", f"Tramos fuera del canal (|f| > {canal / 1e6:.1f} MHz) con la PSD mas de "
               f"{UMBRAL} dB por encima de sin DPD (suavizada en {SUAVE / 1e3:.0f} kHz):"]
    ref = res["sin DPD"][4]
    for nombre in ("h flotante", "h Q0.15"):
        tr = tramos(f, res[nombre][4] - ref, fuera)
        ex = np.where(fuera, res[nombre][4] - ref, -np.inf)
        i = int(np.argmax(ex))
        lineas.append(f"  {nombre}: {'ninguno' if not tr else ''} (lo mas cerca: {ex[i]:+.1f} dB "
                      f"en {f[i] / 1e6:+.2f} MHz)")
        for a, b, m in tr:
            lineas.append(f"    {a / 1e6:+7.2f} a {b / 1e6:+7.2f} MHz, hasta {m:.1f} dB")
    texto = "\n".join(lineas)
    print(texto)
    os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)
    open(os.path.join(AQUI, "resultados", "h_q15_lte.txt"), "w").write(texto + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(8, 4.5))
    colores = {"sin DPD": "tab:red", "h flotante": "tab:green", "h Q0.15": "tab:blue"}
    for nombre, r in res.items():
        a.plot(f / 1e6, r[4], color=colores[nombre], lw=1, label=nombre)
    a.set_xlim(-30.72, 30.72)
    a.set_xlabel("frecuencia [MHz]")
    a.set_ylabel("PSD [dB/Hz]")
    a.set_title(f"BO {BO:g} dB, LTE: modelo del PA en flotante y en Q0.15")
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.join(AQUI, "salida"), exist_ok=True)
    fig.savefig(os.path.join(AQUI, "salida", "h_q15_lte.png"), dpi=200)


if __name__ == "__main__":
    main()
