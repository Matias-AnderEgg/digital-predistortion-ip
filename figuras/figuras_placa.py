"""Figuras de la validación en la placa, con el estilo de las de simulación.

  placa_psd.pdf     PSD a la salida del PA simulado con la señal LTE, a -9 y
                    +3 dB, sin DPD y con la salida real del IP
                    (sw/hw_test/lte/salida_bo<BO>.bin)
  placa_amam.pdf    AM/AM del PA solo, del DPD solo y de los dos en cascada,
                    a -9 dB con la misma corrida
  lazo_m9_nmse.pdf  NMSE por vuelta del lazo LTE con la placa, a -9 y +3 dB
                    (sw/lazo_simulado/resultados/lte_bo<BO>_n2000/vueltas.csv,
                    con el modelo identificado con 16000 muestras)
  lazo_m9_psd.pdf   PSD en la vuelta 0 y en la 14 del lazo, a -9 dB

RAIZ es la raíz del repositorio, de donde salen el modelo
de Python del IP, lte_placa.py y lazo_lte.py. Las PSD del lazo no se guardaron,
así que el lazo se repite en la Mac con lazo_lte.py y el modelo de Python del
IP, que en la placa dio código por código lo mismo, y se verifica que el NMSE
de cada vuelta sea el del CSV.

Uso: PYTHONDONTWRITEBYTECODE=1 python figuras/figuras_placa.py
"""
import csv
import os
import sys

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sw", "hw_test"))
sys.path.insert(0, os.path.join(RAIZ, "sw", "lazo_simulado"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import entrenamiento as E  # noqa: E402
import lte_placa as LP  # noqa: E402
import lazo_lte as LL  # noqa: E402
from fixed_point.acpr import compute_psd  # noqa: E402

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                            "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
                            "savefig.bbox": "tight"})
SALIDA = os.path.dirname(os.path.abspath(__file__))
GV = E.GV
NPERSEG = 256
VUELTAS, N_ENT, N_MOD = 15, 2000, 16000


def coma(ax, eje="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def bandas(ax, p):
    """Canal principal en celeste, adyacentes en rosa y alternos en gris, como
    en las de simulación."""
    ax.axvspan(-p["main_bw"] / 2e6, p["main_bw"] / 2e6, color="C0", alpha=0.12, lw=0, zorder=0)
    for k, c in [(1, "#f6dcdc"), (2, "0.96")]:
        a0 = (k * p["adj_offset"] - p["adj_bw"] / 2) / 1e6
        a1 = (k * p["adj_offset"] + p["adj_bw"] / 2) / 1e6
        for s in (-1, 1):
            ax.axvspan(s * a0, s * a1, color=c, lw=0, zorder=0)
    ax.set_xlim(-p["fs"] / 2e6, p["fs"] / 2e6)


def fig_psd(p, curvas, nombre, nperseg=NPERSEG, ylim=(-95, -55), ax=None):
    solo = ax is None
    if solo:
        fig, ax = plt.subplots(figsize=(6.0, 3.4))
    bandas(ax, p)
    for v, etiqueta, estilo, color, lw in curvas:
        f, pdb, _ = compute_psd(v, fs=p["fs"], nperseg=nperseg)
        ax.plot(f / 1e6, pdb, estilo, color=color, lw=lw, label=etiqueta)
    ax.set_ylim(*ylim)
    ax.set_xlabel("frecuencia [MHz]")
    coma(ax)
    coma(ax, "x")
    if not solo:
        return
    ax.set_ylabel("PSD [dB/Hz]")
    leg = ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower center",
                    bbox_to_anchor=(0.5, 1.0), handlelength=1.8, columnspacing=1.2)
    for ln in leg.get_lines():
        ln.set_linewidth(max(ln.get_linewidth(), 1.4))
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, nombre + ".pdf"))
    plt.close(fig)


def corrida_lte(bo):
    """Lo que dejó la corrida LTE en la placa a ese back-off: la subtrama, la
    salida real del IP y la del PA simulado sin DPD y con el IP."""
    d = np.load(os.path.join(LP.DIR, f"esperada_bo{bo:g}.npz"))
    hw = np.fromfile(os.path.join(LP.DIR, f"salida_bo{bo:g}.bin"), dtype="<u4")
    assert np.array_equal(hw, d["salida"]), f"la salida de la placa a {bo:g} dB no es la del modelo"
    re_, im_ = E.desempaquetar(hw)
    u_hw = (re_ + 1j * im_) / 32768.0 * GV.G_OUT
    x, grilla = d["x"], d["grilla"]
    y_sin, y_ip = LP.pa(x, bo, True), LP.pa(u_hw, bo, True)
    m_sin, m_ip = LP.metricas(x, grilla, y_sin), LP.metricas(x, grilla, y_ip)
    print(f"placa LTE {bo:+g} dB, {len(hw)} muestras iguales al modelo: NMSE {m_sin[0]:.2f} -> "
          f"{m_ip[0]:.2f} dB, EVM {m_sin[1]:.2f} -> {m_ip[1]:.2f} %, ACPR {m_sin[2]:.1f} -> {m_ip[2]:.1f} dB")
    return x, u_hw, y_sin, y_ip, m_sin, m_ip


def placa():
    p = LP.PARAMS
    r = {bo: corrida_lte(bo) for bo in (-9.0, 3.0)}

    fig, ejes = plt.subplots(1, 2, figsize=(6.2, 3.0), sharey=True)
    for ax, bo in zip(ejes, (-9.0, 3.0)):
        x, _, y_sin, y_ip, _, _ = r[bo]
        fig_psd(p, [(y_sin, "sin DPD", "-", "C3", 1.0),
                    (y_ip, "con el núcleo en la placa", "-", "C0", 1.2),
                    (x, "señal de entrada", ":", "k", 1.0)], None, nperseg=2048,
                ylim=(-130, -60), ax=ax)
        ax.set_title(f"{bo:+g} dB".replace("-", "−"), fontsize=9, pad=3)
    ejes[0].set_ylabel("PSD [dB/Hz]")
    leg = fig.legend(*ejes[0].get_legend_handles_labels(), fontsize=8, ncol=3, frameon=False,
                     loc="lower center", bbox_to_anchor=(0.5, 0.98), handlelength=1.8,
                     columnspacing=1.2)
    for ln in leg.get_lines():
        ln.set_linewidth(max(ln.get_linewidth(), 1.4))
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "placa_psd.pdf"))
    plt.close(fig)

    # El DPD solo es la salida del IP como la recibe el PA, con RMS 1. Cerca
    # de la saturación expande mucho, así que el eje se recorta y se dice
    # cuántas muestras quedan afuera.
    x, u_hw, y_sin, y_ip, _, _ = r[-9.0]
    u_pa = u_hw / np.sqrt(np.mean(np.abs(u_hw) ** 2))
    lim_x = 1.05 * np.abs(x).max()
    lim_y = 1.5 * lim_x
    afuera = int(np.sum(np.abs(u_pa) > lim_y))
    papr = lambda v: 10 * np.log10(np.max(np.abs(v) ** 2) / np.mean(np.abs(v) ** 2))  # noqa: E731
    print(f"placa: {afuera} de {len(u_pa)} muestras del DPD afuera, hasta {np.abs(u_pa).max():.2f}, "
          f"PAPR de la entrada {papr(x):.1f} dB y de la salida del DPD {papr(u_pa):.1f} dB")
    fig, ax = plt.subplots(figsize=(5.0, 3.9))
    idx = np.arange(0, len(x), 4)  # una de cada cuatro, para que el PDF no pese
    for v, etiqueta, color in [(y_sin, "PA solo", "C3"), (u_pa, "DPD solo", "C0"),
                               (y_ip, "DPD y PA", "C2")]:
        ax.plot(np.abs(x[idx]), np.abs(v[idx]), ".", ms=1.2, color=color, label=etiqueta,
                rasterized=True)
    ax.plot([0, lim_x], [0, lim_x], "--", color="0.5", lw=0.9, label="lineal")
    ax.set_xlim(0, lim_x)
    ax.set_ylim(0, lim_y)
    ax.annotate(f"{afuera} muestras del DPD\nllegan hasta {np.abs(u_pa).max():.1f}".replace(".", ","),
                xy=(0.97, 0.04), xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8.5, color="C0")
    ax.set_xlabel("amplitud de entrada")
    ax.set_ylabel("amplitud de salida")
    coma(ax)
    coma(ax, "x")
    ax.legend(fontsize=8.5, loc="upper left", markerscale=6)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "placa_amam.pdf"), dpi=300)
    plt.close(fig)


def lazo_en_mac(bo):
    """El lazo de lazo_lte.main con el motor de la Mac. Devuelve el NMSE de
    cada vuelta y la salida del PA en la vuelta 0 y en la última."""
    senal = LL.subtramas(VUELTAS)
    m = LL.EnMac([LL.codigos_x(x) for x, _ in senal])
    m.inicio()
    hist = []
    for k in range(VUELTAS):
        x, grilla = senal[k]
        u_cod = m.aplicar(k)
        y = LL.pa(LL.de_codigos(u_cod, GV.G_OUT), bo)
        hist.append(LP.metricas(x, grilla, y)[0])
        if k == 0:
            y_sin = y
            m.modelo(u_cod, LL.codigos_y(y), N_MOD)
        m.entrenar(k, N_ENT)
    return np.array(hist), y_sin, y, np.concatenate([x for x, _ in senal])


def lazo():
    curvas = {}
    for bo in (-9.0, 3.0):
        ruta = os.path.join(RAIZ, "sw", "lazo_simulado", "resultados", f"lte_bo{bo:g}_n{N_ENT}",
                            "vueltas.csv")
        with open(ruta) as fh:
            csv_nmse = np.array([float(r["nmse_db"]) for r in csv.DictReader(fh)])
        hist, y_sin, y_fin, ref = lazo_en_mac(bo)
        dif = np.max(np.abs(hist - csv_nmse))
        print(f"lazo {bo:+g} dB: NMSE vuelta 0 {hist[0]:.2f}, vuelta {VUELTAS - 1} {hist[-1]:.2f} dB, "
              f"promedio 10 a 14 {np.mean(csv_nmse[10:]):.2f} dB, diferencia maxima con el CSV "
              f"de la placa {dif:.4f} dB")
        assert dif < 1e-2, "el lazo en la Mac no reproduce el de la placa"
        curvas[bo] = csv_nmse
        if bo == -9.0:
            fig_psd(LP.PARAMS, [(y_sin, "sin DPD, vuelta 0", "-", "C3", 1.0),
                                (y_fin, f"con DPD, vuelta {VUELTAS - 1}", "-", "C0", 1.2),
                                (ref, "señal de entrada", ":", "k", 1.0)], "lazo_m9_psd",
                    nperseg=2048, ylim=(-130, -60))

    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for bo, c in [(-9.0, "C0"), (3.0, "C1")]:
        v = curvas[bo]
        etiqueta = f"{bo:+g} dB".replace("-", "−")
        ax.axhline(v[0], color=c, lw=0.9, ls="--")
        ax.plot(np.arange(len(v)), v, "o-", color=c, ms=4, lw=1.2, label=etiqueta)
    todos = np.concatenate(list(curvas.values()))
    ax.set_ylim(todos.min() - 1, todos.max() + 1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    ax.set_xticks(np.arange(0, VUELTAS, 2))
    ax.set_xlabel("vuelta")
    ax.set_ylabel("NMSE [dB]")
    coma(ax)
    ax.legend(fontsize=8.5, loc="upper right", title="back-off, a trazos sin DPD",
              title_fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(SALIDA, "lazo_m9_nmse.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    placa()
    lazo()
    print("listo")
