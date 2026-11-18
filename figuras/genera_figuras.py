#!/usr/bin/env python3
"""Genera las figuras de la tesis hechas con Matplotlib (PDF vectorial) en esta carpeta.

Uso:
    python figuras/genera_figuras.py                       # todas
    python figuras/genera_figuras.py --only conversion,desempeno_bo

La lista de figuras es el diccionario FIGS, al final. Las de los capítulos
teóricos se calculan en el momento con sys_with_acpr. Las de resultados leen
lo que dejan los scripts de sys_with_acpr que se corren aparte (barrido_mp.py,
efecto_pa.py, robustez.py, validacion_canal.py y los demás). Las de la placa y la de
cuantizacion las hacen figuras_placa.py y cuantizacion_zf.py.

El script es defensivo: si una figura no se puede generar, avisa y sigue con
las demás. Requiere numpy, scipy y matplotlib.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

matplotlib.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 10,
    "axes.grid": True,
    "grid.linestyle": "--",
    "grid.alpha": 0.4,
    "savefig.bbox": "tight",
})

# Configuración


def _lte(n_sym, seed, filtro=True):
    """Señal LTE de 10 MHz de las simulaciones (sys_with_acpr/lte.py), con el
    filtro de transmisión, y su grilla. Devuelve también los PARAMS."""
    from sys_with_acpr.main import generate_ofdm_signal
    from sys_with_acpr.barrido_mp import PARAMS
    p = dict(PARAMS, n_ofdm_symbols=n_sym)
    bits = np.random.default_rng(seed).integers(0, 2, n_sym * p["Nused"] * 4)
    x, grid = generate_ofdm_signal(bits, p)
    return x, grid, p


def _bandas(ax, p, alterno=True, color="0.92"):
    """Sombrea los canales adyacentes (y los alternos, mas claros) donde se
    integra el ACPR, y pone el eje de frecuencia en toda la banda simulada."""
    for k, c in [(1, color)] + ([(2, "0.96")] if alterno else []):
        a0 = (k * p["adj_offset"] - p["adj_bw"] / 2) / 1e6
        a1 = (k * p["adj_offset"] + p["adj_bw"] / 2) / 1e6
        for s in (-1, 1):
            ax.axvspan(s * a0, s * a1, color=c, lw=0, zorder=0)
    ax.set_xlim(-p["fs"] / 2e6, p["fs"] / 2e6)


# Paleta categórica para las curvas de una familia (M, P o back-off). Viridis
# y plasma daban tonos vecinos parecidos (M = 3 y M = 5 casi del mismo verde).
# Esta es la de Okabe e Ito, distinguible
# también con daltonismo, sin el amarillo, que no se ve sobre blanco, y
# ordenada para que dos valores consecutivos tengan colores bien distintos.
PALETA = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#000000"]


def _paleta(n: int) -> list:
    if n > len(PALETA):
        raise ValueError(f"la paleta tiene {len(PALETA)} colores y se pidieron {n}")
    return PALETA[:n]


def _save(fig, name: str) -> None:
    out = HERE / f"{name}.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"  ok  -> {out.relative_to(REPO)}")


# Curvas de conversión AM-AM y AM-PM

def fig_conversion() -> None:
    """Las dos curvas de conversion del modelo de PA, en paneles contiguos.

    Comparten el eje de abscisas para que el lector pueda leer las dos
    distorsiones al mismo nivel de entrada. El rango se recorta al tramo útil,
    desde bastante antes de la compresión hasta entrada la saturación, porque
    mas allá de eso las dos curvas son planas y no aportan.
    """
    from sys_with_acpr.pa import compute_pa_power_curve, _pa_static_nonlinearity

    c = compute_pa_power_curve(pa_mode="arctan_mem", pa_nl="arctan",
                               use_memory=False)
    pin, pout, pideal = c["pin_dbm"], c["pout_dbm"], c["pout_ideal_dbm"]
    p1 = c.get("p1db")
    x0, x1 = -32.0, 26.0

    amp = np.logspace(x0 / 20.0, x1 / 20.0, 400)
    y = _pa_static_nonlinearity(amp.astype(complex), pa_nl="arctan", sat_amp=1.0)
    fase = np.degrees(np.angle(y))
    fase -= fase[0]
    pin_f = 20 * np.log10(amp)

    # Se incluye a ancho completo del texto, unas 6 pulgadas, así que el
    # lienzo se hace de ese tamanio para que la tipografia no se reduzca.
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.1, 2.7), sharex=True)

    m = (pin >= x0) & (pin <= x1)

    # Regiones de operacion sobre la curva real. El limite entre lineal y
    # compresión es el punto de 1 dB. El de compresión a saturación se toma
    # donde la pendiente cae por debajo de 0.2 dB/dB, o sea donde la salida
    # prácticamente deja de crecer.
    xc = float(p1["Pin_1dB_dBm"]) if p1 else -12.0
    pend = np.gradient(pout, pin)
    idx = np.where((pin > xc) & (pend < 0.2))[0]
    xs = float(pin[idx[0]]) if len(idx) else 0.0
    for lo, hi, col, txt in ((x0, xc, "0.94", "lineal"),
                             (xc, xs, "0.88", "compresión"),
                             (xs, x1, "0.82", "saturación")):
        a1.axvspan(lo, hi, color=col, lw=0, zorder=0)
        # El de saturación no entra centrado en su franja, así que va contra el
        # borde derecho, con un fondo del color de la franja para que no lo
        # cruce la recta de la respuesta lineal.
        xt, ha = ((x1 - 0.6, "right") if txt == "saturación"
                  else ((max(lo, x0) + min(hi, x1)) / 2, "center"))
        a1.text(xt, 27.0, txt, ha=ha, va="top", fontsize=7.5, color="0.35",
                zorder=4, bbox=dict(fc=col, ec="none", pad=0.6))

    a1.plot(pin[m], pideal[m], "--", color="0.55", lw=1.2,
            label="respuesta lineal")
    # Limitador ideal: la respuesta lineal hasta la salida máxima del
    # amplificador y plana desde ahí.
    psat = float(np.max(pout))
    a1.plot(pin[m], np.minimum(pideal[m], psat), ":", color="C2", lw=1.6,
            label="limitador ideal")
    a1.plot(pin[m], pout[m], "-", color="C0", lw=1.8, label="amplificador")
    if p1:
        # El rotulo va en la leyenda y no como anotacion, porque con este
        # lienzo el texto suelto choca con la caja de la leyenda.
        a1.plot(p1["Pin_1dB_dBm"], p1["Pout_1dB_dBm"], "o", color="C3", ms=5,
                zorder=5, label="compresión de 1 dB")
    a1.set_ylabel(r"$P_{out}$ [dB rel.]")
    a1.set_title("conversión AM–AM y regiones de operación",
                  fontsize=9.5)
    a1.legend(loc="lower right", fontsize=7.5, framealpha=0.9,
              handlelength=1.6, borderpad=0.35, labelspacing=0.3)

    a2.axhline(0.0, ls="--", color="0.55", lw=1.2, label="amplificador ideal")
    a2.plot(pin_f, fase, "-", color="C0", lw=1.8)
    a2.set_ylabel(r"desfasaje [$^\circ$]")
    a2.set_title("conversión AM–PM", fontsize=9.5)
    a2.legend(loc="upper left", fontsize=7.5, framealpha=0.9,
              handlelength=1.6, borderpad=0.35, labelspacing=0.3)

    for ax in (a1, a2):
        ax.set_xlabel(r"$P_{in}$ [dB rel.]")
        ax.set_xlim(x0, x1)
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "conversion")


# Punto de operación y back-off

def fig_backoff_bo() -> None:
    """El back-off sobre la característica de transferencia.

    Mismas unidades y mismo estilo que fig_conversion, porque las dos grafican
    la misma curva y aparecen en paginas contiguas. El retroceso se marca sobre
    el eje de entrada, que es donde lo aplica sys_with_acpr/pa.py.

    Sin leyenda a propósito. La figura se incluye al 50 % del ancho del texto y
    una caja de cuatro entradas tapaba la curva, así que los dos puntos van
    rotulados directamente. Que la recta punteada es la respuesta lineal ya lo
    establece la figura anterior.
    """
    from sys_with_acpr.pa import compute_pa_power_curve

    c = compute_pa_power_curve(pa_mode="arctan_mem", pa_nl="arctan",
                               use_memory=False)
    pin, pout, pideal = c["pin_dbm"], c["pout_dbm"], c["pout_ideal_dbm"]
    x1db = float(c["p1db"]["Pin_1dB_dBm"])
    y1db = float(c["p1db"]["Pout_1dB_dBm"])

    BO = 6.0                       # back-off de ejemplo, en dB
    xop = x1db - BO
    yop = float(np.interp(xop, pin, pout))

    # El ancho de texto de la clase es 441 pt = 6.10 in. La figura se incluye
    # al 50 %, o sea 3.05 in, y el lienzo se hace de ese ancho exacto para que
    # LaTeX no la reduzca y la tipografia coincida con la de fig_conversion.
    x0, x1 = -30.0, 2.0
    m = (pin >= x0) & (pin <= x1)

    fig, ax = plt.subplots(figsize=(3.05, 1.95))
    ax.plot(pin[m], pideal[m], "--", color="0.55", lw=1.2)
    ax.plot(pin[m], pout[m], "-", color="C0", lw=1.8)
    ax.plot(x1db, y1db, "o", color="C3", ms=5, zorder=5)
    ax.plot(xop, yop, "o", color="C2", ms=5, zorder=5)

    ax.annotate("compresión\nde 1 dB", (x1db, y1db), textcoords="offset points",
                xytext=(6, -2), fontsize=7.5, color="C3", va="top")
    ax.annotate("punto de\noperación", (xop, yop), textcoords="offset points",
                xytext=(-4, 4), fontsize=7.5, color="C2", ha="right", va="bottom")

    yb = -25.5
    for xx, yy in ((x1db, y1db), (xop, yop)):
        ax.plot([xx, xx], [yb, yy], ":", color="0.5", lw=0.9, zorder=1)
    ax.annotate("", xy=(x1db, yb), xytext=(xop, yb),
                arrowprops=dict(arrowstyle="<->", color="0.25", lw=1.1))
    # El rotulo va debajo de la flecha, no encima, porque arriba lo cruzan las
    # lineas de guia que bajan desde los dos puntos.
    ax.text((x1db + xop) / 2, yb - 0.8, f"BO = {BO:.0f} dB", ha="center",
            va="top", fontsize=7.5, color="0.25")

    ax.set_xlabel(r"$P_{in}$ [dB rel.]")
    ax.set_ylabel(r"$P_{out}$ [dB rel.]")
    ax.set_xlim(x0, x1)
    ax.set_ylim(-30, 0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _save(fig, "backoff_bo")


# Dispersión AM-AM por la memoria

def fig_amam_memoria() -> None:
    """Característica AM-AM con y sin memoria, superpuestas.

    Sin memoria el caso es una curva univoca, así que se dibuja como línea
    encima de la nube del caso con memoria. De ese modo la dispersión se lee
    directamente contra su referencia, en vez de comparar dos paneles.
    Mismas unidades que fig_conversion, potencia relativa en dB.
    """
    from sys_with_acpr.pa import apply_selected_pa

    x, _, p = _lte(19, seed=3)      # unas 83 000 muestras
    fs = p["fs"]

    def curva(mem):
        y, _, _ = apply_selected_pa(
            x, pa_mode=f"arctan_{'mem' if mem else 'nomem'}", pa_nl="arctan",
            use_memory=mem, bo_db=0.0, R0=50.0, fs_in=fs, fs_pa=fs,
            sat_amp=1.0, p_soft=3.0, normalize_input=True,
            normalize_output=False)
        n = min(len(x), len(y))
        pin = 20 * np.log10(np.abs(x[:n]) + 1e-12)
        pout = 20 * np.log10(np.abs(y[:n]) + 1e-12)
        return pin - np.max(pin), pout - np.max(pout)

    pin_s, pout_s = curva(False)
    pin_m, pout_m = curva(True)

    # Se incluye al 68 % de 6.10 in, así que el lienzo va de ese ancho.
    fig, ax = plt.subplots(figsize=(4.15, 2.9))
    ax.plot(pin_m, pout_m, ".", ms=0.7, alpha=0.22, color="C0",
            rasterized=True, label="ecualizador por subportadora")
    o = np.argsort(pin_s)
    ax.plot(pin_s[o], pout_s[o], "-", color="C3", lw=1.4,
            label="sin memoria")
    ax.set_xlabel(r"$P_{in}$ [dB rel.]")
    ax.set_ylabel(r"$P_{out}$ [dB rel.]")
    ax.set_xlim(-35, 2)
    ax.set_ylim(-35, 2)
    ax.grid(True, alpha=0.3)
    leg = ax.legend(loc="lower right", fontsize=8, framealpha=0.9,
                    handlelength=1.6, borderpad=0.35, labelspacing=0.3)
    leg.legend_handles[0].set_alpha(1.0)
    leg.legend_handles[0].set_markersize(5)
    fig.tight_layout()
    _save(fig, "amam_memoria")


# Los dos efectos de la distorsión, fuera y dentro de banda

def fig_efectos_pa() -> None:
    """Los dos efectos de la no linealidad, en un mismo punto de operacion.

    A la izquierda el recrecimiento espectral, a la derecha lo que ve el
    receptor sobre la constelación. Las dos mitades son la misma corrida, de
    modo que el lector compara el mismo experimento en los dos dominios.

    El back-off elegido es -3 dB, o sea por encima del punto de compresión.
    """
    from scipy.signal import welch
    from sys_with_acpr.main import demodulate
    from sys_with_acpr.pa import apply_selected_pa

    BO = -3.0
    x, ideal, p = _lte(42, seed=11)  # unos 25 000 símbolos 16-QAM
    fs = p["fs"]
    main_half = p["main_bw"] / 2
    adj_off = p["adj_offset"]
    y, _, _ = apply_selected_pa(x, pa_mode="arctan_mem", pa_nl="arctan",
                                use_memory=True, bo_db=BO, R0=50.0, fs_in=fs,
                                fs_pa=fs, sat_amp=1.0, p_soft=3.0,
                                normalize_input=True, normalize_output=True)

    fig, (a1, a2) = plt.subplots(
        1, 2, figsize=(6.1, 2.65),
        gridspec_kw={"width_ratios": [1.6, 1.0]})

    # --- fuera de banda ---
    def psd(v):
        f, P = welch(v, fs=fs, nperseg=2048, return_onesided=False,
                     detrend=False)
        i = np.argsort(f)
        return f[i] / 1e6, 10 * np.log10(P[i] + 1e-30)

    fx, Px = psd(x)
    fy, Py = psd(y)
    ref = np.max(Px)
    a1.axvspan(-main_half / 1e6, main_half / 1e6, color="C0", alpha=0.10)
    for c in (-adj_off, adj_off):
        a1.axvspan((c - main_half) / 1e6, (c + main_half) / 1e6,
                   color="C3", alpha=0.08)
    a1.plot(fy, Py - ref, "-", color="C0", lw=0.8, label="con amplificador")
    a1.plot(fx, Px - ref, "--", color="0.25", lw=1.1, label="señal ideal")
    a1.set_xlabel("frecuencia [MHz]")
    a1.set_ylabel("densidad espectral [dB rel.]")
    a1.set_xlim(-fs / 2e6, fs / 2e6)
    a1.set_ylim(-80, 6)
    a1.grid(True, alpha=0.3)
    a1.legend(loc="lower center", fontsize=7, framealpha=0.9,
              handlelength=1.6, borderpad=0.3)
    a1.set_title("fuera de banda", fontsize=9.5)

    # --- dentro de banda ---
    # Ecualización por subportadora, que es lo que hace el receptor OFDM. Sin
    # ella la respuesta no plana del filtro de memoria deforma la constelación
    # en arcos que no son distorsión no lineal.
    g = demodulate(y, p)
    H = np.sum(g * np.conj(ideal), axis=0) / (
        np.sum(np.abs(ideal) ** 2, axis=0) + 1e-30)
    eq = g / H[None, :]
    a2.plot(eq.real.ravel(), eq.imag.ravel(), ".", ms=0.8, alpha=0.25,
            color="C0", rasterized=True)
    pts = np.unique(np.round(ideal.ravel(), 6))
    a2.plot(np.real(pts), np.imag(pts), "x", color="C3", ms=5, mew=1.2)
    a2.set_xlabel("en fase")
    a2.set_ylabel("en cuadratura")
    a2.set_aspect("equal")
    a2.set_xlim(-1.6, 1.6)
    a2.set_ylim(-1.6, 1.6)
    a2.grid(True, alpha=0.3)
    a2.set_title("dentro de banda", fontsize=9.5)

    fig.tight_layout()
    _save(fig, "efectos_pa")


# Estadística de la señal OFDM

def _ofdm_demo(n_sym=4000, seed=7):
    """Señal LTE de las simulaciones. n_sym cuenta símbolos de 320 muestras
    (un símbolo LTE tiene unas 4390). Devuelve dos valores, y el segundo no
    se usa."""
    x, _, _ = _lte(max(1, round(n_sym * 320 / 4390)), seed)
    return x, None


def fig_envolvente_pdf() -> None:
    """Histograma del modulo de la senal OFDM frente a la densidad de Rayleigh."""
    x, _ = _ofdm_demo()
    r = np.abs(x) / np.sqrt(np.mean(np.abs(x) ** 2))    # normalizado a RMS = 1
    sigma = np.sqrt(0.5)                                # Rayleigh con E{r^2} = 1

    # Se incluye al 60 % de 6.10 in, así que el lienzo va de ese ancho.
    fig, ax = plt.subplots(figsize=(3.66, 2.4))
    ax.hist(r, bins=160, density=True, color="C0", alpha=0.55,
            label="muestras OFDM")
    t = np.linspace(0, r.max(), 400)
    ax.plot(t, t / sigma**2 * np.exp(-t**2 / (2 * sigma**2)), "--",
            color="C3", lw=1.8, label="Rayleigh teórica")
    ax.set_xlabel(r"$|x(n)|$ normalizado al valor eficaz")
    ax.set_ylabel("densidad de probabilidad")
    ax.set_xlim(0, min(r.max(), 4.2))
    ax.legend(fontsize=8, framealpha=0.9)
    _coma(ax)
    _coma(ax, "x")
    fig.tight_layout()
    _save(fig, "envolvente_pdf")


def fig_papr_ccdf() -> None:
    """CCDF de la PAPR por símbolo OFDM, para varios números de subportadoras.
    Con 50000 símbolos por curva (20000 la LTE, en tandas para no llenar la
    memoria) la cola se lee hasta 1e-3 con al menos 20 símbolos por encima."""
    from sys_with_acpr.tx import qam16_gray_mod, ofdm_modulate

    def ccdf(papr_db):
        pa = np.sort(papr_db)
        return pa, 1.0 - np.arange(pa.size) / pa.size

    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for Nused, Nfft, col in [(12, 16, "C2"), (26, 32, "C1"), (52, 64, "C0")]:
        rng = np.random.default_rng(11)
        n_sym = 50000
        bits = rng.integers(0, 2, n_sym * Nused * 4)
        x, _, Nfft_os, cp_os = ofdm_modulate(qam16_gray_mod(bits), Nfft=Nfft,
                                             Nused=Nused, cp_len=Nfft // 4,
                                             os_factor=4)
        blk = Nfft_os + cp_os
        b = x[: (len(x) // blk) * blk].reshape(-1, blk)[:, cp_os:]
        papr_db = 10 * np.log10(np.max(np.abs(b) ** 2, axis=1)
                                / np.mean(np.abs(b) ** 2, axis=1))
        ax.semilogy(*ccdf(papr_db), color=col, lw=1.6, label=f"{Nused}")

    # La señal LTE de las simulaciones, 600 subportadoras, sin el filtro para
    # medir la PAPR de cada símbolo sobre la ventana de la FFT. Con el filtro,
    # sobre la ventana del receptor o en bloques contiguos, da lo mismo
    # (mediana 8,90 dB, y el 1 % supera 10,80 dB, en los dos casos).
    from sys_with_acpr import lte
    n, papr_db = lte.NFFT * 4, []
    for tanda in range(10):
        n_sym = 2000
        bits = np.random.default_rng(11 + tanda).integers(0, 2, n_sym * lte.NUSED * 4)
        x, _ = lte.modular(qam16_gray_mod(bits), n_sym)
        pos = 0
        for cp in lte.prefijos(n_sym, 4):
            b = x[pos + cp:pos + cp + n]
            papr_db.append(10 * np.log10(np.max(np.abs(b) ** 2) / np.mean(np.abs(b) ** 2)))
            pos += cp + n
    pa, cc = ccdf(np.asarray(papr_db))
    print(f"  LTE: mediana {np.median(pa):.2f} dB, 1 % supera {np.quantile(pa, 0.99):.2f} dB")
    ax.semilogy(pa, cc, color="C3", lw=1.6, label="600 (LTE)")

    # portadora única de referencia: 16QAM sin OFDM
    ax.axvline(2.55, ls=":", color="0.4", lw=1.5)
    ax.text(2.75, 0.3, "portadora única\n16-QAM", fontsize=8, color="0.4", va="center")

    ax.set_xlabel("umbral de PAPR [dB]")
    ax.set_ylabel("fracción de símbolos")
    ax.set_ylim(1e-3, 1)
    ax.set_xlim(1.5, 13)
    ax.legend(title="subportadoras", title_fontsize=8.5, fontsize=8.5, framealpha=0.9,
              loc="lower left", bbox_to_anchor=(0.0, 0.0))
    fig.tight_layout()
    _save(fig, "papr_ccdf")


# Barrido de M, P, método y back-off

BARRIDO_PKL = REPO / "sys_with_acpr" / "resultados_barrido" / "barrido_mp.pkl"


def _barrido():
    """Resultados de sys_with_acpr/barrido_mp.py, que se corre aparte (11 min)."""
    import pickle
    if not BARRIDO_PKL.exists():
        raise FileNotFoundError(f"falta {BARRIDO_PKL}, correr sys_with_acpr/barrido_mp.py")
    with open(BARRIDO_PKL, "rb") as f:
        d = pickle.load(f)
    return d["meta"], d["res"]


def _acpr_peor(r) -> float:
    """El peor de los dos canales adyacentes. Sin DPD difieren menos de 0,3 dB,
    pero con DPD el inferior llega a quedar casi 5 dB por encima."""
    return max(r["acpr_l"], r["acpr_u"])


def _barrido_bo(corte: str, metrica: str) -> None:
    """EVM, EVM con el receptor escalar o ACPR contra back-off, una curva por
    valor del parámetro barrido.

    Dos paneles con la misma escala, LS a la izquierda y GD a la derecha, y la
    curva sin DPD como referencia. El EVM va en escala logaritmica porque LS
    con orden alto pasa del 80 % y GD a back-off alto baja del 2 %. El piso de la señal ideal, -120,8 dB,
    queda muy por debajo y no se dibuja. El valor elegido (M = 3 o P = 7)
    va con línea mas gruesa, y cada valor tiene su marcador para que se
    distingan donde las curvas se superponen.
    """
    meta, res = _barrido()
    bo = np.asarray(meta["BO_LIST"], dtype=float)
    # La EVM del corte de memoria va en escala lineal: todas sus curvas quedan por debajo del 30 %, y lo que importa (el piso de 6,5 %
    # de M = 1 con el receptor escalar, la zona comprimida) se lee mejor así.
    # Los valores de back-off alto (0,02 a 0,6 %) están en el texto y en el
    # Apéndice B. El corte de orden sigue en log porque LS pasa del 160 %.
    lineal = True
    if corte == "M":
        vals, elegido, clave = meta["M_CORTE"], 3, (lambda m, v: (m, v, 7))
    else:
        vals, elegido, clave = meta["P_CORTE"], 7, (lambda m, v: (m, 3, v))
    valor = (lambda r: r[metrica]) if metrica.startswith("evm") else _acpr_peor
    sin_dpd = [valor(res[("no_dpd", int(b))]) for b in bo]
    colores = _paleta(len(vals))
    marcas = ["o", "s", "^", "D", "v", "P"]

    # Los dos paneles de EVM comparten la escala de 0 a 35 % en los dos cortes. En el de orden LS
    # llega a 85 %: los puntos que se salen se marcan en el borde con su valor.
    propia = False
    tope = 35.0
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=not propia)
    for ax, metodo, tit in [(axs[0], "ls", "(a) indirecto, mínimos cuadrados"),
                            (axs[1], "gd", "(b) directo, descenso de gradiente")]:
        ax.plot(bo, sin_dpd, "--", color="0.35", lw=1.2, label="sin DPD")
        for v, c, mk in zip(vals, colores, marcas):
            y = [valor(res[clave(metodo, v) + (int(b),)]) for b in bo]
            ax.plot(bo, y, "-" + mk, ms=3.8, color=c, lw=2.2 if v == elegido else 1.1,
                    label=f"${corte} = {v}$")
            if metrica.startswith("evm"):
                fuera = [(b, yy) for b, yy in zip(bo, y) if yy > tope]
                for k, (b, yy) in enumerate(fuera):
                    ax.annotate(f"{yy:.0f} %", xy=(b, tope), xytext=(0, -9 - 8 * (v > 9)),
                                textcoords="offset points", ha="center", va="top",
                                fontsize=6.5, color=c, clip_on=False)
                    ax.plot([b], [tope], marker="^", ms=4.5, color=c, clip_on=False)
        if metrica.startswith("evm") and not lineal:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.LogLocator(subs=(1, 2, 5)))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        _coma(ax)
        _coma(ax, "x")
        ax.set_xticks(bo)
        ax.set_xlabel("back-off [dB]")
        ax.set_title(tit, fontsize=10)
    axs[0].set_ylabel({"evm": "EVM [%]", "evm_escalar": "EVM, receptor escalar [%]"}.get(metrica, "ACPR [dB]"))
    if metrica.startswith("evm") and propia:
        for ax in axs:
            ax.set_ylim(bottom=0)
    elif metrica.startswith("evm") and lineal:
        axs[0].set_ylim(0, tope)
    elif metrica.startswith("evm"):
        # El eje arranca en una marca, para que el punto mas bajo se pueda leer.
        ymin = min(np.min(ln.get_ydata()) for ax in axs for ln in ax.get_lines())
        marcas_y = [k * 10.0 ** e for e in range(-3, 3) for k in (1, 2, 5)]
        axs[0].set_ylim(0.9 * max(t for t in marcas_y if t <= ymin), axs[0].get_ylim()[1])
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5, frameon=False,
               handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, f"barrido_{corte}_{metrica}")


def fig_barrido_M_evm_escalar() -> None:
    _barrido_bo("M", "evm_escalar")


def fig_barrido_P_evm_escalar() -> None:
    # El orden se elige con el receptor escalar, como la memoria. Los valores
    # con el ecualizador están en la tabla del Apéndice A.
    _barrido_bo("P", "evm_escalar")


def fig_barrido_M_acpr() -> None:
    _barrido_bo("M", "acpr")


def fig_barrido_P_acpr() -> None:
    _barrido_bo("P", "acpr")


_BO_PEOR = -9   # peor caso de operacion, el que decide la eleccion
_BOS_FIG = (_BO_PEOR, 3)   # PSD y constelaciones: peor caso y uno alto, donde LS anda


def _corte(meta, corte):
    """Valores del corte, valor elegido y funcion (metodo, valor) -> clave."""
    if corte == "M":
        return meta["M_CORTE"], 3, (lambda m, v: (m, v, 7))
    return meta["P_CORTE"], 7, (lambda m, v: (m, 3, v))


def _suave(pdb, n=33):
    """Solo para el dibujo: promedio móvil de la PSD en potencia lineal, para
    que las curvas se distingan. El ACPR se calcula sobre la PSD sin suavizar."""
    lin = 10 ** (np.asarray(pdb, float) / 10)
    lin = np.pad(lin, n // 2, mode="edge")
    return 10 * np.log10(np.convolve(lin, np.ones(n) / n, mode="valid"))


def _barrido_psd_juntos(corte: str, bo: int, sufijo: str = "") -> None:
    """PSD a la salida del PA en un back-off, con el mismo armado que las
    figuras de EVM y ACPR contra back-off: dos paneles, LS a la izquierda y
    GD a la derecha, una curva por valor del corte con los mismos colores, y
    la de sin DPD de referencia. El valor elegido va con línea más gruesa.
    Así se comparan las memorias (u órdenes) dentro de cada método, como en
    el texto."""
    meta, res = _barrido()
    vals, elegido, clave = _corte(meta, corte)
    f = res[("ideal",)]["f"] / 1e6
    p = meta["PARAMS"]
    colores = _paleta(len(vals))
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.1), sharey=True)
    for ax, metodo, tit in [(axs[0], "ls", "(a) indirecto, mínimos cuadrados"),
                            (axs[1], "gd", "(b) directo, descenso de gradiente")]:
        _bandas(ax, p, alterno=False, color="#f6dcdc")
        ax.plot(f, _suave(res[("no_dpd", bo)]["Pxx_dB"]), "--", color="0.35", lw=1.2,
                label="sin DPD")
        for v, c in zip(vals, colores):
            r = res[clave(metodo, v) + (bo,)]
            ax.plot(f, _suave(r["Pxx_dB"]), color=c, lw=2.0 if v == elegido else 1.0,
                    label=f"${corte} = {v}$")
        ax.set_xlim(-20, 20)
        ax.set_ylim(-140, -62)
        ax.set_xticks([-20, -10, 0, 10, 20])
        ax.set_xlabel("frecuencia [MHz]")
        ax.set_title(tit, fontsize=10)
        _coma(ax)
        _coma(ax, "x")
    axs[0].set_ylabel("PSD [dB/Hz]")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=8.5, frameon=False,
               handlelength=1.8, columnspacing=1.2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, f"barrido_{corte}_psd{sufijo}")


def _barrido_const(corte: str, bos=_BOS_FIG, receptor: str = "", sufijo: str = "") -> None:
    """Constelaciones recibidas: fila por back-off y metodo, columna por valor
    del corte, con el EVM de cada caso y los puntos ideales en rojo."""
    meta, res = _barrido()
    vals, elegido, clave = _corte(meta, corte)
    ideal = np.array([(a + 1j * b) / np.sqrt(10) for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
    n = len(vals)
    filas = [(bo, metodo, nombre) for bo in bos for metodo, nombre in [("ls", "LS"), ("gd", "GD")]]
    fig, axs = plt.subplots(len(filas), n, figsize=(1.28 * n + 0.7, 1.5 * len(filas)),
                            sharex=True, sharey=True)
    for i, (bo, metodo, nombre) in enumerate(filas):
        for j, v in enumerate(vals):
            ax = axs[i, j]
            r = res[clave(metodo, v) + (bo,)]
            z = r["const" + receptor]
            ax.plot(z.real, z.imag, ".", ms=1.0, color="C0", alpha=0.5, rasterized=True)
            ax.plot(ideal.real, ideal.imag, "+", ms=4, mew=0.8, color="C3")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            # El recuadro marca la configuración elegida sólo en el peor caso, que
            # es donde se elige. A +3 dB no tiene sentido.
            if v == elegido and metodo == "gd" and bo == _BO_PEOR:
                for sp in ax.spines.values():
                    sp.set_linewidth(2.0)
            e = r["evm" + receptor]
            evm = f"{e:.1f}" if e >= 1 else f"{e:.2f}"
            ax.set_title(f"${corte}={v}$, {evm} %".replace(".", ","), fontsize=7.5, pad=2)
            if j == 0:
                ax.set_ylabel(f"{nombre}, {_dB(bo)}", fontsize=8.5)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.9)
    _save(fig, f"barrido_{corte}_const{sufijo}")


def fig_barrido_M_psd_m9() -> None:
    # Una figura por back-off, cada una junto a su párrafo.
    _barrido_psd_juntos("M", -9, sufijo="_m9")


def fig_barrido_M_psd_p3() -> None:
    _barrido_psd_juntos("M", 3, sufijo="_p3")


def fig_barrido_M_const_m9() -> None:
    _barrido_const("M", bos=(-9,), receptor="_escalar", sufijo="_m9")


def fig_barrido_M_const_p3() -> None:
    _barrido_const("M", bos=(3,), receptor="_escalar", sufijo="_p3")


def fig_barrido_P_psd_m9() -> None:
    # Una figura por back-off, como las de la memoria.
    _barrido_psd_juntos("P", -9, sufijo="_m9")


def fig_barrido_P_psd_p3() -> None:
    _barrido_psd_juntos("P", 3, sufijo="_p3")


def fig_barrido_P_const_m9() -> None:
    _barrido_const("P", bos=(-9,), receptor="_escalar", sufijo="_m9")


def fig_barrido_P_const_p3() -> None:
    _barrido_const("P", bos=(3,), receptor="_escalar", sufijo="_p3")


def fig_barrido_coefs() -> None:
    """Magnitud de los coeficientes del DPD, LS contra GD, en el peor caso.

    M = 3, P = 7: doce coeficientes agrupados por retardo m, y dentro de cada
    grupo por orden p. Los dos métodos tienen el termino lineal sin retardo en
    uno (LS lo normaliza, GD lo deja fijo), así que son comparables.
    """
    meta, res = _barrido()
    w = {m: np.abs(res[(m, 3, 7, _BO_PEOR)]["w"]) for m in ("ls", "gd")}
    ordenes = [1, 3, 5, 7]
    M = 3
    x = np.arange(M * len(ordenes)) + np.repeat(np.arange(M), len(ordenes)) * 0.8
    fig, ax = plt.subplots(figsize=(6.0, 3.1))
    ax.bar(x - 0.2, w["ls"], 0.4, color="C3", label="mínimos cuadrados")
    ax.bar(x + 0.2, w["gd"], 0.4, color="C0", label="descenso de gradiente")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda y, _: f"{y:g}".replace(".", ",")))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p}" for _ in range(M) for p in ordenes])
    for m in range(M):
        ax.text(x[m * len(ordenes):(m + 1) * len(ordenes)].mean(), 1.02,
                f"retardo $m = {m}$", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("orden $p$")
    ax.set_ylabel("$|w_{p,m}|$")
    ax.grid(axis="x", visible=False)
    fig.legend(loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    _save(fig, "barrido_coefs")


def fig_barrido_condicion() -> None:
    """Número de condicion de Phi^H Phi del aprendizaje indirecto contra P.

    La matriz es la que arma train_dpd_ls: regresores sobre la salida del PA
    con memoria, M = 3, con el mismo registro de entrenamiento que el barrido
    (semilla 123, N_train_max muestras). Las columnas se normalizan antes de formar
    la matriz de Gram, para medir colinealidad y no la diferencia de escala
    entre |y| y |y|^10, que corrige cualquier precondicionador diagonal.
    """
    from sys_with_acpr.main import generate_ofdm_signal
    from sys_with_acpr.pa import apply_selected_pa, normalize_rms
    from sys_with_acpr.dpd import build_memory_poly_regressor

    meta, _ = _barrido()
    p = meta["PARAMS"]
    n_bits = p["n_ofdm_symbols"] * p["Nused"] * 4
    bits = np.random.default_rng(123).integers(0, 2, n_bits)
    x, _ = generate_ofdm_signal(bits, p)
    x, _ = normalize_rms(x[:meta["DPD_CFG"]["N_train_max"]], target_rms=1.0)
    Ps = meta["P_CORTE"]
    # Sólo tres back-off: lo que importa es que las curvas
    # casi no se separan entre back-off y suben mucho con P.
    bos = [-9, 0, 9]
    colores = _paleta(len(bos))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for bo, c in zip(bos, colores):
        y, _, _ = apply_selected_pa(x, pa_mode=p["pa_mode"], pa_nl=p["pa_nl"],
                                    use_memory=p["use_memory"], bo_db=bo, R0=p["R0"],
                                    fs_in=p["fs"], fs_pa=p["fs_pa"], sat_amp=p["sat_amp"],
                                    p_soft=p["p_soft"], normalize_input=True,
                                    normalize_output=True)
        cond = []
        for P in Ps:
            Phi, _, _ = build_memory_poly_regressor(y, Pmax=P, M=3)
            Phi = Phi / np.linalg.norm(Phi, axis=0, keepdims=True)
            cond.append(np.linalg.cond(Phi.conj().T @ Phi))
        ax.semilogy(Ps, cond, "-o", ms=3.5, lw=1.2, color=c,
                    label=f"{bo:+d} dB".replace("+0", "0").replace("-", "−"))
    ax.set_xticks(Ps)
    ax.set_xlabel("orden de no linealidad $P$")
    ax.set_ylabel("número de condición de $\\Phi^H\\Phi$")
    ax.legend(title="back-off", fontsize=8, title_fontsize=8, framealpha=0.9)
    _save(fig, "barrido_condicion")


def fig_barrido_costo() -> None:
    """Operaciones de entrenamiento contra número de coeficientes.

    Los contadores salen del propio barrido (16000 muestras, 40 épocas del
    modelo y 5 del DPD), en el peor caso de back-off: no dependen del
    back-off salvo cuando
    entra el respaldo NLMS (P = 11), que suma el intento cortado. Dos paneles:
    creciendo la memoria con P = 7 y creciendo el orden con M = 3.
    """
    meta, res = _barrido()
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0), sharey=True)
    for ax, corte, tit in [(axs[0], "M", "(a) creciendo la memoria, $P = 7$"),
                           (axs[1], "P", "(b) creciendo el orden, $M = 3$")]:
        vals, _, clave = _corte(meta, corte)
        K = [(v if corte == "M" else 3) * ((7 if corte == "M" else v) + 1) // 2 for v in vals]
        for metodo, col, mk, lab in [("ls", "C3", "o", "mínimos cuadrados"),
                                     ("gd", "C0", "s", "descenso de gradiente")]:
            ops = [res[clave(metodo, v) + (_BO_PEOR,)]["ops"] for v in vals]
            ax.semilogy(K, ops, "-" + mk, ms=4, lw=1.6, color=col, label=lab)
        ax.set_xticks(K)
        ax.set_xticklabels([f"{k}\n(${corte}={v}$)" for k, v in zip(K, vals)], fontsize=8)
        ax.set_xlabel("coeficientes $K$")
        ax.set_title(tit, fontsize=10)
    axs[0].set_ylabel("operaciones")
    axs[1].legend(framealpha=0.9, fontsize=8.5)
    fig.tight_layout()
    _save(fig, "barrido_costo")


# Efecto del amplificador sin DPD

EFECTO_PKL = REPO / "sys_with_acpr" / "resultados_efecto_pa" / "efecto_pa.pkl"


def _efecto():
    """Resultados de sys_with_acpr/efecto_pa.py, que se corre aparte."""
    import pickle
    if not EFECTO_PKL.exists():
        raise FileNotFoundError(f"falta {EFECTO_PKL}, correr sys_with_acpr/efecto_pa.py")
    with open(EFECTO_PKL, "rb") as f:
        return pickle.load(f)


def _coma(ax, eje="y"):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def _pb_16qam(x):
    """BER exacta de 16-QAM Gray en ruido blanco, con x = Es/N0 lineal:
    1/4 [3 Q(a) + 2 Q(3a) - Q(5a)], a = sqrt(x/5)."""
    from scipy.special import erfc
    q = lambda v: 0.5 * erfc(v / np.sqrt(2))
    a = np.sqrt(np.asarray(x, float) / 5)
    return 0.25 * (3 * q(a) + 2 * q(3 * a) - q(5 * a))


def fig_efecto_pa_filtros() -> None:
    """Módulo y retardo de grupo de los dos filtros lineales del
    Wiener-Hammerstein y de su cascada, sobre el canal principal y los
    adyacentes.

    Coeficientes de sys_with_acpr/pa.py, que corren a la fs de la simulación
    (61,44 MHz con la señal LTE). En el canal principal la cascada varía
    0,6 dB y su retardo de grupo 0,1 muestras, y en los adyacentes H2 atenúa entre 0,2 y 1,1 dB, que son los
    números que cita el texto.
    """
    from scipy.signal import freqz, group_delay
    p, _ = _barrido()
    p = p["PARAMS"]
    fs = p["fs"]
    from sys_with_acpr.pa import PA_MEMORIA_K as k

    def esp(c):
        """Coeficientes con los retardos espaciados k muestras (z^-1 -> z^-k)."""
        out = np.zeros((len(c) - 1) * k + 1)
        out[::k] = c
        return out

    filtros = [("$H_1$, entrada", esp([1 / 1.5, 0, 0.25 / 1.5]), esp([1, 0.6 / 1.5]), "C0"),
               ("$H_2$, salida", esp([1 / 0.52, -0.1 / 0.52]), esp([1, -0.104 / 0.52]), "C1")]
    lim = 15.0
    f = np.linspace(-lim, lim, 1201)
    w = 2 * np.pi * f * 1e6 / fs
    bw = p["main_bw"] / 2e6
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.0))
    total_h = np.ones_like(w, dtype=complex)
    total_gd = np.zeros_like(w)
    for nom, b, a, c in filtros:
        h = freqz(b, a, worN=w)[1] / freqz(b, a, worN=[0.0])[1][0]
        gd = group_delay((b, a), w=np.mod(w, 2 * np.pi))[1]
        total_h *= h
        total_gd += gd
        axs[0].plot(f, 20 * np.log10(np.abs(h)), color=c, lw=1.3, label=nom)
        axs[1].plot(f, gd, color=c, lw=1.3, label=nom)
    axs[0].plot(f, 20 * np.log10(np.abs(total_h)), "k", lw=1.6, label="cascada")
    axs[1].plot(f, total_gd, "k", lw=1.6, label="cascada")
    for ax in axs:
        ax.axhline(0, color="0.45", lw=1.2, ls="--", label="sin memoria (ideal)")
        ax.axvspan(-bw, bw, color="C0", alpha=0.12, lw=0, zorder=0)
        _bandas(ax, p, alterno=False, color="#f6dcdc")
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("frecuencia [MHz]")
        _coma(ax)
        _coma(ax, "x")
    axs[0].set_ylim(-4, 1)
    axs[0].set_ylabel("ganancia [dB]")
    axs[0].set_title("(a) módulo", fontsize=10)
    axs[1].set_ylabel("retardo de grupo [muestras]")
    axs[1].set_title("(b) retardo de grupo", fontsize=10)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    _save(fig, "efecto_pa_filtros")


def fig_efecto_pa_snr() -> None:
    """BER y EVM contra la SNR por subportadora, con PA en varios back-off.
    La BER va con el receptor ecualizado y la EVM con el escalar, y la
    punteada es el PA sin memoria a 0 dB con el escalar, que muestra lo que agrega la memoria. Referencias teóricas de
    16-QAM en ruido blanco: la BER exacta (_pb_16qam) y EVM = 1/sqrt(SNR)."""
    d = _efecto()
    snr = np.asarray(d["meta"]["SNR_GRILLA"])
    fino = np.linspace(snr.min(), snr.max(), 200)
    lin = 10 ** (fino / 10)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4))
    axs[0].semilogy(fino, _pb_16qam(lin), color="0.6", lw=3, alpha=0.6,
                    label="teórica")
    axs[1].plot(fino, 100 / np.sqrt(lin), color="0.6", lw=3, alpha=0.6, label="teórica")
    # Sin la curva sin PA (la teórica la representa), sin la del PA sin
    # memoria ecualizada (cae sobre la con memoria) y sin marcar
    # los puntos sin errores, para no recargar la figura.
    colores = _paleta(len(d["meta"]["BO_SNR"]))
    # Una muestra de 10^7 bits no resuelve BER menores que 10^-7, y los
    # puntos sin errores no se dibujan.
    for bo, c in zip(d["meta"]["BO_SNR"], colores):
        r = d["snr"][("pa", bo, True)]
        lab = f"PA, {bo:+.0f} dB".replace("+0 ", "0 ").replace("-", "−")
        ber = np.asarray(r["ber"], float)
        ok = ber > 0
        axs[0].semilogy(snr[ok], ber[ok], "-o", ms=3.2, lw=1.1, color=c, label=lab)
        axs[1].plot(snr, r["evm_escalar"], "-o", ms=3.2, lw=1.1, color=c, label=lab)
    c0 = colores[list(d["meta"]["BO_SNR"]).index(0.0)]
    r = d["snr"][("pa", 0.0, False)]
    axs[1].plot(snr, r["evm_escalar"], ":^", ms=3.2, lw=1.1, color=c0,
                label="PA sin memoria, 0 dB")
    axs[0].set_ylim(1e-7, 0.5)
    axs[0].set_ylabel("BER")
    axs[1].set_ylabel("EVM [%]")
    axs[1].set_ylim(0, 105)
    _coma(axs[1])
    for ax, tit in [(axs[0], "(a) tasa de error de bit"), (axs[1], "(b) EVM")]:
        ax.set_xlabel("SNR por subportadora [dB]")
        ax.set_title(tit, fontsize=10)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "efecto_pa_snr")


def fig_efecto_pa_bo() -> None:
    """EVM contra back-off sin ruido, con el receptor escalar, PA con memoria y
    sin memoria. La diferencia entre las dos es lo que agrega la memoria."""
    d = _efecto()
    bo = np.asarray([b for b in d["meta"]["BO_GRILLA"] if b <= 9.0])   # hasta +9 dB
    con = [d["bo"][(b, True)]["evm_escalar"] for b in bo]
    sin = [d["bo"][(b, False)]["evm_escalar"] for b in bo]
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    # Eje lineal, como toda EVM de la tesis.
    ax.plot(bo, con, "-s", ms=3.5, color="C3", lw=1.4, label="con memoria")
    ax.plot(bo, sin, ":", color="C3", lw=1.6, label="sin memoria")
    ax.set_xticks(np.arange(-9, 10, 3))
    ax.set_ylim(0, 25)
    ax.set_xlabel("back-off [dB]")
    ax.set_ylabel("EVM [%]")
    _coma(ax)
    _coma(ax, "x")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save(fig, "efecto_pa_bo")


def fig_efecto_pa_psd() -> None:
    """PSD a la salida del PA con memoria para varios back-off, con la señal
    ideal (piso de la señal de prueba). Bandas sombreadas como en acpr_piso:
    principal en celeste, adyacentes en rosa y alternos en gris."""
    d = _efecto()
    _, resb = _barrido()
    p = d["meta"]["PARAMS"]
    bos = [-9.0, -3.0, 3.0, 9.0]
    colores = _paleta(len(bos))
    m = p["main_bw"] / 2e6
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.axvspan(-m, m, color="C0", alpha=0.12, lw=0, zorder=0)
    _bandas(ax, p, color="#f6dcdc")
    for b, c in zip(bos, colores):
        r = d["bo"][(b, True)]
        ax.plot(r["f"] / 1e6, _suave(r["Pxx_dB"]), color=c, lw=1.2,
                label=f"{b:+.0f} dB".replace("-", "−"))
    ax.plot(resb[("ideal",)]["f"] / 1e6, _suave(resb[("ideal",)]["Pxx_dB"]), ":", color="k",
            lw=1.0, label="señal ideal")
    ax.set_ylim(-185, -52)
    for k, nombre in [(1, "adyacente"), (2, "alterno")]:
        for sg in (-1, 1):
            ax.text(sg * k * p["adj_offset"] / 1e6, -58, nombre, ha="center", va="center",
                    fontsize=7.5)
    ax.text(0, -58, "principal", ha="center", va="center", fontsize=7.5)
    ax.set_xlim(-25, 25)   # canal, adyacentes y alternos
    ax.set_xlabel("frecuencia [MHz]")
    ax.set_ylabel("PSD [dB/Hz]")
    _coma(ax)
    _coma(ax, "x")
    ax.legend(title="back-off", fontsize=8, title_fontsize=8, ncol=5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    _save(fig, "efecto_pa_psd")


def fig_efecto_pa_acpr() -> None:
    """ACPR contra back-off, PA con memoria, en el canal adyacente y en el
    alterno, en un solo panel: con el mismo eje se ve que el alterno baja el
    doble de rápido. La curva sin memoria queda 0,5 dB por encima en todo el
    rango y no se dibuja. Se dibuja el peor de los dos lados: el inferior y el superior
    difieren en menos de 0,03 dB en el adyacente y 0,05 dB en el alterno."""
    d = _efecto()
    bo = np.asarray(d["meta"]["BO_GRILLA"])
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for (l, u), ls, mk, nom in [(("acpr_l", "acpr_u"), "-", "s", "canal adyacente"),
                                (("acpr_alt_l", "acpr_alt_u"), ":", "o", "canal alterno")]:
        peor = [max(d["bo"][(b, True)][l], d["bo"][(b, True)][u]) for b in bo]
        ax.plot(bo, peor, ls + mk, ms=3.2, color="C3", lw=1.3, label=nom)
    ax.set_xticks(np.arange(-9, 16, 3))
    ax.set_xlabel("back-off [dB]")
    ax.set_ylabel("ACPR [dB]")
    _coma(ax)
    _coma(ax, "x")
    ax.legend(fontsize=8.5, framealpha=0.9, loc="lower left")
    fig.tight_layout()
    _save(fig, "efecto_pa_acpr")


# Validación de la cadena

VALIDACION_PKL = REPO / "sys_with_acpr" / "resultados_validacion" / "validacion_canal.pkl"


def fig_validacion_awgn() -> None:
    """BER y EVM contra SNR de la cadena sin PA, sobre las teóricas de 16-QAM
    en ruido blanco. Mismos datos que la curva sin PA de efecto_pa_snr."""
    d = _efecto()
    r = d["snr"][("sin_pa",)]
    snr = np.asarray(r["snr"])
    fino = np.linspace(snr.min(), snr.max(), 200)
    lin = 10 ** (fino / 10)
    ber = np.asarray(r["ber"], float)
    ok = ber > 0
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    axs[0].semilogy(fino, _pb_16qam(lin), color="0.6", lw=3, alpha=0.6, label="teórica")
    axs[0].semilogy(snr[ok], ber[ok], "o", ms=4, color="k", label="simulada")
    # Sin errores en los 10^7 bits: se marcan contra el borde inferior.
    piso = 1e-7
    axs[0].semilogy(snr[~ok], np.full((~ok).sum(), piso * 1.6), "v", ms=4, mfc="none", color="k",
                    clip_on=False, label="sin errores")
    axs[0].set_ylim(piso, 0.5)
    axs[0].set_ylabel("BER")
    axs[1].plot(fino, 100 / np.sqrt(lin), color="0.6", lw=3, alpha=0.6, label="teórica")
    axs[1].plot(snr, r["evm"], "o", ms=4, color="k", label="simulada")
    axs[1].set_ylabel("EVM [%]")
    axs[1].set_ylim(0, 105)
    _coma(axs[1])
    for ax, tit in [(axs[0], "(a) tasa de error de bit"), (axs[1], "(b) EVM")]:
        ax.set_xlabel("SNR por subportadora [dB]")
        ax.set_title(tit, fontsize=10)
    axs[0].legend(fontsize=8, framealpha=0.9, loc="lower left")
    axs[1].legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.tight_layout()
    _save(fig, "validacion_awgn")


def _validacion():
    import pickle
    if not VALIDACION_PKL.exists():
        raise FileNotFoundError(f"falta {VALIDACION_PKL}, correr sys_with_acpr/validacion_canal.py")
    with open(VALIDACION_PKL, "rb") as f:
        return pickle.load(f)


def fig_validacion_canal() -> None:
    """Módulo de la respuesta en frecuencia de una realización del canal
    Rayleigh, con la banda ocupada sombreada. En esta realización, dentro de la banda,
    va de -34,3 a +7,2 dB."""
    d = _validacion()
    e, m = d["ejemplo"], d["meta"]
    fig, ax = plt.subplots(figsize=(5.4, 2.9))
    f = e["f"] / 1e6
    banda = m["PARAMS"]["main_bw"] / 2e6
    ax.axvspan(-banda, banda, color="C0", alpha=0.10, lw=0, zorder=0)
    ax.axhline(0, color="0.4", lw=0.8, ls="--")
    ax.plot(f, 20 * np.log10(np.abs(e["H"])), color="C0", lw=1.0)
    ax.set_xlim(-5.5, 5.5)
    ax.set_xlabel("frecuencia [MHz]")
    ax.set_ylabel("$|H|$ [dB]")
    _coma(ax, "x")
    _coma(ax)
    fig.tight_layout()
    _save(fig, "validacion_canal")


def fig_validacion_rayleigh() -> None:
    """BER con canal Rayleigh, sin diversidad, sobre la teórica, con la de
    ruido blanco de referencia. La tesis no muestra la combinación de 8 ramas
    ni la EVM, que validacion_canal.py también calcula (res[8]). Todos los puntos sin diversidad tienen errores."""
    d = _validacion()
    r = d["res"][1]
    s = np.asarray(r["snr"])
    snr = np.linspace(0, 40, 300)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    ax.semilogy(snr, _pb_16qam(10 ** (snr / 10)), ":", color="0.4", lw=1.2, label="ruido blanco, teórica")
    ax.semilogy(s, r["ber_teo"], color="C3", lw=3, alpha=0.35, label="Rayleigh, teórica")
    ax.semilogy(s, r["ber"], "o", ms=3.5, color="C3", label="Rayleigh, simulada")
    ax.set_ylim(1e-5, 0.5)
    ax.set_xlim(0, 40)
    ax.set_xlabel("SNR media por subportadora [dB]")
    ax.set_ylabel("BER")
    ax.legend(fontsize=8, framealpha=0.9, loc="lower left")
    fig.tight_layout()
    _save(fig, "validacion_rayleigh")


DESEMPENO_PKL = REPO / "sys_with_acpr" / "resultados_desempeno" / "desempeno.pkl"


def _desempeno():
    """Resultados de sys_with_acpr/desempeno.py, que se corre aparte."""
    import pickle
    if not DESEMPENO_PKL.exists():
        raise FileNotFoundError(f"falta {DESEMPENO_PKL}, correr sys_with_acpr/desempeno.py")
    with open(DESEMPENO_PKL, "rb") as f:
        return pickle.load(f)


def _dB(v):
    return f"{v:+.0f} dB".replace("+0 ", "0 ").replace("-", "−")


def fig_desempeno_bo() -> None:
    """EVM con el receptor escalar y ACPR (el peor adyacente) contra back-off,
    con y sin DPD. Sin DPD sale de efecto_pa.py, con la misma grilla, señal de
    evaluación y receptor."""
    d, e = _desempeno(), _efecto()
    _, resb = _barrido()
    bo = np.asarray(d["meta"]["BO_GRILLA"])
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.9))
    for clave, fuente, c, mk, lab in [(lambda b: (b, True), e["bo"], "C3", "s", "sin DPD"),
                                      (lambda b: b, d["bo"], "C2", "o", "con DPD")]:
        rs = [fuente[clave(b)] for b in bo]
        axs[0].plot(bo, [r["evm_escalar"] for r in rs], "-" + mk, ms=3.2, lw=1.3, color=c, label=lab)
        axs[1].plot(bo, [_acpr_peor(r) for r in rs], "-" + mk, ms=3.2, lw=1.3, color=c, label=lab)
    # Sin los límites de la norma: con -9 dB el PA está 9 dB por encima de
    # P1dB, un caso de estrés, y la tesis compara contra el caso sin
    # predistorsión.
    axs[0].set_ylabel("EVM [%]")
    axs[0].set_ylim(0, 25)
    axs[1].set_ylabel("ACPR [dB]")
    for ax, tit in [(axs[0], "(a) EVM"), (axs[1], "(b) ACPR, peor adyacente")]:
        ax.set_xticks(np.arange(-9, 16, 6))
        ax.set_xlabel("back-off [dB]")
        ax.set_title(tit, fontsize=10)
        _coma(ax)
        _coma(ax, "x")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    _save(fig, "desempeno_bo")


def fig_desempeno_const() -> None:
    """Constelaciones recibidas con el receptor escalar, sin y con DPD en tres
    back-off, con la EVM de cada caso. Salen del barrido (const_escalar), cuyo
    GD con M = 3, P = 7 es la configuración elegida y da los mismos valores que
    desempeno.pkl. El escalar deja ver la memoria lineal del PA sin DPD."""
    _, resb = _barrido()
    ideal = np.array([(a + 1j * b) / np.sqrt(10) for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
    bos = [-9, -3, 3]  # los mismos que la PSD
    fig, axs = plt.subplots(2, len(bos), figsize=(4.6, 3.2), sharex=True, sharey=True)
    for i, nombre in enumerate(["sin DPD", "con DPD"]):
        for j, b in enumerate(bos):
            ax = axs[i, j]
            r = resb[("no_dpd", b)] if i == 0 else resb[("gd", 3, 7, b)]
            z = r["const_escalar"]
            ax.plot(z.real, z.imag, ".", ms=1.0, color="C0", alpha=0.5, rasterized=True)
            ax.plot(ideal.real, ideal.imag, "+", ms=4, mew=0.8, color="C3")
            ax.set_xlim(-1.6, 1.6)
            ax.set_ylim(-1.6, 1.6)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            ax.set_title(f"{_dB(b)}, {r['evm_escalar']:.1f} %".replace(".", ","), fontsize=7.5, pad=2)
            if j == 0:
                ax.set_ylabel(nombre, fontsize=9)
    fig.tight_layout(pad=0.3, w_pad=0.2, h_pad=0.9)
    _save(fig, "desempeno_const")


def fig_desempeno_psd() -> None:
    """PSD a la salida del PA sin DPD (gris) y con DPD (verde, como el GD del
    barrido), un panel por back-off, con la señal ideal y el ACPR del peor
    adyacente sin y con DPD en el titulo. Bandas sombreadas como en acpr_piso:
    principal en celeste, adyacentes en rosa y alternos en gris."""
    d, e = _desempeno(), _efecto()
    _, resb = _barrido()
    p = d["meta"]["PARAMS"]
    f = np.asarray(resb[("ideal",)]["f"]) / 1e6
    bos = [-9.0, -3.0, 3.0]
    fig, axs = plt.subplots(1, len(bos), figsize=(6.4, 2.7), sharex=True, sharey=True)
    for ax, b in zip(axs, bos):
        ax.axvspan(-p["main_bw"] / 2e6, p["main_bw"] / 2e6, color="C0", alpha=0.12, lw=0, zorder=0)
        _bandas(ax, p, color="#f6dcdc")
        ax.plot(f, _suave(resb[("ideal",)]["Pxx_dB"]), ":", color="k", lw=0.9)
        ax.plot(f, _suave(e["bo"][(b, True)]["Pxx_dB"]), color="0.45", lw=0.9)
        r = d["bo"][b]
        ax.plot(f, _suave(r["Pxx_dB"]), color="C2", lw=1.2)
        fmt = lambda q: f"{max(q['acpr_l'], q['acpr_u']):.1f}".replace(".", ",").replace("-", "−")  # noqa: E731
        ax.set_title(f"{_dB(b)}, ACPR {fmt(e['bo'][(b, True)])} → {fmt(r)} dB", fontsize=7.5, pad=3)
        ax.set_xlim(-25, 25)   # canal, adyacentes y alternos
        ax.set_ylim(-165, -65)   # la ideal sigue de largo, ninguna del PA baja de −160
        ax.set_xlabel("frec. [MHz]")
        _coma(ax)
        _coma(ax, "x")
    axs[0].set_ylabel("PSD [dB/Hz]")
    h = [plt.Line2D([], [], color="0.45", lw=1.2), plt.Line2D([], [], color="C2", lw=1.4),
         plt.Line2D([], [], ls=":", color="k", lw=1.2)]
    fig.tight_layout(rect=(0, 0.09, 1, 1), w_pad=0.6)
    fig.legend(h, ["sin DPD", "con DPD", "señal ideal"], loc="lower center", ncol=3,
               fontsize=8.5, frameon=False)
    _save(fig, "desempeno_psd")


def fig_desempeno_snr() -> None:
    """BER y EVM contra la SNR por subportadora, con y sin DPD en los mismos
    back-off, sobre los mismos bloques y el mismo ruido (efecto_pa.py). La BER
    va con el receptor ecualizado y la EVM con el escalar, como en
    efecto_pa_snr."""
    d, e = _desempeno(), _efecto()
    snr = np.asarray(d["meta"]["SNR_GRILLA"])
    fino = np.linspace(snr.min(), snr.max(), 200)
    lin = 10 ** (fino / 10)
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 3.4))   # como efecto_pa_snr
    axs[0].semilogy(fino, _pb_16qam(lin), color="0.6", lw=3, alpha=0.6, label="teórica")
    axs[1].plot(fino, 100 / np.sqrt(lin), color="0.6", lw=3, alpha=0.6, label="teórica")
    colores = _paleta(len(d["meta"]["BO_SNR"]))
    curvas = [(r, ls, mk, c, lab) for bo, c in zip(d["meta"]["BO_SNR"], colores)
              for r, ls, mk, lab in [(e["snr"][("pa", bo, True)], "--", "x", None),
                                     (d["snr"][bo], "-", "o", _dB(bo))]]
    # Los puntos sin errores en los 10^7 bits no se dibujan, como en
    # efecto_pa_snr.
    for r, ls, mk, c, lab in curvas:
        ber = np.asarray(r["ber"], float)
        ok = ber > 0
        axs[0].semilogy(snr[ok], ber[ok], ls + mk, ms=3.2, lw=1.1, color=c, label=lab)
        axs[1].plot(snr, r["evm_escalar"], ls + mk, ms=3.2, lw=1.1, color=c, label=lab)
    for ax in axs:
        ax.plot([], [], "--x", ms=3.2, color="0.4", lw=1.1, label="sin DPD")
        ax.plot([], [], "-o", ms=3.2, color="0.4", lw=1.1, label="con DPD")
    axs[0].set_ylim(1e-7, 0.5)
    axs[0].set_ylabel("BER")
    axs[1].set_ylabel("EVM [%]")
    axs[1].set_ylim(0, 105)
    _coma(axs[1])
    for ax, tit in [(axs[0], "(a) tasa de error de bit"), (axs[1], "(b) EVM")]:
        ax.set_xlabel("SNR por subportadora [dB]")
        ax.set_title(tit, fontsize=10)
    h, l = axs[1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    _save(fig, "desempeno_snr")


def _potencia_salida_entrenamiento(params, bo, n=None):
    """Potencia media de la salida del PA sobre el registro de entrenamiento
    (N_train_max de DPD_CFG, 16000, semilla del barrido),
    sin normalizar, como la ve _train_pa_model_gd: sirve para pasar su MSE a
    NMSE."""
    import sys
    sys.path.insert(0, str(REPO))
    from sys_with_acpr import barrido_mp, dpd as _dpd
    from sys_with_acpr.main import generate_ofdm_signal
    bits_train, _ = barrido_mp._bits()
    tx, _ = generate_ofdm_signal(bits_train, params)
    n = n or barrido_mp.DPD_CFG["N_train_max"]
    x, _ = _dpd.normalize_rms(np.asarray(tx[:n], complex), target_rms=1.0)
    y, _, _ = _dpd.apply_selected_pa(
        x, pa_mode=params["pa_mode"], pa_nl=params["pa_nl"], use_memory=params["use_memory"],
        bo_db=bo, R0=params["R0"], fs_in=params["fs"], fs_pa=params["fs_pa"],
        sat_amp=params["sat_amp"], p_soft=params["p_soft"], normalize_input=True,
        normalize_output=False)
    return float(np.mean(np.abs(y) ** 2))


def _conv_figura(curvas, limite, ylabel, nombre, desde=1):
    """Una curva por back-off contra la época, con la línea punteada en el
    número de épocas que usa el entrenamiento. desde = 0 si la primera
    muestra es antes de entrenar."""
    bos = [-9.0, -3.0, 3.0, 9.0]
    colores = _paleta(len(bos))
    fig, ax = plt.subplots(figsize=(5.0, 2.6))
    for b, c in zip(bos, colores):
        v = curvas[b]
        ax.plot(np.arange(desde, desde + len(v)), v, "-", color=c, lw=1.3, label=_dB(b))
    ax.axvline(limite, color="0.3", ls=":", lw=1.2)
    ax.set_xlabel("época")
    ax.set_ylabel(ylabel)
    _coma(ax)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h, l = ax.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, fontsize=8.5, frameon=False)
    _save(fig, nombre)


def fig_desempeno_conv_modelo() -> None:
    """NMSE por época de la identificación del modelo del PA, contra la salida
    medida (el MSE de dpd.py dividido por la potencia de esa salida), con 160
    épocas, de sys_with_acpr/epocas_pa.py. Las primeras 40 son exactamente las
    del entrenamiento de la tesis, y la línea marca ese límite."""
    import pickle
    d = _desempeno()
    ruta = REPO / "sys_with_acpr" / "resultados_identificacion" / "epocas_pa.pkl"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}, correr sys_with_acpr/epocas_pa.py")
    with open(ruta, "rb") as fh:
        ep = pickle.load(fh)
    curvas = {b: 10 * np.log10(np.asarray(ep["res"][(160, int(b))]["mse_pa"])
                               / _potencia_salida_entrenamiento(d["meta"]["PARAMS"], b))
              for b in [-9.0, -3.0, 3.0, 9.0]}
    _conv_figura(curvas, 40, "NMSE del modelo [dB]", "desempeno_conv_modelo")


def fig_desempeno_conv_dpd() -> None:
    """NMSE (Ec. eq:nmse) entre la salida del PA real y la señal original, con
    la ganancia alineada, después de cada época del DPD, de
    sys_with_acpr/epocas_dpd.py (40 épocas, M = 3, P = 7); la línea marca las
    5 del entrenamiento. El MSE del lazo del DPD pasa por el modelo y en
    compresión llega a 1e9 aunque el DPD mejore, así que no se dibuja."""
    import pickle
    ruta = REPO / "sys_with_acpr" / "resultados_identificacion" / "epocas_dpd.pkl"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}, correr sys_with_acpr/epocas_dpd.py")
    with open(ruta, "rb") as fh:
        ep = pickle.load(fh)
    # Arranca en la época 0, el DPD sin entrenar (igual a sin DPD), para que se
    # vea la convergencia.
    curvas = {b: [x["nmse"] for x in ep[(3, int(b))]["por_epoca"]]
              for b in [-9.0, -3.0, 3.0, 9.0]}
    _conv_figura(curvas, 5, "NMSE con el PA real [dB]", "desempeno_conv_dpd", desde=0)


def fig_robustez_divergencia() -> None:
    """(a) NMSE por época del modelo del PA a -9 dB, sin guarda, en los 200
    registros de sys_with_acpr/robustez.py --registros 200 (MSE dividido por
    la potencia de la salida del PA de cada registro). (b) Paso efectivo
    máximo del ajuste, mu_h sum_i |phi_i|^2 / P_i, contra el pico de amplitud
    de cada registro (valor eficaz 1), con el límite de 2 del NLMS y la franja
    entre el menor paso de los que divergen y el mayor de los que convergen.
    Diverge si en algún back-off el error termina más de mil veces por encima
    de su mínimo, el criterio de la guarda."""
    import pickle
    import sys
    sys.path.insert(0, str(REPO))
    from sys_with_acpr import robustez as R
    ruta = REPO / "sys_with_acpr" / "resultados_robustez" / "robustez_200.pkl"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}, correr sys_with_acpr/robustez.py --registros 200")
    with open(ruta, "rb") as fh:
        d = pickle.load(fh)
    semillas = d["meta"]["SEMILLAS"]
    p = R.PARAMS

    def potencia(s):
        y, _, _ = R.apply_selected_pa(
            R.registro(s), pa_mode=p["pa_mode"], pa_nl=p["pa_nl"], use_memory=p["use_memory"],
            bo_db=-9, R0=p["R0"], fs_in=p["fs"], fs_pa=p["fs_pa"], sat_amp=p["sat_amp"],
            p_soft=p["p_soft"], normalize_input=True, normalize_output=False)
        return float(np.mean(np.abs(y) ** 2))

    def diverge_en(m):
        m = np.asarray(m)
        f = m[np.isfinite(m)]
        return len(f) < len(m) or m[-1] > 1e3 * np.min(f)

    diverge = [s for s in semillas
               if any(diverge_en(d["mse"][(s, b)]) for b in d["meta"]["BO_LIST"])]
    c_ok, c_div = "0.7", PALETA[3]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for s in semillas:
        m = 10 * np.log10(np.asarray(d["mse"][(s, -9)]) / potencia(s))
        div = s in diverge
        axs[0].plot(np.arange(1, len(m) + 1), m, "-", color=c_div if div else c_ok,
                    lw=0.8 if div else 0.6, alpha=0.8, zorder=3 if div else 2)
    axs[0].set_ylim(-30, 60)
    axs[0].set_xlabel("época")
    axs[0].set_ylabel("NMSE del modelo [dB]")
    axs[0].set_title("(a) error del modelo a −9 dB", fontsize=9)
    for s in semillas:
        r = d["registros"][s]
        div = s in diverge
        axs[1].plot(r["pico"], r["g_pico"], "o", ms=3,
                    color=c_div if div else c_ok, zorder=3 if div else 2)
    g_div = min(d["registros"][s]["g_pico"] for s in diverge)
    g_ok = max(d["registros"][s]["g_pico"] for s in semillas if s not in diverge)
    axs[1].axhspan(g_div, g_ok, color="0.93", lw=0, zorder=1)
    axs[1].axhline(2, color="0.3", ls=":", lw=1.2)
    axs[1].set_ylim(0, 12.5)
    axs[1].set_xlabel("pico de amplitud")
    axs[1].set_ylabel("paso efectivo máximo")
    axs[1].set_title("(b) paso en la muestra del pico", fontsize=9)
    for ax in axs:
        _coma(ax)
    _coma(axs[1], "x")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h = [plt.Line2D([], [], color=c_ok, lw=1.2, marker="o", ms=4),
         plt.Line2D([], [], color=c_div, lw=1.2, marker="o", ms=4)]
    fig.legend(h, [f"converge ({len(semillas) - len(diverge)})", f"diverge ({len(diverge)})"],
               loc="lower center", ncol=2, fontsize=8.5, frameon=False)
    _save(fig, "robustez_divergencia")


def fig_robustez_guarda() -> None:
    """Los 34 registros de sys_with_acpr/robustez.py --registros 200 en que el
    modelo del PA diverge, con la cadena completa a -9 dB sin guarda y con
    guarda y respaldo (robustez_200.pkl, "cadena"). (a) NMSE por época del
    modelo, el MSE dividido por la potencia de la salida del PA de cada
    registro; con guarda es la historia del respaldo NLMS, que reemplaza a la
    del lazo cortado. (b) EVM de cada registro contra su paso efectivo máximo;
    las que pasan del eje van como triángulos en el borde. La EVM es la del
    receptor escalar."""
    import pickle
    import sys
    sys.path.insert(0, str(REPO))
    from sys_with_acpr import robustez as R
    ruta = REPO / "sys_with_acpr" / "resultados_robustez" / "robustez_200.pkl"
    if not ruta.exists():
        raise FileNotFoundError(f"falta {ruta}, correr sys_with_acpr/robustez.py --registros 200")
    with open(ruta, "rb") as fh:
        d = pickle.load(fh)
    p = R.PARAMS
    semillas = sorted({s for (s, g) in d["cadena"] if not g and s != 123})

    def potencia(s):
        y, _, _ = R.apply_selected_pa(
            R.registro(s), pa_mode=p["pa_mode"], pa_nl=p["pa_nl"], use_memory=p["use_memory"],
            bo_db=-9, R0=p["R0"], fs_in=p["fs"], fs_pa=p["fs_pa"], sat_amp=p["sat_amp"],
            p_soft=p["p_soft"], normalize_input=True, normalize_output=False)
        return float(np.mean(np.abs(y) ** 2))

    c_sin, c_con = PALETA[3], PALETA[0]
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5))
    for s in semillas:
        pot = potencia(s)
        for g, c in [(False, c_sin), (True, c_con)]:
            m = 10 * np.log10(np.asarray(d["cadena"][(s, g)]["mse_pa"]) / pot)
            axs[0].plot(np.arange(1, len(m) + 1), m, "-", color=c, lw=0.7, alpha=0.8)
    axs[0].set_ylim(-30, 60)
    axs[0].set_xlabel("época")
    axs[0].set_ylabel("NMSE del modelo [dB]")
    axs[0].set_title("(a) error del modelo a −9 dB", fontsize=9)
    tope = 35.0
    for s in semillas:
        gp = d["registros"][s]["g_pico"]
        e = [d["cadena"][(s, g)]["evm_escalar"] for g in (False, True)]
        axs[1].plot([gp, gp], [min(e[0], tope), e[1]], "-", color="0.75", lw=0.8, zorder=1)
        for v, c in zip(e, (c_sin, c_con)):
            if v > tope:
                axs[1].plot(gp, tope, "^", ms=4, color=c, zorder=3, clip_on=False)
            else:
                axs[1].plot(gp, v, "o", ms=3, color=c, zorder=3)
    axs[1].axhline(22.8, color="0.3", ls=":", lw=1.2)
    axs[1].set_ylim(0, tope)
    axs[1].set_xlabel("paso efectivo máximo")
    axs[1].set_ylabel("EVM [%]")
    axs[1].set_title("(b) EVM a −9 dB", fontsize=9)
    for ax in axs:
        _coma(ax)
    _coma(axs[1], "x")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h = [plt.Line2D([], [], color=c_sin, lw=1.2, marker="o", ms=4),
         plt.Line2D([], [], color=c_con, lw=1.2, marker="o", ms=4),
         plt.Line2D([], [], color="0.3", ls=":", lw=1.2)]
    fig.legend(h, ["sin guarda", "con guarda y respaldo", "sin DPD"],
               loc="lower center", ncol=3, fontsize=8.5, frameon=False)
    _save(fig, "robustez_guarda")


FIGS = {
    "desempeno_conv_modelo": fig_desempeno_conv_modelo,
    "desempeno_conv_dpd": fig_desempeno_conv_dpd,
    "robustez_divergencia": fig_robustez_divergencia,
    "robustez_guarda": fig_robustez_guarda,
    "desempeno_bo": fig_desempeno_bo,
    "desempeno_const": fig_desempeno_const,
    "desempeno_psd": fig_desempeno_psd,
    "desempeno_snr": fig_desempeno_snr,
    "validacion_awgn": fig_validacion_awgn,
    "validacion_canal": fig_validacion_canal,
    "validacion_rayleigh": fig_validacion_rayleigh,
    "efecto_pa_filtros": fig_efecto_pa_filtros,
    "efecto_pa_snr": fig_efecto_pa_snr,
    "efecto_pa_bo": fig_efecto_pa_bo,
    "efecto_pa_psd": fig_efecto_pa_psd,
    "efecto_pa_acpr": fig_efecto_pa_acpr,
    "barrido_M_evm_escalar": fig_barrido_M_evm_escalar,
    "barrido_P_evm_escalar": fig_barrido_P_evm_escalar,
    "barrido_P_psd_m9": fig_barrido_P_psd_m9,
    "barrido_P_psd_p3": fig_barrido_P_psd_p3,
    "barrido_P_const_m9": fig_barrido_P_const_m9,
    "barrido_P_const_p3": fig_barrido_P_const_p3,
    "barrido_M_acpr": fig_barrido_M_acpr,
    "barrido_P_acpr": fig_barrido_P_acpr,
    "barrido_M_psd_m9": fig_barrido_M_psd_m9,
    "barrido_M_psd_p3": fig_barrido_M_psd_p3,
    "barrido_M_const_m9": fig_barrido_M_const_m9,
    "barrido_M_const_p3": fig_barrido_M_const_p3,
    "barrido_coefs": fig_barrido_coefs,
    "barrido_condicion": fig_barrido_condicion,
    "barrido_costo": fig_barrido_costo,
    "conversion": fig_conversion,
    "backoff_bo": fig_backoff_bo,
    "amam_memoria": fig_amam_memoria,
    "efectos_pa": fig_efectos_pa,
    "envolvente_pdf": fig_envolvente_pdf,
    "papr_ccdf": fig_papr_ccdf,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="",
                    help="lista separada por comas: " + ", ".join(FIGS))
    args = ap.parse_args()

    names = [s.strip() for s in args.only.split(",") if s.strip()] or list(FIGS)
    ok, failed = [], []
    for n in names:
        if n not in FIGS:
            print(f"[skip] figura desconocida: {n}")
            continue
        print(f"[{n}]")
        try:
            FIGS[n]()
            ok.append(n)
        except Exception as e:  # noqa: BLE001
            failed.append(n)
            print(f"  FALLO: {e}")
            traceback.print_exc(limit=1)

    print(f"\nGeneradas: {', '.join(ok) or '(ninguna)'}")
    if failed:
        print(f"Pendientes: {', '.join(failed)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
