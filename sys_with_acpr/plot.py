import numpy as np
import matplotlib.pyplot as plt
from sys_with_acpr.acpr import compute_psd

EPS = 1e-30

_CASES = {
    "ideal":   {"label": "Ideal",       "color": "black",      "marker": "x", "ls": ":"},
    "no_dpd":  {"label": "Sin DPD",     "color": "tab:blue",   "marker": "o", "ls": "-"},
    "ls":      {"label": "DPD-LS",      "color": "tab:orange", "marker": "s", "ls": "--"},
    "gd":      {"label": "DPD-GD",      "color": "tab:green",  "marker": "^", "ls": "-."},
}

def _case_style(key):
    return _CASES.get(key, {"label": key, "color": None, "marker": "o", "ls": "-"})

def _db20_abs(x, ref=1.0):
    x = np.asarray(x)
    return 20.0 * np.log10(np.abs(x) / (float(ref) + EPS) + EPS)

def compute_papr_db(x):
    x = np.asarray(x)
    p = np.abs(x) ** 2
    return float(10.0 * np.log10((np.max(p) + EPS) / (np.mean(p) + EPS)))


# Métricas vs BO
def plot_acpr_vs_bo(results_dict, pa_mode=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for key, results in results_dict.items():
        if not results:
            continue
        st = _case_style(key)
        bo = [r["bo_db"] for r in results]
        acl = [r["acpr_l_db"] for r in results]
        acu = [r["acpr_u_db"] for r in results]
        kw = dict(color=st["color"], marker=st["marker"], linestyle=st["ls"])
        axes[0].plot(bo, acl, label=st["label"], **kw)
        axes[1].plot(bo, acu, label=st["label"], **kw)

    for ax, title in zip(axes, ["ACPR canal inferior", "ACPR canal superior"]):
        ax.set_xlabel("Back-Off BO [dB]")
        ax.set_ylabel("ACPR [dB]")
        ax.set_title(f"{title} ({pa_mode.upper()})")
        ax.grid(True, alpha=0.35)
        ax.legend()

    fig.tight_layout()
    return fig


def plot_evm_ber_vs_bo(results_dict, pa_mode=""):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for key, results in results_dict.items():
        if not results:
            continue
        st = _case_style(key)
        bo = [r["bo_db"] for r in results]
        evm = [r["evm_percent"] for r in results]
        ber = [r["ber"] for r in results]
        kw = dict(color=st["color"], marker=st["marker"], linestyle=st["ls"])
        axes[0].plot(bo, evm, label=st["label"], **kw)
        axes[1].plot(bo, ber, label=st["label"], **kw)

    axes[0].set_xlabel("Back-Off BO [dB]")
    axes[0].set_ylabel("EVM RMS [%]")
    axes[0].set_title(f"EVM vs BO ({pa_mode.upper()})")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].set_xlabel("Back-Off BO [dB]")
    axes[1].set_ylabel("BER")
    axes[1].set_title(f"BER vs BO ({pa_mode.upper()})")
    axes[1].set_yscale("log")
    axes[1].grid(True, which="both", alpha=0.35)
    axes[1].legend()

    fig.tight_layout()
    return fig

# AM-AM

def plot_am_am_curve(curve, op_points, pa_mode):
    """
    Curva AM-AM en dB relativos.

    Este gráfico muestra solamente la curva CW/tono del PA.
    No se dibujan OP promedio de OFDM porque esos puntos no tienen
    por qué caer sobre una curva AM-AM medida con amplitud constante.

    """
    fig, ax = plt.subplots(figsize=(9, 7))

    pin = curve["pin_dbm"]
    pout = curve["pout_dbm"]
    pout_ideal = curve.get("pout_ideal_dbm")

    ax.plot(pin, pout, linewidth=2.2, label="PA real")

    if pout_ideal is not None:
        ax.plot(pin, pout_ideal, "--", linewidth=1.5, color="gray", label="Lineal ideal")

    p1db = curve.get("p1db")
    if p1db is not None and "Pin_1dB_dBm" in p1db:
        ax.scatter(p1db["Pin_1dB_dBm"], p1db["Pout_1dB_dBm"], s=110, marker="v",
                   color="red", edgecolors="k", linewidths=0.8, zorder=6, label="P1dB")
        ax.annotate("P1dB", (p1db["Pin_1dB_dBm"], p1db["Pout_1dB_dBm"]),
                    xytext=(8, 8), textcoords="offset points")

    if curve.get("clip_point") is not None:
        cp = curve["clip_point"]
        ax.scatter(cp["Pin_clip_dBm"], cp["Pout_clip_dBm"], s=120, marker="X",
                   color="crimson", edgecolors="k", linewidths=0.8, zorder=6, label="Psat (BO=0)")

    pin_ref = curve.get("Pin_ref_dBm", None)
    if pin_ref is not None:
        ax.axvline(pin_ref, color="black", linestyle=":", linewidth=1.2, alpha=0.8, label="Referencia BO=0")
    ax.set_xlabel("Pin [dB relativo]")
    ax.set_ylabel("Pout [dB relativo]")
    ax.set_title(f"Curva AM-AM del PA ({pa_mode.upper()})")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_am_am_all_pas(curves_dict):
    fig, ax = plt.subplots(figsize=(10, 7))

    style_map = {
        "fixed": {"label": "W-H",  "color": "tab:blue"},
        "soft":  {"label": "Soft", "color": "tab:green"},
        "clip":  {"label": "Clip", "color": "tab:red"},
    }

    first_curve = next(iter(curves_dict.values()))
    if "pout_ideal_dbm" in first_curve:
        ax.plot(first_curve["pin_dbm"],first_curve["pout_ideal_dbm"],"--",color="black",linewidth=1.8,label="Lineal ideal",)

    for pa_mode, curve in curves_dict.items():
        st = style_map.get(pa_mode, {"label": pa_mode, "color": None})
        ax.plot(curve["pin_dbm"], curve["pout_dbm"], color=st["color"], linewidth=2.2, label=st["label"])

        p1db = curve.get("p1db")
        if p1db is not None:
            ax.scatter(p1db["Pin_1dB_dBm"], p1db["Pout_1dB_dBm"], s=120, marker="v",
                       color=st["color"], edgecolors="k", linewidths=0.8, zorder=6,
                       label=f"P1dB {st['label']}")

    ax.axhline(float(next(iter(curves_dict.values()))["psat_point"]["Pout_sat_dBm"])
               if curves_dict and "psat_point" in next(iter(curves_dict.values())) else 0.0,
               color="gray", linestyle="--", linewidth=1.0, alpha=0.6, label="Psat ref")
    ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Pin [dB relativo]")
    ax.set_ylabel("Pout [dB relativo]")
    ax.set_title("Curvas AM-AM — Psat igualada para los 3 PA")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig


# PSD

def plot_psd_per_bo(results_dict, bo_list, params, pa_mode=""):
    active_keys = [k for k in ("no_dpd", "ls", "gd") if results_dict.get(k)]
    if not active_keys:
        return []

    indexed = {key: {round(float(r["bo_db"]), 6): r for r in results} for key, results in results_dict.items()}

    main_bw = params["main_bw"] / 1e6
    adj_offset = params["adj_offset"] / 1e6
    adj_bw = params["adj_bw"] / 1e6
    main_bw_hz = params["main_bw"]

    figs = []

    for bo in bo_list:
        bkey = round(float(bo), 6)
        cases = [
            (key, indexed[key][bkey])
            for key in active_keys
            if bkey in indexed.get(key, {})
        ]
        if not cases:
            continue

        # Referencia dBc: nivel in-band del primer caso (todos tienen ~igual
        # potencia in-band). Usamos la mediana para que el plateau quede en ~0 dBc
        # sin que un pico del peine de subportadoras corra la referencia.
        r0 = cases[0][1]
        inband = np.abs(r0["f"]) <= main_bw_hz / 2
        ref0 = float(np.median(r0["Pxx_dB"][inband])) if np.any(inband) else 0.0

        # Proporción pensada para la figura del informe: al insertarla a
        # ancho de texto los ejes quedan del mismo tamaño que en el resto
        # de las figuras. Guardar con fig.savefig(..., bbox_inches="tight")
        # en PDF, no imprimir a A4 desde el visor.
        fig, ax = plt.subplots(figsize=(8.2, 4.1))

        ax.axvspan(-main_bw / 2, main_bw / 2, color="lightblue", alpha=0.20, label="Banda principal")
        ax.axvspan(adj_offset - adj_bw / 2, adj_offset + adj_bw / 2, color="lightsalmon", alpha=0.25, label="Canal adyacente")
        ax.axvspan(-adj_offset - adj_bw / 2, -adj_offset + adj_bw / 2, color="lightsalmon", alpha=0.25)

        # Referencia ideal: PSD de la señal SIN PA = piso de fuga de la OFDM.
        # Lo que quede por encima de esta curva es regrowth del PA; el DPD debería
        # empujar hacia ella.
        if "tx_signal" in r0:
            f_id, Pid_dB, _ = compute_psd(r0["tx_signal"], fs=params["fs"], nperseg=8192)
            ax.plot(f_id / 1e6, Pid_dB - ref0, color="black", linestyle=":",
                    linewidth=2.0, alpha=0.85, zorder=20, label="Ideal (señal sin PA)")

        for key, r in cases:
            st = _case_style(key)
            ax.plot(
                r["f"] / 1e6,
                r["Pxx_dB"] - ref0,
                linewidth=1.7,
                linestyle=st["ls"],
                color=st["color"],
                label=(
                    f"{st['label']} | "
                    f"ACPRL={r['acpr_l_db']:.2f} dB | "
                    f"ACPRU={r['acpr_u_db']:.2f} dB | "
                    f"EVM={r['evm_percent']:.2f}%"
                ),
            )

        ax.set_title(f"PSD de salida del PA — BO = {bo} dB — {pa_mode.upper()}")
        ax.set_xlabel("Frecuencia [MHz]")
        ax.set_ylabel("PSD [dBc/Hz]")
        ax.set_ylim(-40, 5)
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=9, loc="best")
        fig.tight_layout()
        figs.append(fig)

    return figs


# Constelaciones

def plot_constellations(results_dict, bo_list, pa_mode="", n_show=2500):
    indexed = {key: {round(float(r["bo_db"]), 6): r for r in results} for key, results in results_dict.items()}

    figs = []
    for bo in bo_list:
        bkey = round(float(bo), 6)
        cases_present = [
            (k, indexed[k][bkey])
            for k in ("ideal", "no_dpd", "ls", "gd")
            if bkey in indexed.get(k, {})
        ]
        if not cases_present:
            continue

        ncols = len(cases_present)
        fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
        if ncols == 1:
            axes = [axes]

        for ax, (key, r) in zip(axes, cases_present):
            ref = r["qam_tx_ref"][:n_show]
            rx = r["qam_rx_aligned"][:n_show]
            st = _case_style(key)

            ax.scatter(np.real(ref), np.imag(ref), s=5, alpha=0.35, label="Ideal")
            ax.scatter(np.real(rx), np.imag(rx), s=5, alpha=0.50, color=st["color"], label="Recibida")

            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.set_title(
                f"{st['label']}\n"
                f"BO={r['bo_db']:.0f} dB | EVM={r['evm_percent']:.2f}% | BER={r['ber']:.2e}"
            )
            ax.set_aspect("equal")
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.grid(True, alpha=0.3)
            ax.legend(markerscale=2)

        fig.suptitle(f"Constelación 16QAM-OFDM — BO = {bo} dB ({pa_mode.upper()})", fontsize=12)
        fig.tight_layout()
        figs.append(fig)

    return figs


# Recorte del PA en el tiempo
def plot_pa_time_clipping(
    x_pa_in,
    y_pa_out,
    *,
    bo_db=None,
    n0=0,
    N=600,
    title=None,
    clip_level=None,
    align_output_gain=False,
):
    """
    Diagnóstico temporal RAW del PA referido al RMS de entrada.

    Convención visual:
    - 0 dB = RMS de |x_pa_in| en el tramo mostrado.
    - Para clip sin memoria y BO=0, el nivel de clip aparece en 0 dB.
    - Para BO=6 dB, aparece en +6 dB.
    - Para BO=12 dB, aparece en +12 dB.

    Nota para PA con memoria:
    - La rama LTI del Wiener-Hammerstein puede introducir ganancia/atenuación.
    - Si align_output_gain=True, la salida se corrige sólo para el gráfico
      mediante una ganancia compleja LS calculada en el tramo mostrado.
    """
    x = np.asarray(x_pa_in, dtype=np.complex128).reshape(-1)
    y = np.asarray(y_pa_out, dtype=np.complex128).reshape(-1)

    n1 = min(n0 + N, len(x), len(y))
    n = np.arange(n0, n1)

    xs = x[n0:n1]
    ys = y[n0:n1]

    ref = np.sqrt(np.mean(np.abs(xs) ** 2) + EPS)

    gain_db = 0.0

    if align_output_gain:
        g_xy = np.vdot(ys, xs) / (np.vdot(ys, ys) + EPS)
        ys_plot = ys * g_xy
        gain_db = 20.0 * np.log10(np.abs(g_xy) + EPS)
    else:
        ys_plot = ys

    x_db = _db20_abs(xs, ref=ref)
    y_db = _db20_abs(ys_plot, ref=ref)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(n,x_db,linewidth=1.7,label="Entrada física PA |x_pa[n]|",)

    y_label = "Salida RAW PA |y_raw[n]|"
    if align_output_gain:
        y_label += f" alineada LS ({gain_db:+.2f} dB)"
    ax.plot(n,y_db,linewidth=1.7,label=y_label,)
    ax.axhline(0.0,linestyle="--",linewidth=1.3,color="black",label="RMS entrada tramo = 0 dB",)
    if clip_level is not None:
        clip_db = 20.0 * np.log10(float(clip_level) / ref + EPS)
        ax.axhline(clip_db,linestyle=":",linewidth=1.8,color="red",label=f"Nivel de clip = {clip_db:.2f} dB rel. RMS entrada",)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Muestra n")
    ax.set_ylabel("Amplitud [dB rel. RMS entrada]")

    if title is None:
        title = "Diagnóstico temporal RAW del PA"
        if bo_db is not None:
            title += f" — BO = {bo_db} dB"

    ax.set_title(title)
    ax.legend()
    fig.tight_layout()

    return fig

def plot_ofdm_am_am_cloud(results_dict, bo_list, params, pa_mode="", n_show=15000):
    active_keys = [k for k in ("no_dpd", "ls", "gd") if results_dict.get(k)]
    if not active_keys:
        return []

    figs = []

    for bo_db in bo_list:
        cases = []
        for key in active_keys:
            r = next((rr for rr in results_dict.get(key, []) if rr["bo_db"] == bo_db), None)
            if r is not None and "u_dpd" in r and "y_pa" in r:
                cases.append((key, r))

        if not cases:
            continue

        fig, axes = plt.subplots(1, len(cases), figsize=(6 * len(cases), 5.5), sharex=True, sharey=True)
        if len(cases) == 1:
            axes = [axes]

        for ax, (key, r) in zip(axes, cases):
            st = _case_style(key)

            u = np.asarray(r["u_dpd"], dtype=np.complex128).reshape(-1)
            y = np.asarray(r["y_pa"], dtype=np.complex128).reshape(-1)

            n = min(len(u), len(y), n_show)
            u = u[:n]
            y = y[:n]

            u_rms = np.sqrt(np.mean(np.abs(u) ** 2) + EPS)
            y_rms = np.sqrt(np.mean(np.abs(y) ** 2) + EPS)

            u_db = 20*np.log10(np.abs(u)/(u_rms + EPS) + EPS)
            y_db = 20*np.log10(np.abs(y)/(y_rms + EPS) + EPS)

            ax.scatter(u_db, y_db, s=4, alpha=0.18, color=st["color"], edgecolors="none")
            ax.axvline(0.0, color="black", linestyle=":", linewidth=1.0)
            ax.axhline(0.0, color="black", linestyle=":", linewidth=1.0)
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{st['label']}\nBO={bo_db} dB | PAPR={compute_papr_db(u):.2f} dB")
            ax.set_xlabel("Entrada PA |u[n]| [dB rel. RMS]")
            ax.set_ylabel("Salida PA |y[n]| [dB rel. RMS]")

        fig.suptitle(f"Nube AM-AM OFDM — {pa_mode.upper()} — BO={bo_db} dB")
        fig.tight_layout()
        figs.append(fig)

    return figs
