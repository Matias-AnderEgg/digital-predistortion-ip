"""Figuras de cuantización del núcleo (cuantizacion_psd y cuantizacion_const).

Con la señal LTE y la cadena de las simulaciones: el predistorsionador elegido se
entrena como en el barrido (sys_with_acpr/barrido_mp.py, M = 3, P = 7, BO -9
dB, registro de semilla 123) y se evalúa sobre la misma señal de 364 símbolos
(semilla 456), con el receptor escalar. Los mismos coeficientes se aplican en
flotante, que es el resultado del barrido, y con el modelo de punto
fijo del núcleo (fixed_point/dpd.py, que da bit a bit lo mismo que la
placa), con los coeficientes en
Q0.15. Todo lo demás de la cadena es lo mismo en los dos casos.

Lo que imprime, con el entrenamiento de la tesis (DPD_CFG de
sys_with_acpr/barrido_mp.py), BO -9 dB, 364 símbolos (entre paréntesis la
EVM con el ecualizador por subportadora):
    sin DPD                    EVM 22,83 % (22,44 %)  ACPR -20,36 dB  alterno -39,44 dB
    DPD en flotante            EVM  4,46 % ( 3,71 %)  ACPR -31,24 dB  alterno -45,45 dB
    DPD en el núcleo (Q0.15)   EVM  4,45 % ( 3,70 %)  ACPR -31,24 dB  alterno -45,44 dB
    diferencia flotante - núcleo en la constelación: -60,8 dB

Los otros números de la escala de salida (pico, recortes, ruido de
cuantización) salen de experimentos/analisis_salida_zf.py.

Uso: PYTHONDONTWRITEBYTECODE=1 python figuras/cuantizacion_zf.py
"""
import contextlib
import io
import os
import sys

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "fixed_point", "golden_vectors"))
sys.path.insert(0, RAIZ)
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import generate as GV  # noqa: E402
from fixed_point import fxp_format as fxp  # noqa: E402
from fixed_point.dpd import apply_memory_poly_fxp  # noqa: E402
from sys_with_acpr import dpd as D  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, _bits  # noqa: E402
from sys_with_acpr.main import (eval_chain, generate_ofdm_signal,  # noqa: E402
                                simulate_no_dpd, simulate_with_dpd)

SALIDA = os.path.dirname(os.path.abspath(__file__))
BO = -9.0
M, P = 3, 7

bits_tr, bits = _bits()
cfg = dict(DPD_CFG, M=M, Pmax=P)
with contextlib.redirect_stdout(io.StringIO()):
    r_sin = simulate_no_dpd(bits, PARAMS, BO)
    r_flot = simulate_with_dpd(bits_tr, bits, PARAMS, BO, method="gd", dpd_cfg=cfg)
w = r_flot["train_out"]["dpd_coeffs"]
w_q = fxp.quantize(w, GV.CFG.coeff, GV.CFG)

# El núcleo: el mismo apply_dpd_and_pa de las simulaciones, con el polinomio
# en punto fijo.
flotante = D.apply_memory_poly
D.apply_memory_poly = lambda x, c, Pmax, M: apply_memory_poly_fxp(x, c, Pmax=Pmax, M=M, cfg=GV.CFG)
tx, grilla = generate_ofdm_signal(bits, PARAMS)
with contextlib.redirect_stdout(io.StringIO()):
    _, y_fx, info = D.apply_dpd_and_pa(
        tx, w_q, pa_mode=PARAMS["pa_mode"], pa_nl=PARAMS["pa_nl"], use_memory=PARAMS["use_memory"],
        bo_db=BO, fs_in=PARAMS["fs"], fs_pa=PARAMS["fs_pa"], R0=PARAMS["R0"], Pmax=P, M=M,
        sat_amp=PARAMS["sat_amp"], p_soft=PARAMS["p_soft"], normalize_dpd_input=True,
        normalize_dpd_output=True, normalize_pa_input=False)
    r_fx = eval_chain(y_fx, grilla, bits, PARAMS, BO, info)
D.apply_memory_poly = flotante

casos = {"sin DPD": r_sin, "DPD en punto flotante": r_flot, "DPD en el núcleo (Q0.15)": r_fx}
for n, r in casos.items():
    acpr = max(r["acpr_l_db"], r["acpr_u_db"])
    alt = max(r["acpr_alt_l_db"], r["acpr_alt_u_db"])
    print(f"{n:26s} EVM {r['evm_escalar_percent']:6.3f} % (ecualizada {r['evm_percent']:6.3f} %)  BER {r['ber']:.2e}  ACPR {acpr:7.3f} dB  "
          f"alterno {alt:6.2f} dB")
d = r_flot["qam_rx_escalar"] - r_fx["qam_rx_escalar"]
ref = r_flot["qam_tx_ref"]
print("diferencia flotante - nucleo en la constelacion, relativa:",
      f"{10 * np.log10(np.mean(np.abs(d) ** 2) / np.mean(np.abs(ref) ** 2)):.1f} dB")

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 10,
                            "axes.grid": True, "grid.linestyle": "--", "grid.alpha": 0.4,
                            "savefig.bbox": "tight"})


def coma(v, d=2):
    return f"{v:.{d}f}".replace(".", ",").replace("-", "−")


def coma_ejes(ax):
    fmt = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)


def suave(pdb, n=33):  # solo para el dibujo, como _suave de genera_figuras.py
    lin = np.pad(10 ** (np.asarray(pdb, float) / 10), n // 2, mode="edge")
    return 10 * np.log10(np.convolve(lin, np.ones(n) / n, mode="valid"))


est = {"sin DPD": ("C3", "-", 1.0), "DPD en punto flotante": ("C0", "-", 1.2),
       "DPD en el núcleo (Q0.15)": ("C1", "--", 1.2)}
fig, ax = plt.subplots(figsize=(6.0, 3.4))
ax.axvspan(-PARAMS["main_bw"] / 2e6, PARAMS["main_bw"] / 2e6, color="C0", alpha=0.12, lw=0, zorder=0)
for k, c in [(1, "#f6dcdc"), (2, "0.96")]:
    a0 = (k * PARAMS["adj_offset"] - PARAMS["adj_bw"] / 2) / 1e6
    a1 = (k * PARAMS["adj_offset"] + PARAMS["adj_bw"] / 2) / 1e6
    for sg in (-1, 1):
        ax.axvspan(sg * a0, sg * a1, color=c, lw=0, zorder=0)
for n, r in casos.items():
    c, ls, lw = est[n]
    acpr = max(r["acpr_l_db"], r["acpr_u_db"])
    ax.plot(np.asarray(r["f"]) / 1e6, suave(r["Pxx_dB"]), ls, color=c, lw=lw,
            label=f"{n}, ACPR {coma(acpr, 1)} dB")
ax.set_xlim(-25, 25)   # canal, adyacentes y alternos, como el resto
ax.set_xlabel("frecuencia [MHz]")
ax.set_ylabel("PSD [dB/Hz]")
coma_ejes(ax)
leg = ax.legend(fontsize=8, ncol=2, frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0),
                handlelength=1.8, columnspacing=1.2)
for ln in leg.get_lines():
    ln.set_linewidth(max(ln.get_linewidth(), 1.4))
fig.tight_layout()
fig.savefig(os.path.join(SALIDA, "cuantizacion_psd.pdf"))

# constelaciones como las de desempeno_const
ideal = np.array([(a + 1j * b) / np.sqrt(10) for a in (-3, -1, 1, 3) for b in (-3, -1, 1, 3)])
fig, axs = plt.subplots(1, 3, figsize=(6.0, 2.3))
idx = np.linspace(0, ref.size - 1, 4000).astype(int)
for a, (n, r) in zip(axs, casos.items()):
    p = r["qam_rx_escalar"][idx]
    a.plot(p.real, p.imag, ".", ms=1.0, color="C0", alpha=0.5, rasterized=True)
    a.plot(ideal.real, ideal.imag, "+", ms=5, mew=0.9, color="C3")
    # un decimal: 4,46 y 4,45 % son la misma EVM (Gastón, 20/11/2026)
    a.set_title(f"{n}\nEVM {coma(r['evm_escalar_percent'], 1)} %", fontsize=8, pad=3)
    a.set_xlim(-1.6, 1.6)
    a.set_ylim(-1.6, 1.6)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.grid(False)
fig.tight_layout(pad=0.3, w_pad=0.6)
fig.savefig(os.path.join(SALIDA, "cuantizacion_const.pdf"), dpi=300)
print("listo")
