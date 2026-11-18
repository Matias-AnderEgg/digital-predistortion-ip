"""Prueba en la placa con la señal LTE de 10 MHz de las simulaciones de la tesis.

Repite en la placa la cadena completa con la LTE de las simulaciones de la
tesis, para que las métricas de la placa se puedan comparar con las de las
simulaciones:

  entrenamiento  las primeras 16000 muestras de la señal de semilla 123,
                 como N_train_max en sys_with_acpr/barrido_mp.py
  evaluación     la primera subtrama (14 símbolos, 61440 muestras) de la
                 señal de semilla 456, la misma con que se evalúa la tesis

La señal es la bajada LTE de fixed_point/lte.py: 16-QAM, FFT de 1024,
sobremuestreo 4 (61,44 MSPS) y el filtro de transmisión. La subtrama no
entra en la BRAM del reproductor (8192 muestras), así que entrenar_lte.c la
pasa por el IP de a trozos (ver ahí).

Uso:
  python lte_placa.py preparar [--bo -9]   escribe lte/lte_ent.bin, lte/lte_eval_in.bin y lte/lte_ref.h
  python lte_placa.py medir [--bo -9]      lee lte/lte_salida.bin, la compara y mide
  python lte_placa.py figura               PSD de las dos corridas, -9 y +3 dB
  python lte_placa.py separar [--bo -9]    de donde sale la diferencia con la simulación

`preparar` deja además lte/esperada_bo<BO>.npz, con lo que `medir` necesita
y no viaja a la placa. De los .bin y .npz solo se versionan las salidas de
la placa de las dos corridas, lte/salida_bo-9.bin y lte/salida_bo3.bin.

Medición, igual que eval_chain en sys_with_acpr/main.py: el PA simulado con
fixed_bo, NMSE directo con la ganancia alineada, EVM con un ecualizador por
subportadora sin realinear, y ACPR del canal adyacente y del alterno con
Welch de 8192 muestras. Con una sola subtrama el ecualizador estima cada
subportadora con 14 símbolos, y no con 364 como en la tesis.

`separar` corre solo en la Mac. Con los mismos coeficientes que la placa,
aplica el DPD en flotante, con los coeficientes en Q0.15 y el camino de
datos en flotante, y todo en punto fijo (lo que hace el IP, bit a bit con
la placa), sobre la subtrama y sobre los 364 símbolos de la tesis. Así se
ve cuanto cuesta el punto fijo y cuanto medir con una sola subtrama.
"""
import dataclasses
import argparse
import ctypes
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, AQUI)

import entrenamiento as E  # noqa: E402
from fixed_point import lte  # noqa: E402
from fixed_point.acpr import compute_acpr, compute_psd  # noqa: E402
from fixed_point.tx import qam16_gray_mod  # noqa: E402

T, GV, dpd, fxp = E.T, E.GV, E.dpd, E.fxp
DIR = os.path.join(AQUI, "lte")
RESULTADOS = os.path.join(AQUI, "resultados", "lte")
N_ENT = 16000
N_SIM_TOTAL = 364          # largo de las corridas de la tesis
N_SIM_EVAL = 14            # una subtrama de 1 ms
N_EVAL = lte.largo(N_SIM_EVAL)
PARAMS = dict(lte.params(), n_ofdm_symbols=N_SIM_EVAL)


def senal(semilla):
    """La senal de la tesis entera (364 simbolos), filtrada y con RMS 1."""
    bits = np.random.default_rng(semilla).integers(0, 2, N_SIM_TOTAL * lte.NUSED * 4)
    x, grilla = lte.modular(qam16_gray_mod(bits), N_SIM_TOTAL)
    return lte.filtrar(x, lte.filtro_tx()), grilla


def pa(u, bo, normalizar_salida):
    """El PA simulado con fixed_bo, como en entrenamiento.py."""
    kw = dict(T.pa_kwargs(PARAMS), bo_db=bo)
    y, _, _ = T.apply_selected_pa(u, **kw, normalize_input=True,
                                  normalize_output=normalizar_salida)
    return y


def entrenar_c(bo):
    """El mismo C que corre la placa, en la Mac, con la configuración por
    defecto. Devuelve los datos de entrenamiento, h, w y w en Q0.15."""
    # Las N_ENT muestras se renormalizan a RMS 1, como train_dpd_gd en la
    # tesis y T.señal con la otra señal. Sin esto quedan en RMS 1,014
    # (+0,12 dB), el DPD se entrena para otra escala de amplitud y a -9 dB
    # la EVM sube de 3,10 a 3,62 %.
    x_ent = senal(123)[0][:N_ENT]
    x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
    y_ent = np.ascontiguousarray(pa(x_ent, bo, False))
    lib = T.cargar_lib()
    cfg = T.Cfg()
    lib.dpd_train_cfg_default(ctypes.byref(cfg))
    h = np.zeros(12, np.complex128)
    w = np.zeros(12, np.complex128)
    lib.dpd_train_pa_model(x_ent, y_ent, N_ENT, ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
    lib.dpd_train_dla(x_ent, N_ENT, h, ctypes.byref(cfg), w, np.zeros(cfg.ep_dpd))
    re = np.zeros(12, np.int16)
    im = np.zeros(12, np.int16)
    lib.dpd_coef_to_q15(w, re, im)
    w_q = (re.astype(float) + 1j * im.astype(float)) / 32768.0
    return x_ent, y_ent, h, w, re, im, w_q


def preparar(bo):
    os.makedirs(DIR, exist_ok=True)
    x_ent, y_ent, h, w, re, im, w_q = entrenar_c(bo)
    with open(os.path.join(DIR, "lte_ent.bin"), "wb") as f:
        f.write(np.int32(N_ENT).tobytes())
        f.write(x_ent.astype("<c16").tobytes())
        f.write(y_ent.astype("<c16").tobytes())

    x_todo, grilla = senal(456)
    x = x_todo[:N_EVAL]
    xi, xq = GV.to_code(fxp.quantize(x / GV.G, GV.CFG.data, GV.CFG))
    E.empaquetar(xi, xq).astype("<u4").tofile(os.path.join(DIR, "lte_eval_in.bin"))
    # La subtrama entera de una vez: es lo que tiene que dar el IP de a trozos.
    ui, uq = E.salida_esperada(x, w_q)
    np.savez(os.path.join(DIR, f"esperada_bo{bo:g}.npz"), x=x, grilla=grilla[:N_SIM_EVAL],
             w_q=w_q, salida=E.empaquetar(ui, uq).astype(np.uint32))

    with open(os.path.join(DIR, "lte_ref.h"), "w") as fh:
        fh.write(f"/* Generado por lte_placa.py preparar --bo {bo:g}. No editar. */\n")
        fh.write("#ifndef LTE_REF_H\n#define LTE_REF_H\n\n#include <stdint.h>\n\n")
        fh.write(f"#define BO_DB  {bo:g}\n#define N_ENT  {N_ENT}\n#define N_EVAL {N_EVAL}\n\n")
        fh.write("/* Lo que dio el mismo C en la Mac. */\n")
        dbl = lambda v: f"{v:.17g}"  # noqa: E731
        E.arreglo(fh, "double", "H_REF", np.column_stack([h.real, h.imag]).ravel(), dbl)
        E.arreglo(fh, "double", "W_REF", np.column_stack([w.real, w.imag]).ravel(), dbl)
        E.arreglo(fh, "uint32_t", "COEFS_REF", E.empaquetar(re, im), lambda v: f"0x{int(v):08X}U")
        fh.write("#endif\n")
    print(f"lte/lte_ent.bin, lte/lte_eval_in.bin y lte/lte_ref.h escritos, BO {bo:g} dB, "
          f"{N_ENT} muestras de entrenamiento y {N_EVAL} de evaluacion")


def ecualizar(ref, rx):
    """Forzador de ceros por subportadora, como equalize_per_subcarrier."""
    H = np.sum(rx * np.conj(ref), axis=0) / (np.sum(np.abs(ref) ** 2, axis=0) + 1e-30)
    return rx / (H[None, :] + 1e-30)


def metricas(x, grilla, y):
    """grilla trae un renglon por simbolo: la subtrama o la senal entera."""
    y = np.vdot(y, x) / np.vdot(y, y) * y
    nmse = 10 * np.log10(np.mean(np.abs(x - y) ** 2) / np.mean(np.abs(x) ** 2))
    rx = lte.demodular(y, grilla.shape[0], adelanto=lte.adelanto_filtro())
    eq = ecualizar(grilla, rx)
    evm = 100 * np.sqrt(np.mean(np.abs(eq - grilla) ** 2) / np.mean(np.abs(grilla) ** 2))
    f, psd_db, pxx = compute_psd(y, fs=PARAMS["fs"], nperseg=8192)
    bw, off = PARAMS["main_bw"], PARAMS["adj_offset"]
    acpr = max(compute_acpr(f, pxx, bw, off, bw))
    alt = max(compute_acpr(f, pxx, bw, 2 * off, bw))
    return nmse, evm, acpr, alt, f, psd_db


def medir(bo):
    d = np.load(os.path.join(DIR, f"esperada_bo{bo:g}.npz"))
    x, grilla = d["x"], d["grilla"]
    hw = np.fromfile(os.path.join(DIR, "lte_salida.bin"), dtype="<u4")
    assert len(hw) == N_EVAL, f"{len(hw)} muestras, se esperaban {N_EVAL}"
    iguales = int(np.sum(hw == d["salida"]))
    print(f"salida del IP: {iguales} de {N_EVAL} muestras iguales al modelo de Python")

    re_, im_ = E.desempaquetar(hw)
    u_hw = (re_ + 1j * im_) / 32768.0 * GV.G_OUT
    filas = [("senal limpia", metricas(x, grilla, x)),
             ("sin DPD", metricas(x, grilla, pa(x, bo, True))),
             ("con el IP", metricas(x, grilla, pa(u_hw, bo, True)))]
    print(f"BO {bo:g} dB, subtrama LTE de {N_EVAL} muestras:")
    print(f"  {'':14s} {'NMSE':>8s} {'EVM':>7s} {'ACPR':>7s} {'alterno':>8s}")
    for nombre, (nm, ev, ac, al, _, _) in filas:
        print(f"  {nombre:14s} {nm:7.2f}  {ev:5.2f}%  {ac:6.1f}  {al:7.1f}")
    # Copia por BO, para la figura: la proxima corrida pisa lte_salida.bin.
    hw.tofile(os.path.join(DIR, f"salida_bo{bo:g}.bin"))
    return iguales == N_EVAL


def separar(bo):
    """Mismos coeficientes, flotante contra punto fijo, subtrama contra 364."""
    _, _, _, w, _, _, w_q = entrenar_c(bo)
    flot = dataclasses.replace(GV.CFG, enabled=False)
    aplicar = lambda x, c, cfg: dpd.apply_memory_poly_fxp(  # noqa: E731
        x, c, Pmax=GV.PMAX, M=GV.M_TAPS, cfg=cfg)
    x_todo, g_todo = senal(456)
    x, g = x_todo[:N_EVAL], g_todo[:N_SIM_EVAL]
    casos = [("subtrama", "sin DPD", x, g, x),
             ("subtrama", "DPD en flotante", x, g, aplicar(x, w, flot)),
             ("subtrama", "w en Q0.15, camino en flotante", x, g, aplicar(x, w_q, flot)),
             ("subtrama", "todo en punto fijo (el IP)", x, g, aplicar(x, w_q, GV.CFG)),
             ("364 simbolos", "sin DPD", x_todo, g_todo, x_todo),
             ("364 simbolos", "DPD en flotante", x_todo, g_todo, aplicar(x_todo, w, flot))]
    print(f"BO {bo:g} dB, coeficientes del C en la Mac (los de la placa):")
    print(f"  {'':14s} {'':32s} {'NMSE':>8s} {'EVM':>7s} {'ACPR':>7s} {'alterno':>8s}")
    for largo, nombre, xx, gg, u in casos:
        nm, ev, ac, al, _, _ = metricas(xx, gg, pa(u, bo, True))
        print(f"  {largo:14s} {nombre:32s} {nm:7.2f}  {ev:5.2f}%  {ac:6.1f}  {al:7.1f}", flush=True)


def figura(bos=(-9.0, 3.0)):
    """PSD a la salida del PA simulado, sin DPD y con la salida real del IP.

    Relativa a la media dentro del canal, con Welch de 2048 muestras (30 kHz,
    unos 60 promedios en la subtrama), mas suave que la de 8192 con que se
    mide el ACPR. Deja resultados/lte/placa_lte_psd.{pdf,png}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.signal import welch
    bw, off, fs = PARAMS["main_bw"], PARAMS["adj_offset"], PARAMS["fs"]

    def psd_rel(y):
        f, p = welch(y, fs=fs, window="hann", nperseg=2048, return_onesided=False)
        f, p = np.fft.fftshift(f), np.fft.fftshift(p)
        return f / 1e6, 10 * np.log10(p / np.mean(p[np.abs(f) < bw / 2]))

    fig, ejes = plt.subplots(1, len(bos), figsize=(11, 3.8), sharey=True)
    for ax, bo in zip(np.atleast_1d(ejes), bos):
        d = np.load(os.path.join(DIR, f"esperada_bo{bo:g}.npz"))
        x, grilla = d["x"], d["grilla"]
        re_, im_ = E.desempaquetar(np.fromfile(os.path.join(DIR, f"salida_bo{bo:g}.bin"), dtype="<u4"))
        u_hw = (re_ + 1j * im_) / 32768.0 * GV.G_OUT
        for nombre, u, color in (("sin DPD", x, "C3"), ("con el IP", u_hw, "C0")):
            y = pa(u, bo, True)
            acpr = metricas(x, grilla, y)[2]
            f, p = psd_rel(y)
            ax.plot(f, p, color=color, lw=1.2,
                    label=f"{nombre}, ACPR {acpr:.1f} dB".replace(".", ","))
        for s in (-1, 1):
            ax.axvspan((s * off - bw / 2) / 1e6, (s * off + bw / 2) / 1e6, color="0.92", zorder=0)
            ax.text(s * off / 1e6, -82, "adyacente", ha="center", fontsize=8, color="0.4")
        ax.set_title(f"BO {bo:+g} dB")
        ax.set_xlabel("frecuencia [MHz]")
        ax.set_xlim(-30, 30)
        ax.set_ylim(-85, 5)
        ax.set_xticks(np.arange(-30, 31, 10))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    np.atleast_1d(ejes)[0].set_ylabel("PSD relativa al canal [dB]")
    fig.tight_layout()
    os.makedirs(RESULTADOS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(RESULTADOS, f"placa_lte_psd.{ext}"), dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("accion", choices=("preparar", "medir", "figura", "separar"))
    ap.add_argument("--bo", type=float, default=-9.0)
    a = ap.parse_args()
    if a.accion == "preparar":
        preparar(a.bo)
    elif a.accion == "medir":
        sys.exit(0 if medir(a.bo) else 1)
    elif a.accion == "separar":
        separar(a.bo)
    else:
        figura()
