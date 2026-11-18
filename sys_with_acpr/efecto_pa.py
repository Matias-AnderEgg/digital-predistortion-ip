"""Efecto del amplificador sin predistorsión, para los efectos de memoria y las
consecuencias de la distorsión sobre el enlace.

Misma cadena, PA y receptor que main.py (receptor ecualizado por
subportadora). Calcula y guarda:

1) Contra el back-off, sin ruido, con y sin memoria: EVM con el receptor
   ecualizado y con el escalar (para mostrar la diferencia), BER,
   ACPR_L/U y PSD.

2) Contra la SNR, sin PA y con PA en algunos back-off: BER y EVM. El ruido
   blanco gaussiano complejo se suma sobre cada subportadora después de la
   FFT, con varianza Es/SNR, donde Es es la potencia media recibida por
   subportadora, de modo que la SNR es Es/N0 por subportadora.
   Cada punto acumula bloques de 364 símbolos LTE con datos distintos hasta
   50 errores de bit o 1e7 bits, lo que llegue primero.

Uso:
    python sys_with_acpr/efecto_pa.py [--jobs 8]
Escribe sys_with_acpr/resultados_efecto_pa/efecto_pa.pkl.
"""
import argparse
import contextlib
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

from sys_with_acpr.main import generate_ofdm_signal, demodulate  # noqa: E402
from sys_with_acpr.pa import apply_selected_pa  # noqa: E402
from sys_with_acpr.rx import (evm_rms_percent,  # noqa: E402
                              equalize_per_subcarrier, evm_percent, qam16_gray_demod)
from sys_with_acpr.acpr import compute_psd, compute_acpr  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_efecto_pa", "efecto_pa.pkl")

BO_GRILLA = [float(b) for b in np.arange(-9.0, 15.1, 1.5)]
BO_SNR = [-9.0, -3.0, 0.0, 3.0]    # back-off de las curvas contra SNR
SNR_GRILLA = [float(s) for s in np.arange(0.0, 32.1, 2.0)]
MIN_ERRORES = 50
MAX_BITS = 1e7
N_BLOQUES = 12                     # 12 x 8,7e5 bits = 1,05e7 bits como tope
SEED_BO = 456                      # la misma señal de evaluación que el barrido


def _bits(seed):
    n = PARAMS["n_ofdm_symbols"] * PARAMS["Nused"] * 4
    return np.random.default_rng(seed).integers(0, 2, n)


def _pa(x, bo, memoria):
    p = dict(PARAMS)
    y, _, _ = apply_selected_pa(x, pa_mode="arctan_" + ("mem" if memoria else "nomem"),
                                pa_nl="arctan", use_memory=memoria, bo_db=bo, R0=p["R0"],
                                fs_in=p["fs"], fs_pa=p["fs_pa"], sat_amp=p["sat_amp"],
                                p_soft=p["p_soft"], normalize_input=True, normalize_output=True)
    return y


def _grilla(y):
    return demodulate(y, PARAMS)


def punto_bo(tarea):
    """Un back-off, con o sin memoria, sin ruido."""
    bo, memoria = tarea
    bits = _bits(SEED_BO)
    with contextlib.redirect_stdout(io.StringIO()):
        x, tx_grid = generate_ofdm_signal(bits, PARAMS)
        y = _pa(x, bo, memoria)
    g = _grilla(y)
    evm_esc, _ = evm_rms_percent(tx_grid.ravel(), g.ravel())
    eq = equalize_per_subcarrier(tx_grid, g)
    evm_eq = evm_percent(tx_grid, eq)
    bits_rx, _ = qam16_gray_demod(eq.ravel())
    f, Pdb, P = compute_psd(y, fs=PARAMS["fs"], nperseg=8192)
    al_l, al_u = compute_acpr(f, P, main_bw=PARAMS["main_bw"], adj_offset=PARAMS["adj_offset"],
                              adj_bw=PARAMS["adj_bw"])
    alt_l, alt_u = compute_acpr(f, P, main_bw=PARAMS["main_bw"], adj_offset=2 * PARAMS["adj_offset"],
                                adj_bw=PARAMS["adj_bw"])
    return tarea, {"evm": float(evm_eq), "evm_escalar": float(evm_esc),
                   "ber": float(np.mean(bits != bits_rx[:len(bits)])),
                   "acpr_l": float(al_l), "acpr_u": float(al_u),
                   "acpr_alt_l": float(alt_l), "acpr_alt_u": float(alt_u),
                   "Pxx_dB": np.asarray(Pdb, dtype=np.float32), "f": np.asarray(f)}


def _escalar(tx_grid, g):
    """Receptor escalar: un único coeficiente complejo para toda la grilla, que
    deja como error la respuesta lineal del PA. Forzador de ceros, como el
    ecualizador por subportadora, para que con ruido las dos EVM sean
    comparables."""
    h = np.vdot(tx_grid, g) / np.vdot(tx_grid, tx_grid)
    return g / h


def curva_snr(tarea):
    """BER y EVM contra SNR para un caso: ('sin_pa',) o ('pa', bo, memoria).
    Con el receptor ecualizado y con el escalar. El escalar usa su propio
    generador de ruido, así que no cambia el ruido de las curvas ecualizadas."""
    bloques = []
    for i in range(N_BLOQUES):
        bits = _bits(1000 + i)
        with contextlib.redirect_stdout(io.StringIO()):
            x, tx_grid = generate_ofdm_signal(bits, PARAMS)
            y = x if tarea[0] == "sin_pa" else _pa(x, tarea[1], tarea[2])
        bloques.append((bits, tx_grid, _grilla(y)))
    out = {"snr": SNR_GRILLA}
    for sufijo, receptor, semilla in (("", equalize_per_subcarrier, 7), ("_escalar", _escalar, 8)):
        rng = np.random.default_rng(semilla)
        for k in ("ber", "evm", "bits", "errores"):
            out[k + sufijo] = []
        for snr in SNR_GRILLA:
            err = tot = 0
            evms = []
            for bits, tx_grid, g in bloques:
                sigma = np.sqrt(np.mean(np.abs(g) ** 2) / 10 ** (snr / 10) / 2)
                ruido = sigma * (rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape))
                eq = receptor(tx_grid, g + ruido)
                evms.append(evm_percent(tx_grid, eq))
                bits_rx, _ = qam16_gray_demod(eq.ravel())
                err += int(np.sum(bits != bits_rx[:len(bits)]))
                tot += len(bits)
                if err >= MIN_ERRORES or tot >= MAX_BITS:
                    break
            out["ber" + sufijo].append(err / tot)
            out["evm" + sufijo].append(float(np.mean(evms)))
            out["bits" + sufijo].append(tot)
            out["errores" + sufijo].append(err)
    return tarea, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    t0 = time.time()
    tareas_bo = [(bo, mem) for mem in (False, True) for bo in BO_GRILLA]
    tareas_snr = [("sin_pa",)] + [("pa", bo, True) for bo in BO_SNR] + [("pa", 0.0, False)]
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut_snr = [ex.submit(curva_snr, t) for t in tareas_snr]
        res_bo = dict(ex.map(punto_bo, tareas_bo))
        print(f"contra back-off listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res_snr = dict(f.result() for f in fut_snr)
    meta = {"BO_GRILLA": BO_GRILLA, "BO_SNR": BO_SNR, "SNR_GRILLA": SNR_GRILLA,
            "MIN_ERRORES": MIN_ERRORES, "MAX_BITS": MAX_BITS, "PARAMS": PARAMS,
            "fecha": time.strftime("%Y-%m-%d %H:%M"), "minutos": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as f:
        pickle.dump({"meta": meta, "bo": res_bo, "snr": res_snr}, f)
    print(f"Guardado en {SALIDA} ({meta['minutos']:.1f} min)")

    print("\n  BO    EVM sin mem   EVM con mem (escalar)   ACPR L/U con mem")
    for bo in BO_GRILLA:
        a, b = res_bo[(bo, False)], res_bo[(bo, True)]
        print(f"{bo:+5.1f}   {a['evm']:6.2f}     {b['evm']:6.2f} ({b['evm_escalar']:5.2f})"
              f"      {b['acpr_l']:6.2f} / {b['acpr_u']:6.2f}")
    for t, r in res_snr.items():
        print(f"\n{t}")
        for s, ber, ev, n, e in zip(r["snr"], r["ber"], r["evm"], r["bits"], r["errores"]):
            print(f"  SNR {s:4.0f} dB  BER {ber:.2e} ({e} errores en {n:.2e} bits)  EVM {ev:5.2f} %")


if __name__ == "__main__":
    main()
