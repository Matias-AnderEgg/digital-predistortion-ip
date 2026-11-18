"""Desempeño de la configuración elegida.

DPD por aprendizaje directo (GD) con M = 3 y P = 7, la configuración y el
entrenamiento de barrido_mp.py (PARAMS y DPD_CFG, semillas 123 para entrenar y
456 para evaluar), sobre la misma grilla de back-off que efecto_pa.py para
poder superponer las curvas sin DPD. Calcula y guarda:

1) Contra el back-off, sin ruido: EVM, BER, ACPR_L/U, PSD, constelación y las
   curvas de MSE por época del entrenamiento.

2) Contra la SNR, en los back-off de efecto_pa.py: BER y EVM con el DPD
   entrenado en ese back-off, con el receptor ecualizado y con el escalar,
   que es el de toda EVM de la tesis. Mismos bloques
   de datos (semillas 1000 + i), mismo ruido (semilla 7 para el ecualizado y 8
   para el escalar) y mismo criterio de parada que efecto_pa.py, así que las
   curvas con y sin DPD se comparan punto a punto.

Uso:
    python sys_with_acpr/desempeno.py [--jobs 8]
Escribe sys_with_acpr/resultados_desempeno/desempeno.pkl.
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

from sys_with_acpr.main import generate_ofdm_signal, simulate_with_dpd  # noqa: E402
from sys_with_acpr.dpd import apply_dpd_and_pa  # noqa: E402
from sys_with_acpr.rx import equalize_per_subcarrier, evm_percent, qam16_gray_demod  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, _bits, _resumen  # noqa: E402
from sys_with_acpr.efecto_pa import (BO_GRILLA, BO_SNR, SNR_GRILLA, MIN_ERRORES,  # noqa: E402
                                     MAX_BITS, N_BLOQUES, _bits as _bits_bloque, _grilla,
                                     _escalar)

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_desempeno",
                      "desempeno.pkl")

M, P = 3, 7
CFG = dict(DPD_CFG, M=M, Pmax=P)


def punto_bo(bo):
    """Entrena y evalúa el DPD en un back-off, sin ruido."""
    bits_train, bits_eval = _bits()
    t0 = time.time()
    with contextlib.redirect_stdout(io.StringIO()):
        r = simulate_with_dpd(bits_train, bits_eval, PARAMS, bo, method="gd", dpd_cfg=CFG)
    res = _resumen(r)
    res["target_gain"] = complex(r["train_out"]["target_gain"])
    res["seg"] = time.time() - t0
    return bo, res


def curva_snr(tarea):
    """BER y EVM contra SNR con el DPD w entrenado en el back-off bo."""
    bo, w = tarea
    p = PARAMS
    bloques = []
    for i in range(N_BLOQUES):
        bits = _bits_bloque(1000 + i)
        with contextlib.redirect_stdout(io.StringIO()):
            x, tx_grid = generate_ofdm_signal(bits, p)
            _, y, _ = apply_dpd_and_pa(x, w, pa_mode=p["pa_mode"], pa_nl=p["pa_nl"],
                                       use_memory=p["use_memory"], bo_db=bo, fs_in=p["fs"],
                                       fs_pa=p["fs_pa"], R0=p["R0"], Pmax=P, M=M,
                                       sat_amp=p["sat_amp"], p_soft=p["p_soft"],
                                       normalize_dpd_input=True, normalize_dpd_output=True,
                                       normalize_pa_input=False)
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
    return bo, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        res_bo = dict(ex.map(punto_bo, BO_GRILLA))
        print(f"contra back-off listo ({(time.time() - t0) / 60:.1f} min)", flush=True)
        res_snr = dict(ex.map(curva_snr, [(bo, res_bo[bo]["w"]) for bo in BO_SNR]))
    meta = {"M": M, "P": P, "CFG": CFG, "BO_GRILLA": BO_GRILLA, "BO_SNR": BO_SNR,
            "SNR_GRILLA": SNR_GRILLA, "MIN_ERRORES": MIN_ERRORES, "MAX_BITS": MAX_BITS,
            "PARAMS": PARAMS, "fecha": time.strftime("%Y-%m-%d %H:%M"),
            "minutos": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as f:
        pickle.dump({"meta": meta, "bo": res_bo, "snr": res_snr}, f)
    print(f"Guardado en {SALIDA} ({meta['minutos']:.1f} min)")

    print("\n  BO     EVM     ACPR L/U        respaldo")
    for bo in BO_GRILLA:
        r = res_bo[bo]
        print(f"{bo:+5.1f}  {r['evm']:6.2f}  {r['acpr_l']:6.2f} / {r['acpr_u']:6.2f}"
              f"   {r['pa_respaldo']}")
    for bo, r in res_snr.items():
        print(f"\nBO {bo:+.1f} dB con DPD")
        for s, ber, ev, n, e in zip(r["snr"], r["ber"], r["evm"], r["bits"], r["errores"]):
            print(f"  SNR {s:4.0f} dB  BER {ber:.2e} ({e} errores en {n:.2e} bits)  EVM {ev:5.2f} %")


if __name__ == "__main__":
    main()
