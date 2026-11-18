"""Módulo de soporte del estudio de punto fijo del DPD (GD).

Provee el testbench (generación OFDM, cadena TX→DPD→PA→RX, métricas) y la
simulación con DPD entrenado por GD, en float o en punto fijo según fxp_cfg.
Lo usa sw/dpd_train/test/test_dpd_train.py.

Nota: solo se soporta el método GD. El LS-ILA quedó fuera del análisis de
punto fijo porque diverge a bajo back-off y no es el que va a la FPGA.
"""
import numpy as np
import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fixed_point.tx   import qam16_gray_mod, ofdm_modulate
from fixed_point.rx   import qam16_gray_demod, ofdm_demodulate, evm_rms_percent, compute_ber
from fixed_point.pa   import normalize_rms, apply_selected_pa
from fixed_point.acpr import compute_psd, compute_acpr
from fixed_point.dpd  import (
    train_dpd_gd, apply_dpd_and_pa,
    reset_ops, get_ops, total_ops,
)

# Generación de señal OFDM

def generate_ofdm_signal(bits_tx, params):
    qam_symbols = qam16_gray_mod(bits_tx)
    tx_signal, tx_grid, _, _ = ofdm_modulate(qam_symbols,Nfft=params["Nfft"],Nused=params["Nused"],cp_len=params["cp_len"],os_factor=params["os_factor"],)
    tx_signal, _ = normalize_rms(tx_signal, target_rms=1.0)
    return tx_signal, tx_grid

def print_pa_debug(tag, pa_info):
    if pa_info is None:
        return
    print(
        f"[{tag}] "
        f"Pin_in={pa_info.get('Pin_input_dB', np.nan): .2f} dB | "
        f"Pin_norm={pa_info.get('Pin_norm_dB', np.nan): .2f} dB | "
        f"Pin_PA={pa_info.get('Pin_PA_dB', np.nan): .2f} dB | "
        f"Pout_raw={pa_info.get('Pout_PA_raw_dB', np.nan): .2f} dB | "
        f"Pout_out={pa_info.get('Pout_output_dB', np.nan): .2f} dB | "
        f"G_raw={pa_info.get('G_PA_raw_dB', np.nan): .2f} dB"
    )

# Impresión de los coeficientes

def print_poly_coeff_debug(tag, coeffs, Pmax, M):
    coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    orders_odd = list(range(1, Pmax + 1, 2))
    orders_all = list(range(1, Pmax + 1))
    orders = orders_odd
    if coeffs.size == M * Pmax:
        orders = orders_all

    print(f"[{tag} poly coeffs]")
    col = 0
    for q in range(M):
        for p in orders:
            if col >= coeffs.size:
                break
            c = coeffs[col]
            print(
                f"    q={q}, p={p}: "
                f"abs={abs(c):.3e} | "
                f"phase={np.angle(c): .3f} rad"
            )
            col += 1

# Impresión de la señal predistorsionada

def print_dpd_signal_debug(tag, x_in, u_dpd):
    x_in = np.asarray(x_in, dtype=np.complex128).reshape(-1)
    u_dpd = np.asarray(u_dpd, dtype=np.complex128).reshape(-1)

    px = np.mean(np.abs(x_in) ** 2) + 1e-30
    pu = np.mean(np.abs(u_dpd) ** 2) + 1e-30
    g_rms_db = 10.0 * np.log10(pu / px)

    peak_x_db = 20.0 * np.log10(np.max(np.abs(x_in)) / np.sqrt(px) + 1e-30)
    peak_u_db = 20.0 * np.log10(np.max(np.abs(u_dpd)) / np.sqrt(pu) + 1e-30)

    print(
        f"[{tag} signal-check] "
        f"Ptx={10*np.log10(px): .2f} dB | "
        f"Pu={10*np.log10(pu): .2f} dB | "
        f"Pu/Ptx={g_rms_db: .2f} dB | "
        f"PAPR_x={peak_x_db: .2f} dB | "
        f"PAPR_u={peak_u_db: .2f} dB"
    )

# Evaluación común TX → PA → RX
def eval_chain(y_pa, tx_grid, bits_tx, params, bo_db, pa_info):
    rx_grid = ofdm_demodulate(y_pa,n_ofdm_symbols=params["n_ofdm_symbols"],Nfft=params["Nfft"],Nused=params["Nused"],cp_len=params["cp_len"],os_factor=params["os_factor"],)
    qam_tx_ref = tx_grid.reshape(-1)
    qam_rx     = rx_grid.reshape(-1)
    evm_percent, qam_rx_aligned = evm_rms_percent(qam_tx_ref, qam_rx)
    bits_rx, _ = qam16_gray_demod(qam_rx_aligned)
    ber = compute_ber(bits_tx, bits_rx)
    f, Pxx_dB, Pxx = compute_psd(y_pa, fs=params["fs"], nperseg=8192)
    acpr_l_db, acpr_u_db = compute_acpr(f, Pxx,main_bw=params["main_bw"],adj_offset=params["adj_offset"],adj_bw=params["adj_bw"],)

    return {
        "bo_db":          float(bo_db),     "qam_tx_ref":     qam_tx_ref,
        "qam_rx_aligned": qam_rx_aligned,   "evm_percent":    evm_percent,
        "ber":            ber,              "acpr_l_db":      acpr_l_db,
        "acpr_u_db":      acpr_u_db,        "f":              f,
        "Pxx_dB":         Pxx_dB,           "pa_info":        pa_info,
    }

# Simulación sin DPD (referencia: TX -> PA -> RX)
def simulate_no_dpd(bits_eval, params, bo_db):
    tx_signal, tx_grid = generate_ofdm_signal(bits_eval, params)
    y_pa, pa_info, _ = apply_selected_pa(tx_signal, pa_mode=params["pa_mode"], pa_nl=params["pa_nl"],
                                         use_memory=params["use_memory"], bo_db=bo_db, R0=params["R0"],
                                         fs_in=params["fs"], fs_pa=params["fs_pa"], sat_amp=params["sat_amp"],
                                         p_soft=params["p_soft"], normalize_input=True, normalize_output=True)
    result = eval_chain(y_pa, tx_grid, bits_eval, params, bo_db, pa_info)
    result["tx_signal"] = tx_signal
    result["u_dpd"] = tx_signal
    result["y_pa"] = y_pa
    result["fxp_label"] = "no_dpd"
    return result


# Simulación con DPD (GD), float o punto fijo
def simulate_with_dpd(bits_train, bits_eval, params, bo_db, *, method="gd", dpd_cfg, fxp_cfg=None):
    if method != "gd":
        raise ValueError(f"Solo se soporta GD en el flujo de punto fijo. Recibido: {method}")

    tx_train, _ = generate_ofdm_signal(bits_train, params)
    N_train_max_common = dpd_cfg.get("N_train_max", None)
    if N_train_max_common is not None:
        tx_train = tx_train[:int(N_train_max_common)]

    common = dict(pa_mode=params["pa_mode"],pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=bo_db,fs_in=params["fs"],
                  fs_pa=params["fs_pa"],R0=params["R0"],Pmax=dpd_cfg["Pmax"],M=dpd_cfg["M"],sat_amp=params["sat_amp"],p_soft=params["p_soft"],)

    reset_ops()
    train_out = train_dpd_gd(tx_train,**common,normalize_gain=False,
                             mu_h=dpd_cfg["mu_h"], mu_w=dpd_cfg["mu_w"],
                             use_nlms_h=dpd_cfg.get("use_nlms_h", False),
                             mu_h_respaldo=dpd_cfg.get("mu_h_respaldo", 0.05),
                             train_fxp=dpd_cfg.get("train_fxp", False),
                             ep_pa=dpd_cfg["ep_pa"], ep_dpd=dpd_cfg["ep_dpd"],
                             N_train_max=dpd_cfg.get("N_train_max", None),
                             w_amp_clip=dpd_cfg.get("w_amp_clip", None),
                             w_reg_lambda=dpd_cfg.get("w_reg_lambda", 0.0),
                             w_reg_pmin=dpd_cfg.get("w_reg_pmin", 1),
                             fxp_cfg=fxp_cfg,)

    ops_snapshot = train_out.get("ops_snapshot", get_ops())
    ops_total = train_out.get("ops_total", total_ops(ops_snapshot))
    train_out["ops_snapshot"] = ops_snapshot
    train_out["ops_total"] = ops_total
    train_out["N_train_used"] = len(tx_train)

    coeffs = train_out["dpd_coeffs"]
    print_poly_coeff_debug(f"{method} BO={bo_db}", coeffs, dpd_cfg["Pmax"], dpd_cfg["M"])
    h_coeffs = train_out.get("h_model", None)
    if h_coeffs is not None:
        print_poly_coeff_debug(f"gd BO={bo_db} PA-model h",h_coeffs,max(train_out.get("orders_h", [dpd_cfg["Pmax"]])),train_out.get("M", dpd_cfg["M"]),)

    w_identity = np.zeros_like(coeffs)
    if w_identity.size:
        w_identity[0] = 1.0 + 0.0j
    w_delta_norm = np.linalg.norm(coeffs - w_identity)
    mse_pa_last = train_out.get("mse_pa", [np.nan])[-1] if "mse_pa" in train_out else np.nan
    mse_dpd_last = train_out.get("mse_dpd", [np.nan])[-1] if "mse_dpd" in train_out else np.nan
    ops_total = train_out.get("ops_total", np.nan)
    print(
        f"[{method}] BO={bo_db} | "
        f"method={train_out.get('method')} | "
        f"||w||={np.linalg.norm(coeffs):.4e} | "
        f"||w-I||={w_delta_norm:.4e} | "
        f"MSE_PA={mse_pa_last:.4e} | "
        f"MSE_DPD={mse_dpd_last:.4e} | "
        f"OPS={ops_total:.3e} | "
        f"Ntrain={len(tx_train)}"
    )
    # --- Evaluación ---
    tx_eval, tx_grid_eval = generate_ofdm_signal(bits_eval, params)
    dpd_eval_mode = dpd_cfg.get("eval_mode", "free")
    if dpd_eval_mode not in ("free", "fixed_bo"):
        raise ValueError(f"eval_mode inválido: {dpd_eval_mode}. Usar 'free' o 'fixed_bo'.")

    normalize_dpd_output_eval = (dpd_eval_mode == "fixed_bo")

    u_dpd, y_pa, pa_info = apply_dpd_and_pa(tx_eval,coeffs,pa_mode=params["pa_mode"],pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=bo_db,fs_in=params["fs"],
                                            fs_pa=params["fs_pa"],R0=params["R0"],Pmax=train_out.get("Pmax", dpd_cfg["Pmax"]),M=train_out.get("M", dpd_cfg["M"]),
                                            sat_amp=params["sat_amp"],p_soft=params["p_soft"],normalize_dpd_input=True,normalize_dpd_output=normalize_dpd_output_eval,normalize_pa_input=False,fxp_cfg=fxp_cfg,)
    print_pa_debug(f"{method} BO={bo_db}", pa_info)
    print_dpd_signal_debug(f"{method} BO={bo_db}", tx_eval, u_dpd)

    direct_mse_real_pa = train_out.get("direct_mse_real_pa", np.nan)
    direct_nmse_real_pa = train_out.get("direct_nmse_real_pa",train_out.get("direct_nmse_gain_aligned", np.nan),)
    model_mse_last = train_out.get("mse_dpd", [np.nan])[-1] if "mse_dpd" in train_out else np.nan
    print(
        f"[{method} BO={bo_db} consistency-check] "
        f"MSE_model_last={model_mse_last:.4e} | "
        f"MSE_real_PA={direct_mse_real_pa:.4e} | "
        f"NMSE_real_PA={direct_nmse_real_pa:.4e}"
    )

    result = eval_chain(y_pa, tx_grid_eval, bits_eval, params, bo_db, pa_info)
    result["tx_signal"] = tx_eval
    result["u_dpd"]     = u_dpd
    result["train_out"] = train_out
    result["y_pa"] = y_pa
    result["fxp_label"] = (fxp_cfg.label() if (fxp_cfg is not None and fxp_cfg.enabled) else "float")
    return result
