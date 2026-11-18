import numpy as np
import sys
import os
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr.tx   import qam16_gray_mod, ofdm_modulate
from sys_with_acpr      import lte
from sys_with_acpr.rx   import qam16_gray_demod, ofdm_demodulate, evm_rms_percent, compute_ber, equalize_per_subcarrier, evm_percent as evm_sin_realinear
from sys_with_acpr.pa   import apply_selected_pa, compute_pa_power_curve, get_operating_point, normalize_rms, power_db
from sys_with_acpr.acpr import compute_psd, compute_acpr
from sys_with_acpr.dpd  import (
    train_dpd_ls, train_dpd_gd, apply_dpd_and_pa,
    reset_ops, get_ops, total_ops,
)
from sys_with_acpr.plot import (
    compute_papr_db,
    plot_acpr_vs_bo, plot_evm_ber_vs_bo,
    plot_am_am_curve, plot_am_am_all_pas,
    plot_psd_per_bo, plot_constellations,
    plot_pa_time_clipping,
    plot_ofdm_am_am_cloud,
)

# Generación de señal OFDM

# params["senal"]: "lte" es la bajada LTE de 10 MHz de lte.py, con el filtro de
# transmisión, la de todas las simulaciones de la tesis. "ofdm" es una OFDM
# propia (Nfft, Nused y cp_len), que queda para comparar.
def generate_ofdm_signal(bits_tx, params):
    qam_symbols = qam16_gray_mod(bits_tx)
    if params.get("senal", "ofdm") == "lte":
        tx_signal, tx_grid = lte.modular(qam_symbols, params["n_ofdm_symbols"], params["os_factor"])
        # La referencia del DPD y de las métricas es la señal ya filtrada.
        tx_signal = lte.filtrar(tx_signal, lte.filtro_tx(params["os_factor"]))
        return tx_signal, tx_grid
    tx_signal, tx_grid, _, _ = ofdm_modulate(qam_symbols,Nfft=params["Nfft"],Nused=params["Nused"],cp_len=params["cp_len"],os_factor=params["os_factor"],)
    tx_signal, _ = normalize_rms(tx_signal, target_rms=1.0)
    return tx_signal, tx_grid

def demodulate(y, params):
    if params.get("senal", "ofdm") == "lte":
        # La ventana de la FFT arranca medio filtro antes, dentro del prefijo (lte.py).
        return lte.demodular(y, params["n_ofdm_symbols"], params["os_factor"],
                             adelanto=lte.adelanto_filtro(params["os_factor"]))
    return ofdm_demodulate(y,n_ofdm_symbols=params["n_ofdm_symbols"],Nfft=params["Nfft"],Nused=params["Nused"],cp_len=params["cp_len"],os_factor=params["os_factor"],)

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
    rx_grid = demodulate(y_pa, params)
    # Receptor: "subportadora" ecualiza un coeficiente por subportadora, como
    # cualquier receptor OFDM. "escalar" usa un único escalar de ganancia y fase, que deja la
    # respuesta lineal de los filtros del PA como error.
    qam_tx_ref = tx_grid.reshape(-1)
    if params.get("rx_eq", "subportadora") == "subportadora":
        # Forzador de ceros por subportadora y EVM sin realinear (rx.py).
        qam_rx_aligned = equalize_per_subcarrier(tx_grid, rx_grid).reshape(-1)
        evm_percent = evm_sin_realinear(qam_tx_ref, qam_rx_aligned)
    else:
        evm_percent, qam_rx_aligned = evm_rms_percent(qam_tx_ref, rx_grid.reshape(-1))
    # La EVM con el escalar se guarda siempre, porque es la que ve la memoria
    # lineal del PA y lo que el DPD corrige de ella.
    evm_escalar, qam_rx_escalar = evm_rms_percent(qam_tx_ref, rx_grid.reshape(-1))
    bits_rx, _ = qam16_gray_demod(qam_rx_aligned)
    ber = compute_ber(bits_tx, bits_rx)
    f, Pxx_dB, Pxx = compute_psd(y_pa, fs=params["fs"], nperseg=8192)
    acpr_l_db, acpr_u_db = compute_acpr(f, Pxx,main_bw=params["main_bw"],adj_offset=params["adj_offset"],adj_bw=params["adj_bw"],)
    # Canal alterno, a dos veces la separación del adyacente.
    acpr_alt_l_db, acpr_alt_u_db = compute_acpr(f, Pxx,main_bw=params["main_bw"],adj_offset=2*params["adj_offset"],adj_bw=params["adj_bw"],)

    return {
        "bo_db":          float(bo_db),     "qam_tx_ref":     qam_tx_ref,
        "qam_rx_aligned": qam_rx_aligned,   "evm_percent":    evm_percent,
        "evm_escalar_percent": evm_escalar, "qam_rx_escalar": qam_rx_escalar,
        "ber":            ber,              "acpr_l_db":      acpr_l_db,
        "acpr_u_db":      acpr_u_db,        "f":              f,
        "acpr_alt_l_db":  acpr_alt_l_db,    "acpr_alt_u_db":  acpr_alt_u_db,
        "Pxx_dB":         Pxx_dB,           "pa_info":        pa_info,
    }

# Simulación ideal
def simulate_ideal(bits_eval, params):
    tx_signal, tx_grid = generate_ofdm_signal(bits_eval, params)
    print(f"[ideal] Ptx={power_db(tx_signal): .2f} dB")
    # IMPORTANTE: NO RESAMPLEAR
    y = tx_signal.copy()
    result = eval_chain(y, tx_grid, bits_eval, params, bo_db=0.0, pa_info=None)
    result["tx_signal"] = tx_signal
    result["u_dpd"] = tx_signal
    result["y_pa"] = y
    return result

# Simulación sin DPD
def simulate_no_dpd(bits_eval, params, bo_db):
    tx_signal, tx_grid = generate_ofdm_signal(bits_eval, params)

    y_pa, pa_info, _ = apply_selected_pa(tx_signal,pa_mode=params["pa_mode"],pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=bo_db,R0=params["R0"],
                                         fs_in=params["fs"],fs_pa=params["fs_pa"],sat_amp=params["sat_amp"],p_soft=params["p_soft"],normalize_input=True,normalize_output=True,)

    print_pa_debug(f"no_dpd BO={bo_db}", pa_info)

    if params.get("plot_time_debug", False):
        scale_bo = params["sat_amp"] * 10.0 ** (-float(bo_db) / 20.0)
        x_debug = tx_signal

        y_pa_raw_debug, _, _ = apply_selected_pa(tx_signal,pa_mode=params["pa_mode"],pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=bo_db,R0=params["R0"],
                                                 fs_in=params["fs"],fs_pa=params["fs_pa"],sat_amp=params["sat_amp"],p_soft=params["p_soft"],normalize_input=True,normalize_output=False,)

        y_debug = y_pa_raw_debug / (scale_bo + 1e-30)
        clip_level_debug = params["sat_amp"] / (scale_bo + 1e-30)

        debug_title = "Diagnóstico temporal PA en dominio pre-BO"
        if params.get("use_memory", False):
            debug_title += " (PA con memoria: salida final del W-H)"

        plot_pa_time_clipping(x_pa_in=x_debug,y_pa_out=y_debug,bo_db=bo_db,n0=0,N=800,title=debug_title,clip_level=clip_level_debug if (params["pa_nl"] == "clip" and not params.get("use_memory", False)) else None,align_output_gain=params.get("use_memory", False),)
 
    result = eval_chain(y_pa, tx_grid, bits_eval, params, bo_db, pa_info)
    result["tx_signal"] = tx_signal
    result["u_dpd"] = tx_signal
    result["y_pa"] = y_pa
    return result

# Simulación con DPD (LS o GD)
def simulate_with_dpd(bits_train, bits_eval, params, bo_db, *, method, dpd_cfg):
    tx_train, _ = generate_ofdm_signal(bits_train, params)
    N_train_max_common = dpd_cfg.get("N_train_max", None)
    if N_train_max_common is not None:
        tx_train = tx_train[:int(N_train_max_common)]

    common = dict(pa_mode=params["pa_mode"],pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=bo_db,fs_in=params["fs"],
                  fs_pa=params["fs_pa"],R0=params["R0"],Pmax=dpd_cfg["Pmax"],M=dpd_cfg["M"],sat_amp=params["sat_amp"],p_soft=params["p_soft"],)

    reset_ops()
    if method == "ls":
        train_out = train_dpd_ls(tx_train,**common,rcond=dpd_cfg.get("rcond", 1e-10),normalize_gain=True,)
    elif method == "gd":
        train_out = train_dpd_gd(tx_train,**common,normalize_gain=False,
                                 mu_h=dpd_cfg["mu_h"], mu_w=dpd_cfg["mu_w"],
                                 use_nlms_h=dpd_cfg.get("use_nlms_h", False),
                                 mu_h_respaldo=dpd_cfg.get("mu_h_respaldo", 0.05),
                                 h_ganancia_max=dpd_cfg.get("h_ganancia_max", None),
                                 ep_pa=dpd_cfg["ep_pa"], ep_dpd=dpd_cfg["ep_dpd"],
                                 N_train_max=dpd_cfg.get("N_train_max", None),
                                 w_amp_clip=dpd_cfg.get("w_amp_clip", None),
                                 w_reg_lambda=dpd_cfg.get("w_reg_lambda", 0.0),
                                 w_reg_pmin=dpd_cfg.get("w_reg_pmin", 1),)
    else:
        raise ValueError(f"Método DPD desconocido: {method}")

    ops_snapshot = train_out.get("ops_snapshot", get_ops())
    ops_total = train_out.get("ops_total", total_ops(ops_snapshot))
    train_out["ops_snapshot"] = ops_snapshot
    train_out["ops_total"] = ops_total
    train_out["N_train_used"] = len(tx_train)

    coeffs = train_out["dpd_coeffs"]
    print_poly_coeff_debug(f"{method} BO={bo_db}", coeffs, dpd_cfg["Pmax"], dpd_cfg["M"])
    if method == "gd":
        h_coeffs = train_out.get("h_model", None)
        if h_coeffs is None:
            h_coeffs = train_out.get("pa_coeffs", None)
        if h_coeffs is None:
            h_coeffs = train_out.get("h", None)
        if h_coeffs is None:
            h_coeffs = train_out.get("pa_model_coeffs", None)

        if h_coeffs is not None:
            print_poly_coeff_debug(f"gd BO={bo_db} PA-model h",h_coeffs,max(train_out.get("orders_h", [dpd_cfg["Pmax"]])),train_out.get("M", dpd_cfg["M"]),)
        else:
            print(f"[gd BO={bo_db} PA-model h] No se encontraron coeficientes h en train_out. Keys={list(train_out.keys())}")
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
                                            sat_amp=params["sat_amp"],p_soft=params["p_soft"],normalize_dpd_input=True,normalize_dpd_output=normalize_dpd_output_eval,normalize_pa_input=False,)
    print_pa_debug(f"{method} BO={bo_db}", pa_info)
    print_dpd_signal_debug(f"{method} BO={bo_db}", tx_eval, u_dpd)
    if method in ("gd", "ls"):
        px = np.mean(np.abs(tx_eval) ** 2) + 1e-30
        pu = np.mean(np.abs(u_dpd) ** 2) + 1e-30
        print(
            f"[{method} BO={bo_db} OP-check] "
            f"eval_mode={dpd_eval_mode} | "
            f"Pu/Ptx={10*np.log10(pu/px): .2f} dB | "
            f"normalize_dpd_output={normalize_dpd_output_eval} | "
            f"dpd_scale_u_norm={pa_info.get('dpd_scale_u_norm', np.nan):.4e} | "
            f"PA normalize_input=False"
        )
    if method == "gd":
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
    return result


# Tabla de resultados, sólo con los casos activos
def print_results_table(results_dict):
    labels = {"ideal": "Ideal", "no_dpd": "Sin DPD", "ls": "DPD-LS", "gd": "DPD-GD"}
    if results_dict.get("ideal"):
        r = results_dict["ideal"][0]
        print("=" * 70)
        print("Referencia ideal (sin PA)")
        print("=" * 70)
        print(
            f"EVM = {r['evm_percent']:.3f} %   "
            f"BER = {r['ber']:.2e}   "
            f"ACPRL = {r['acpr_l_db']:.2f} dB   "
            f"ACPRU = {r['acpr_u_db']:.2f} dB"
        )
        print()

    active_keys = [k for k in ("no_dpd", "ls", "gd") if results_dict.get(k)]
    if not active_keys:
        return

    header = f"{'BO':>5}"
    for k in active_keys:
        header += f"  {'EVM '+labels[k]:>14}  {'BER '+labels[k]:>14}  {'ACPRL '+labels[k]:>15}"

    sep = "=" * len(header)
    print(sep)
    print("16QAM-OFDM — Comparativa de casos activos")
    print(sep)
    print(header)
    print(sep)

    n_rows = max(len(results_dict[k]) for k in active_keys)
    ref_key = active_keys[0]

    for i in range(n_rows):
        bo_ref = results_dict[ref_key][i]["bo_db"]
        line = f"{bo_ref:5.1f}"

        for k in active_keys:
            r = results_dict[k][i]
            line += f"  {r['evm_percent']:14.3f}  {r['ber']:14.2e}  {r['acpr_l_db']:15.2f}"

        print(line)


# Tabla de diagnóstico de entrada al PA (PAPR)
def print_input_diagnostics_table(results_dict):
    active_keys = [k for k in ("no_dpd", "ls", "gd") if results_dict.get(k)]
    if not active_keys:
        return

    header = (
        f"{'BO':>5}  "
        f"{'PAPR tx':>10}  {'PAPR no_dpd':>12}  {'PAPR ls':>10} {'PAPR gd':>10}"
    )
    sep = "=" * len(header)
    print()
    print(sep)
    print("Diagnóstico de entrada al PA")
    print(sep)
    print(header)
    print(sep)

    ref_key = active_keys[0]
    bo_values = [r["bo_db"] for r in results_dict[ref_key]]

    for bo_db in bo_values:
        r_ref = next((rr for rr in results_dict[ref_key] if rr["bo_db"] == bo_db), None)
        r_no  = next((rr for rr in results_dict.get("no_dpd", []) if rr["bo_db"] == bo_db), None)
        r_ls  = next((rr for rr in results_dict.get("ls", []) if rr["bo_db"] == bo_db), None)
        r_gd  = next((rr for rr in results_dict.get("gd", []) if rr["bo_db"] == bo_db), None)

        papr_tx  = compute_papr_db(r_ref["tx_signal"]) if r_ref is not None else np.nan
        papr_no  = compute_papr_db(r_no["u_dpd"])  if r_no  is not None else np.nan
        papr_ls  = compute_papr_db(r_ls["u_dpd"])  if r_ls  is not None else np.nan
        papr_gd  = compute_papr_db(r_gd["u_dpd"])  if r_gd  is not None else np.nan

        print(
            f"{bo_db:5.1f}  "
            f"{papr_tx:10.2f}  {papr_no:12.2f}  {papr_ls:10.2f} {papr_gd:10.2f}"
        )

# Tabla de complejidad del DPD
def print_dpd_complexity_table(results_dict):
    active_keys = [k for k in ("ls", "gd") if results_dict.get(k)]
    if not active_keys:
        return

    print()
    print("=" * 105)
    print("Complejidad experimental de entrenamiento DPD")
    print("=" * 105)
    print(
        f"{'BO':>5}  {'Método':>8}  {'Ntrain':>8}  {'Ops totales':>14}  "
        f"{'Mul C':>12}  {'Add C':>12}  {'Abs':>10}  {'Pow':>10}  {'Div':>10}  {'Rel. vs LS':>10}"
    )
    print("=" * 105)

    bo_values = sorted({r["bo_db"] for k in active_keys for r in results_dict[k]})

    for bo_db in bo_values:
        ls_total = np.nan
        r_ls = next((rr for rr in results_dict.get("ls", []) if rr["bo_db"] == bo_db), None)
        if r_ls is not None:
            ls_total = r_ls.get("train_out", {}).get("ops_total", np.nan)

        for k in active_keys:
            r = next((rr for rr in results_dict.get(k, []) if rr["bo_db"] == bo_db), None)
            if r is None:
                continue

            train_out = r.get("train_out", {})
            ops = train_out.get("ops_snapshot", {})
            ops_t = train_out.get("ops_total", np.nan)
            ntrain = train_out.get("N_train_used", np.nan)
            rel = ops_t / ls_total if np.isfinite(ls_total) and ls_total > 0 else np.nan

            print(
                f"{bo_db:5.1f}  {k:>8}  {ntrain:8.0f}  {ops_t:14.3e}  "
                f"{ops.get('complex_mul', 0):12.3e}  "
                f"{ops.get('complex_add', 0):12.3e}  "
                f"{ops.get('abs', 0):10.3e}  "
                f"{ops.get('pow', 0):10.3e}  "
                f"{ops.get('div', 0):10.3e}  "
                f"{rel:10.2f}"
            )

    print("=" * 105)
    print("Nota: conteo aproximado de operaciones dominantes, útil para comparación relativa LS vs GD.")

# Casos que se corren
RUN_CASES = {
    "ideal":  False,
    "no_dpd": True,
    "ls":     False,
    "gd":     False,
}


def main():
    rng_train = np.random.default_rng(123)
    rng_eval  = np.random.default_rng(456)

    # Parámetros OFDM / sistema
    # Bajada LTE de 10 MHz (lte.py): 600 portadoras, FFT de 1024, prefijo
    # normal. 364 símbolos son 26 subtramas, 1,6 millones de muestras.
    Nfft        = lte.NFFT
    Nused       = lte.NUSED
    os_factor   = 4
    n_ofdm_syms = 364
    delta_f     = lte.DF

    fs    = Nfft * delta_f * os_factor   # 61,44 MHz
    fs_pa = fs

    main_bw    = Nused * delta_f       # 9 MHz ocupados
    adj_offset = 10e6                  # canal adyacente a 10 MHz, el alterno a 20 MHz
    adj_bw     = main_bw

    # PA configurable:
    # pa_nl: "clip", "soft" o "arctan"
    # use_memory: False -> no linealidad memoryless
    #             True  -> LTI1 + no linealidad + LTI2 del Wiener-Hammerstein
    pa_nl = "arctan"
    use_memory = True
    pa_mode = f"{pa_nl}_{'mem' if use_memory else 'nomem'}"

    sat_amp = 1.0
    p_soft = 3.0
    plot_time_debug = False #solo activar si memoria esta False

    dpd_cfg = {
        "Pmax": 7,
        "M": 3,
        "rcond": 1e-4,
        "eval_mode": "fixed_bo",

        # DLA-GD (precondicionador diagonal + normalizador NLMS-gradiente; h al mismo orden que el DPD)
        # Modelo del PA: el mismo esquema que el ARM, mu_h = 0.001
        # sin NLMS, con respaldo NLMS mu_h = 0.05 si diverge.
        "mu_h":          0.001,
        "use_nlms_h":    False,
        "mu_h_respaldo": 0.05,
        # Paso efectivo del modelo del PA limitado a 1 y 16000 muestras.
        "h_ganancia_max": 1.0,
        "mu_w":         0.5,   # estable en [0.2,1.0]; 0.5 da el mejor NMSE a ep=30
        "ep_pa":        40,
        "ep_dpd":       5,
        "N_train_max":  16000,
        "w_reg_lambda": 1e-4,  # L2 leakage p>=pmin: evita deriva de órdenes altos
        "w_reg_pmin":   3,     # p=1 libre (freeze_w0), p>=3 regularizados
    }

    params = {
            "senal": "lte","Nfft": Nfft,"Nused": Nused,"os_factor": os_factor,"n_ofdm_symbols": n_ofdm_syms,"fs": fs,
            "fs_pa": fs_pa,"main_bw": main_bw,"adj_offset": adj_offset,"adj_bw": adj_bw,"R0": 50.0,"pa_mode": pa_mode,
            "pa_nl": pa_nl,"use_memory": use_memory,"sat_amp": sat_amp,"p_soft": p_soft,"plot_time_debug": plot_time_debug,
            "rx_eq": "subportadora",
            }

    bits_per_sym = 4
    n_bits     = n_ofdm_syms * Nused * bits_per_sym
    bits_train = rng_train.integers(0, 2, n_bits)
    bits_eval  = rng_eval.integers(0, 2, n_bits)

    bo_list = [-9, -6, -3, 0, 3, 6, 9]
    bo_list = [0]
    # Simulación de los casos habilitados
    results_dict = {"ideal": [], "no_dpd": [], "ls": [], "gd": []}

    if RUN_CASES["ideal"]:
        results_dict["ideal"].append(simulate_ideal(bits_eval, params))

    for bo_db in bo_list:
        print(f"Simulando BO = {bo_db} dB ...")
        if RUN_CASES["no_dpd"]:
            results_dict["no_dpd"].append(simulate_no_dpd(bits_eval, params, bo_db))
        if RUN_CASES["ls"]:
            results_dict["ls"].append(simulate_with_dpd(bits_train, bits_eval, params, bo_db, method="ls", dpd_cfg=dpd_cfg))
        if RUN_CASES.get("gd", False):
            results_dict["gd"].append(simulate_with_dpd(bits_train, bits_eval, params, bo_db, method="gd", dpd_cfg=dpd_cfg))
    print()
    print_results_table(results_dict)
    print_input_diagnostics_table(results_dict)
    print_dpd_complexity_table(results_dict)
    print(f"\nModo de evaluación DPD: {dpd_cfg.get('eval_mode', 'free')}  ('fixed_bo' compara a igual BO efectivo)\n")
    print("Simulación terminada. Preparando curvas y gráficos...")
    # Curva AM-AM y puntos de operación
    COMPUTE_STATIC_CURVES = False
    curve = None
    curves_all = {}
    op_points = {}

    if COMPUTE_STATIC_CURVES:
        curve = compute_pa_power_curve(pa_mode=pa_mode,pa_nl=params["pa_nl"],use_memory=params["use_memory"],fs_in=fs,fs_pa=fs_pa,R0=params["R0"],
                                       pin_grid_dbm=np.linspace(-30, 30, 61),sat_amp=params["sat_amp"],p_soft=params["p_soft"],)
        curves_all = {
            "clip": compute_pa_power_curve(pa_mode="clip_nomem",pa_nl="clip",use_memory=False,fs_in=fs,fs_pa=fs_pa,R0=params["R0"],
                                           pin_grid_dbm=np.linspace(-30, 100, 61),sat_amp=params["sat_amp"],p_soft=params["p_soft"],),
            "soft": compute_pa_power_curve(pa_mode="soft_nomem",pa_nl="soft",use_memory=False,fs_in=fs,fs_pa=fs_pa,R0=params["R0"],
                                           pin_grid_dbm=np.linspace(-30, 100, 61),sat_amp=params["sat_amp"],p_soft=params["p_soft"],),
            "arctan": compute_pa_power_curve(pa_mode="arctan_nomem",pa_nl="arctan",use_memory=False,fs_in=fs,fs_pa=fs_pa,R0=params["R0"],
                                             pin_grid_dbm=np.linspace(-30, 100, 61),sat_amp=params["sat_amp"],p_soft=params["p_soft"],),
        }

        # Todos los puntos de operación para todos los BO y todos los casos
        for key, results in results_dict.items():
            ops = []
            for r in results:
                op = get_operating_point(r["u_dpd"],pa_mode=pa_mode,pa_nl=params["pa_nl"],use_memory=params["use_memory"],bo_db=r["bo_db"],
                                         R0=params["R0"],fs_in=fs,fs_pa=fs_pa,sat_amp=params["sat_amp"],p_soft=params["p_soft"],)
                op["bo_db"] = r["bo_db"]
                ops.append(op)
            op_points[key] = ops

    print("Generando plots...")

    # Gráficos globales livianos.
    plot_acpr_vs_bo(results_dict, pa_mode=pa_mode)
    plot_evm_ber_vs_bo(results_dict, pa_mode=pa_mode)

    # PSD por BO: útil para ver ACPR y salida espectral.
    plot_psd_per_bo(results_dict, bo_list=bo_list, params=params, pa_mode=pa_mode)

    # Diagnósticos pesados: activar solo cuando haga falta.
    PLOT_HEAVY_DIAGNOSTICS = True
    if PLOT_HEAVY_DIAGNOSTICS:
        plot_ofdm_am_am_cloud(results_dict, bo_list=bo_list, params=params, pa_mode=pa_mode)
        plot_constellations(results_dict, bo_list=bo_list, pa_mode=pa_mode)

        # Las curvas AM-AM estáticas sólo existen si COMPUTE_STATIC_CURVES=True.
        if COMPUTE_STATIC_CURVES and curve is not None:
            plot_am_am_curve(curve, op_points, pa_mode)
        if COMPUTE_STATIC_CURVES and curves_all:
            plot_am_am_all_pas(curves_all)

    print("Mostrando figuras...")
    plt.show()

if __name__ == "__main__":
    main()
