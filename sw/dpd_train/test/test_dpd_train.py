"""Verificación del entrenamiento en C contra el modelo de Python.

Tres pruebas, a BO = -9 dB. Las dos primeras con la señal de desarrollo
(semilla 123), la tercera con la LTE con que se generan los vectores dorados
(semilla 123):

1. Modelo del PA: C contra `_train_pa_model_gd` en flotante, con el mismo
   esquema (mu_h = 0,001 sin NLMS, con el paso
   efectivo limitado a 1) y sin respaldo.
2. Predistorsionador: C contra `_train_dpd_gd_direct_dla` en flotante, los
   dos con el MISMO h, para que un error del primer lazo no se confunda con
   uno del segundo.
3. Cadena completa en C con la configuración por defecto, pasada a Q0.15.
   Los vectores dorados se entrenan igual, en flotante, así que los 12
   códigos tienen que salir iguales, y el NMSE directo contra el PA real
   también.

Las dos primeras no pueden ser bit a bit: numpy suma en otro orden que un
lazo de C, y la diferencia del último bit se propaga por 640 mil
iteraciones de un lazo adaptativo. Lo que se pide es que la diferencia
quede muchos órdenes por debajo de un LSB de Q0.15 (3e-5).

Uso: python test/test_dpd_train.py [--rapido]
Con --rapido entrena 2000 muestras y 5 épocas, para iterar sobre el código.
"""
from __future__ import annotations

import ctypes
import csv
import dataclasses
import os
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", "..", ".."))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "fixed_point", "golden_vectors"))

import generate as GV  # noqa: E402
from fixed_point import dpd  # noqa: E402
from fixed_point import fxp_format as fxp  # noqa: E402
from fixed_point import main as M  # noqa: E402
from fixed_point.pa import apply_selected_pa, normalize_rms  # noqa: E402

NCOEF = 12
LSB_Q15 = 2.0 ** -15
TOLERANCIA = 1e-3 * LSB_Q15     # mil veces por debajo de un LSB


class Cfg(ctypes.Structure):
    _fields_ = [
        ("G", ctypes.c_double),
        ("mu_h", ctypes.c_double),
        ("ep_pa", ctypes.c_int),
        ("use_nlms_h", ctypes.c_int),
        ("divergence_factor", ctypes.c_double),
        ("quantize_h", ctypes.c_int),
        ("mu_w", ctypes.c_double),
        ("ep_dpd", ctypes.c_int),
        ("use_nlms_w", ctypes.c_int),
        ("freeze_w0", ctypes.c_int),
        ("grad_clip", ctypes.c_double),
        ("w_reg_lambda", ctypes.c_double),
        ("w_reg_pmin", ctypes.c_int),
        ("precond_delta", ctypes.c_double),
        ("mu_h_respaldo", ctypes.c_double),
        ("warm_start", ctypes.c_int),
        ("h_ganancia_max", ctypes.c_double),
        ("h_umbral_pico", ctypes.c_double),
    ]


def cargar_lib():
    nombre = "libdpd_train.dylib" if sys.platform == "darwin" else "libdpd_train.so"
    lib = ctypes.CDLL(os.path.join(AQUI, "..", "build", nombre))
    pc = np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS")
    pd = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    pi16 = np.ctypeslib.ndpointer(np.int16, flags="C_CONTIGUOUS")
    lib.dpd_train_cfg_default.argtypes = [ctypes.POINTER(Cfg)]
    lib.dpd_train_pa_model.argtypes = [pc, pc, ctypes.c_int, ctypes.POINTER(Cfg), pc, pd]
    lib.dpd_train_pa_model.restype = ctypes.c_int
    lib.dpd_train_dla.argtypes = [pc, ctypes.c_int, pc, ctypes.POINTER(Cfg), pc, pd]
    lib.dpd_train_dla.restype = ctypes.c_int
    lib.dpd_coef_to_q15.argtypes = [pc, pi16, pi16]
    lib.dpd_coef_to_q15.restype = ctypes.c_int
    lib.dpd_rms_salida.argtypes = [pc, ctypes.c_int, pc, ctypes.POINTER(Cfg)]
    lib.dpd_rms_salida.restype = ctypes.c_double
    return lib


def senal_lte(n, semilla=123):
    """La de generate.py: la LTE de la tesis, semilla 123, renormalizada a RMS
    1 como en train_dpd_gd, por el PA real."""
    from fixed_point import lte
    x, _ = normalize_rms(GV.senal_lte(semilla)[0][:n], target_rms=1.0)
    params = lte.params()
    y, _, _ = apply_selected_pa(x, **pa_kwargs(params), normalize_input=True, normalize_output=False)
    return params, np.ascontiguousarray(x), np.ascontiguousarray(y)


def senal(n, semilla=123):
    """La de desarrollo: semilla 123, RMS 1, por el PA real."""
    params = GV.build_params()
    bits = np.random.default_rng(semilla).integers(0, 2, params["n_ofdm_symbols"] * 52 * 4)
    tx, _ = M.generate_ofdm_signal(bits, params)
    x, _ = normalize_rms(tx[:n], target_rms=1.0)
    y, _, _ = apply_selected_pa(x, **pa_kwargs(params), normalize_input=True, normalize_output=False)
    return params, np.ascontiguousarray(x), np.ascontiguousarray(y)


def pa_kwargs(params):
    return dict(pa_mode=params["pa_mode"], pa_nl=params["pa_nl"], use_memory=params["use_memory"],
                bo_db=GV.BO_DB, R0=params["R0"], fs_in=params["fs"], fs_pa=params["fs_pa"],
                sat_amp=params["sat_amp"], p_soft=params["p_soft"])


def nmse_directo(params, x, w):
    """Igual que train_dpd_gd: datapath en punto fijo y PA real."""
    u = dpd.apply_memory_poly_fxp(x, w, Pmax=GV.PMAX, M=GV.M_TAPS, cfg=GV.CFG)
    y, _, _ = apply_selected_pa(u, **pa_kwargs(params), normalize_input=False, normalize_output=True)
    a = np.vdot(y, x) / np.vdot(y, y)
    return 10 * np.log10(np.mean(np.abs(x - a * y) ** 2) / np.mean(np.abs(x) ** 2))


def coefs_dorados():
    """Los coeficientes de los vectores dorados."""
    texto = open(os.path.join(RAIZ, "fixed_point/golden_vectors/out/coeffs_w.csv")).read()
    filas = list(csv.DictReader(texto.splitlines()))
    return np.array([int(r["code_re"]) + 1j * int(r["code_im"]) for r in filas]) / 32768.0


def comparar(nombre, c, py, hist_c, hist_py):
    err = float(np.max(np.abs(c - py)))
    err_hist = float(np.max(np.abs(hist_c - hist_py) / np.abs(hist_py)))
    ok = err < TOLERANCIA and len(hist_c) == len(hist_py)
    print(f"  {nombre}: error max {err:.2e} ({err / LSB_Q15:.1e} LSB), "
          f"MSE por epoca {err_hist:.1e} relativo, epocas {len(hist_c)}/{len(hist_py)}  "
          f"{'OK' if ok else 'FALLA'}")
    return ok


def main():
    rapido = "--rapido" in sys.argv
    n, ep = (2000, 5) if rapido else (16000, 40)
    lib = cargar_lib()
    cfg = Cfg()
    lib.dpd_train_cfg_default(ctypes.byref(cfg))
    cfg.ep_pa = cfg.ep_dpd = ep
    # Las pruebas 1 y 2 comparan un solo lazo, sin el respaldo.
    cfg.mu_h_respaldo = 0.0

    # El mismo modelo, con la cuantización apagada: es lo que corre el ARM.
    flot = dataclasses.replace(GV.CFG, enabled=False)
    params, x, y = senal(n)
    print(f"{n} muestras, {ep} epocas, BO {GV.BO_DB:g} dB")
    oks = []

    print("1. Modelo del PA")
    t = time.time()
    h_py, _, hist_h_py = dpd._train_pa_model_gd(
        x, y, Pmax=GV.PMAX, M=GV.M_TAPS, mu=cfg.mu_h, n_epochs=ep, N_train_max=n,
        use_nlms=bool(cfg.use_nlms_h), precond_delta=cfg.precond_delta,
        divergence_factor=cfg.divergence_factor, fxp_cfg=flot,
        h_ganancia_max=cfg.h_ganancia_max)
    t_py = time.time() - t
    h_c = np.zeros(NCOEF, np.complex128)
    hist_h_c = np.zeros(ep)
    t = time.time()
    ep_c = lib.dpd_train_pa_model(x, y, n, ctypes.byref(cfg), h_c, hist_h_c)
    t_c = time.time() - t
    print(f"  Python {t_py:.1f} s, C {t_c * 1e3:.0f} ms")
    oks.append(comparar("h", h_c, h_py, hist_h_c[:ep_c], hist_h_py))

    print("2. Predistorsionador, con el h de Python en los dos")
    h_py = np.ascontiguousarray(h_py, dtype=np.complex128)
    t = time.time()
    w_py, _, hist_w_py, _ = dpd._train_dpd_gd_direct_dla(
        x, h_py, Pmax=GV.PMAX, M=GV.M_TAPS, mu=cfg.mu_w, n_epochs=ep, N_train_max=n,
        target_gain=None, grad_clip=cfg.grad_clip, freeze_w0=True, use_nlms=True,
        w_reg_lambda=cfg.w_reg_lambda, w_reg_pmin=cfg.w_reg_pmin,
        precond_delta=cfg.precond_delta, fxp_cfg=flot)
    t_py = time.time() - t
    w_c = np.zeros(NCOEF, np.complex128)
    hist_w_c = np.zeros(ep)
    t = time.time()
    ep_c = lib.dpd_train_dla(x, n, h_py, ctypes.byref(cfg), w_c, hist_w_c)
    t_c = time.time() - t
    print(f"  Python {t_py:.1f} s, C {t_c * 1e3:.0f} ms")
    oks.append(comparar("w", w_c, w_py, hist_w_c[:ep_c], hist_w_py))

    print("3. Cadena completa en C, a Q0.15, contra el PA real, con la LTE")
    params, x, y = senal_lte(n)
    lib.dpd_train_cfg_default(ctypes.byref(cfg))
    # Con las épocas por defecto (40 del PA, 5 del DPD), como los vectores
    # dorados.
    if rapido:
        cfg.ep_pa = cfg.ep_dpd = ep
    lib.dpd_train_pa_model(x, y, n, ctypes.byref(cfg), h_c, hist_h_c)
    lib.dpd_train_dla(x, n, h_c, ctypes.byref(cfg), w_c, hist_w_c)
    re = np.zeros(NCOEF, np.int16)
    im = np.zeros(NCOEF, np.int16)
    sat = lib.dpd_coef_to_q15(w_c, re, im)
    w_q = (re.astype(float) + 1j * im.astype(float)) / 32768.0
    esperado = fxp.quantize(w_c, GV.CFG.coeff, GV.CFG)
    oks.append(bool(np.all(w_q == esperado)))
    print(f"  conversion a Q0.15 igual a fxpmath: {'OK' if oks[-1] else 'FALLA'} "
          f"({sat} saturado{'s' if sat != 1 else ''})")
    nm_c = nmse_directo(params, x, w_q)
    print(f"  NMSE directo, entrenado en C: {nm_c:6.2f} dB")
    if not rapido:
        dorados = coefs_dorados()
        iguales = int(np.sum(w_q == dorados))
        oks.append(iguales == NCOEF)
        print(f"  codigos Q0.15 iguales a los de los vectores dorados: {iguales} de {NCOEF}  "
              f"{'OK' if oks[-1] else 'FALLA'}")
        nm_d = nmse_directo(params, x, dorados)
        print(f"  NMSE directo, vectores dorados: {nm_d:6.2f} dB")
        oks.append(abs(nm_c - nm_d) < 0.1)
        print(f"  diferencia {nm_c - nm_d:+.2f} dB  {'OK' if oks[-1] else 'FALLA'}")

    print("\nTODO OK" if all(oks) else "\nHAY FALLAS")
    sys.exit(0 if all(oks) else 1)


if __name__ == "__main__":
    main()
