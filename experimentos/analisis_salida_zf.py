"""Números del Apéndice B y del camino de datos sobre la escala de salida del
núcleo, con la señal LTE.

No dibuja nada. Entrena el predistorsionador elegido como en el barrido
(sys_with_acpr/barrido_mp.py, M = 3, P = 7, BO -9 dB) y mide sobre su salida,
antes del PA:

  1. error del modelo de punto fijo contra el flotante, con la magnitud por
     |x|^2 (sin raíz, como el núcleo) y con la raíz;
  2. con G_out = G = 8 y con G_out = 128: cuántas muestras recortan y el
     error contra el flotante;
  3. el pico de la salida a medida que se observan mas muestras, con 10
     señales de 364 símbolos (semillas 456 a 465, la 456 es la de evaluación),
     y cuántas pasan de 128 en total;
  4. el ruido de cuantizacion de la salida con G_out = 128.

Una corrida con el entrenamiento de la tesis (paso del modelo limitado a 1 y
16000 muestras, 10 épocas del DPD):

    señal de evaluación: 1597440 muestras, pico de |u|/rms 31.3

    1. magnitud sin raíz y con raíz, G_out = 128
       squared  error contra el flotante -54.92 dB
       sqrt     error contra el flotante -54.92 dB

    2. escala de salida
       G_out     8: recortan 0.4230 % (1 de cada 236), error contra el flotante -14.7 dB, margen sobre el pico -17.0 dB
       G_out   128: recortan 0.0000 % (ninguna), error contra el flotante -54.9 dB, margen sobre el pico 7.1 dB

    3. pico de la salida en función de las muestras observadas
       N     16000: pico de componente   18.9, margen a 128  16.6 dB
       N    160000: pico de componente   33.4, margen a 128  11.7 dB
       N   1597440: pico de componente   56.7, margen a 128   7.1 dB
       N  15974400: pico de componente  110.8, margen a 128   1.3 dB
       0 muestras pasan de 128 en 15974400 (260 ms de señal)

    4. ruido de cuantizacion de la salida con G_out = 128
       -61.2 dB respecto de la señal

Con las 5 épocas del DPD de la tesis da lo mismo salvo el pico de |u|/rms,
38.1, y 1 de cada 245 con G_out = 8.

Uso: PYTHONDONTWRITEBYTECODE=1 python experimentos/analisis_salida_zf.py
"""
import contextlib
import io
import os
import sys

import numpy as np

RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "fixed_point", "golden_vectors"))
sys.path.insert(0, RAIZ)

import generate as GV  # noqa: E402
from fixed_point import fxp_format as fxp  # noqa: E402
from fixed_point.dpd import apply_memory_poly_fxp  # noqa: E402
from sys_with_acpr import dpd as D  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS, DPD_CFG, _bits  # noqa: E402
from sys_with_acpr.main import generate_ofdm_signal, simulate_with_dpd  # noqa: E402

BO, M, P = -9.0, 3, 7
SEMILLAS = range(456, 466)


def config(g_out, modo="squared"):
    return fxp.make_normalized_config(n_word_data=16, n_word_coef=16, n_word_coef_accum=24,
                                      coef_n_int=0, signal_scale=GV.G, output_scale=g_out,
                                      magnitude_mode=modo)


def db(v):
    return 10 * np.log10(v)


bits_tr, bits = _bits()
with contextlib.redirect_stdout(io.StringIO()):
    r = simulate_with_dpd(bits_tr, bits, PARAMS, BO, method="gd", dpd_cfg=dict(DPD_CFG, M=M, Pmax=P))
w = r["train_out"]["dpd_coeffs"]
wq = fxp.quantize(w, GV.CFG.coeff, GV.CFG)


def senal(semilla_bits):
    tx, _ = generate_ofdm_signal(semilla_bits, PARAMS)
    x, _ = D.normalize_rms(tx, target_rms=1.0)
    return x


x = senal(bits)
uf = D.apply_memory_poly(x, wq, Pmax=P, M=M)
p_u = np.mean(np.abs(uf) ** 2)
pico = max(np.abs(uf.real).max(), np.abs(uf.imag).max())
print(f"senal de evaluacion: {len(x)} muestras, pico de |u|/rms {np.abs(uf).max() / np.sqrt(p_u):.1f}")

print("\n1. magnitud sin raiz y con raiz, G_out = 128")
for modo in ("squared", "sqrt"):
    ux = apply_memory_poly_fxp(x, wq, Pmax=P, M=M, cfg=config(128.0, modo))
    print(f"   {modo:8s} error contra el flotante {db(np.mean(np.abs(ux - uf) ** 2) / p_u):.2f} dB")

print("\n2. escala de salida")
for g_out in (GV.G, 128.0):
    ux = apply_memory_poly_fxp(x, wq, Pmax=P, M=M, cfg=config(g_out))
    lim = g_out * (1 - 2 ** -15) * 0.99999
    rec = np.mean((np.abs(ux.real) >= lim) | (np.abs(ux.imag) >= lim))
    uno_cada = f"1 de cada {1 / rec:.0f}" if rec else "ninguna"
    print(f"   G_out {g_out:5.0f}: recortan {rec * 100:.4f} % ({uno_cada}), error contra el flotante "
          f"{db(np.mean(np.abs(ux - uf) ** 2) / p_u):.1f} dB, margen sobre el pico {20 * np.log10(g_out / pico):.1f} dB")

print("\n3. pico de la salida en funcion de las muestras observadas")
rng_bits = [bits if s == 456 else np.random.default_rng(s).integers(0, 2, len(bits)) for s in SEMILLAS]
u = np.concatenate([D.apply_memory_poly(senal(b), w, Pmax=P, M=M) for b in rng_bits])
comp = np.maximum(np.abs(u.real), np.abs(u.imag))
for n in (16_000, 160_000, len(x), len(u)):
    print(f"   N {n:9d}: pico de componente {comp[:n].max():6.1f}, margen a 128 {20 * np.log10(128 / comp[:n].max()):5.1f} dB")
n_rec = int(np.sum(comp > 128))
print(f"   {n_rec} muestras pasan de 128 en {len(u)} ({len(u) / PARAMS['fs'] * 1e3:.0f} ms de senal)")

print("\n4. ruido de cuantizacion de la salida con G_out = 128")
lsb = 128 / 32768
print(f"   {db(2 * lsb ** 2 / 12 / p_u):.1f} dB respecto de la senal")
