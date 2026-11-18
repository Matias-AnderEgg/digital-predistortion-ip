"""Validación de la cadena con un canal selectivo en frecuencia.

Misma señal LTE de 10 MHz que main.py (lte.py: FFT de 1024, 600 portadoras,
prefijo normal, sobremuestreo x4), sin amplificador y sin el filtro de
transmisión, para que el canal sea el único efecto lineal y la ganancia de cada
subportadora sea exactamente la del canal. La señal sobremuestreada pasa por un
canal Rayleigh de L = 288 coeficientes (el prefijo más corto a la tasa
sobremuestreada) con perfil de potencia exponencial, normalizado a potencia
media uno, de modo que cada subportadora ve una ganancia gaussiana compleja de
potencia media uno. El canal cambia cada bloque de 14 símbolos, una subtrama.

Diversidad por repetición: cada bloque se transmite N veces por N canales
independientes, y el receptor combina las N copias por subportadora con MRC
(máxima razón), con conocimiento perfecto del canal:
    s = sum_i conj(H_i) r_i / sum_i |H_i|^2
El ruido se suma por subportadora después de la FFT. La SNR del eje es Es/N0
media por rama.

Referencias teóricas, 16-QAM Gray:
    BER(N) = E[ Pb(SNR * g) ],  g ~ Gamma(N, 1)  (integral numérica)
    EVM(N) = 1 / sqrt((N - 1) SNR)  para N >= 2 (para N = 1 no existe)
con Pb(x) = 1/4 [3 Q(a) + 2 Q(3a) - Q(5a)], a = sqrt(x/5), la exacta.

Cada punto acumula bloques hasta 50 errores de bit o 1e7 bits de datos, y
como mínimo 4 lotes (200 realizaciones del canal) para promediar el
desvanecimiento.

Uso:
    python sys_with_acpr/validacion_canal.py
Escribe sys_with_acpr/resultados_validacion/validacion_canal.pkl.

La tesis muestra sólo el caso sin diversidad (N = 1), porque la cadena con
predistorsión no combina copias. N = 8 también se calcula.
"""
import os
import pickle
import sys
import time

import numpy as np
from scipy import integrate
from scipy.signal import fftconvolve
from scipy.special import erfc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from sys_with_acpr.tx import qam16_gray_mod  # noqa: E402
from sys_with_acpr import lte  # noqa: E402
from sys_with_acpr.rx import qam16_gray_demod  # noqa: E402
from sys_with_acpr.barrido_mp import PARAMS  # noqa: E402

SALIDA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados_validacion",
                      "validacion_canal.pkl")

L_CANAL = min(lte.CP_RANURA) * PARAMS["os_factor"]  # 288 coeficientes
TAU = L_CANAL / 4                                 # decaimiento, en muestras sobremuestreadas
NSYM_BLK = 14                                     # símbolos por realización del canal
BLK_POR_LOTE = 50
DIVERSIDAD = [1, 8]
SNR_GRILLA = [float(s) for s in np.arange(0.0, 40.1, 2.0)]
MIN_ERRORES = 50
MAX_BITS = 1e7
MIN_LOTES = 4                                     # al menos 200 realizaciones del canal


def pb_awgn(x):
    """BER exacta de 16-QAM Gray en ruido blanco, con x = Es/N0 lineal."""
    q = lambda v: 0.5 * erfc(v / np.sqrt(2))
    a = np.sqrt(np.asarray(x, float) / 5)
    return 0.25 * (3 * q(a) + 2 * q(3 * a) - q(5 * a))


def pb_mrc(snr_db, n):
    """BER de 16-QAM con MRC de n ramas Rayleigh independientes."""
    from scipy.stats import gamma
    s = 10 ** (snr_db / 10)
    f = lambda t: pb_awgn(s * t) * gamma.pdf(t, n)
    return integrate.quad(f, 0, np.inf, limit=200)[0]


def perfil():
    p = np.exp(-np.arange(L_CANAL) / TAU)
    return p / p.sum()


def canales(rng, n):
    """n respuestas al impulso Rayleigh, cada una de potencia media uno."""
    p = perfil()
    return np.sqrt(p / 2)[None, :] * (rng.standard_normal((n, L_CANAL))
                                      + 1j * rng.standard_normal((n, L_CANAL)))


NFO = PARAMS["Nfft"] * PARAMS["os_factor"]
IDX = lte.indices(PARAMS["os_factor"])            # bins de las 600 portadoras
CPS = lte.prefijos(NSYM_BLK, PARAMS["os_factor"])


def respuesta(h):
    """Ganancia del canal en cada subportadora usada, con el mismo orden de
    bins que lte.demodular."""
    return np.fft.fft(h, n=NFO, axis=-1)[..., IDX]


def modular(sims):
    """lte.modular vectorizado sobre la primera dimensión y sin normalizar la
    potencia, para que demodular devuelva los símbolos a su escala."""
    X = np.zeros(sims.shape[:2] + (NFO,), complex)
    X[..., IDX] = sims
    x = np.fft.ifft(X, axis=-1)
    return np.concatenate([np.concatenate([x[:, m, -cp:], x[:, m]], axis=1)
                           for m, cp in enumerate(CPS)], axis=1)


def demodular(y):
    """lte.demodular vectorizado sobre la primera dimensión."""
    out, pos = [], 0
    for cp in CPS:
        out.append(np.fft.fft(y[:, pos + cp:pos + cp + NFO], axis=-1)[:, IDX])
        pos += cp + NFO
    return np.stack(out, axis=1)


def lote(rng):
    """BLK_POR_LOTE bloques: bits, símbolos y señal sobremuestreada."""
    nb = PARAMS["Nused"] * 4 * NSYM_BLK
    bits = rng.integers(0, 2, (BLK_POR_LOTE, nb))
    sims = np.asarray([qam16_gray_mod(b).reshape(NSYM_BLK, PARAMS["Nused"]) for b in bits])
    return bits, sims, modular(sims)


def punto(n, snr, rng):
    err = tot = 0
    num = den = 0.0
    sigma = np.sqrt(10 ** (-snr / 10) / 2)
    lotes = 0
    while (err < MIN_ERRORES or lotes < MIN_LOTES) and tot < MAX_BITS:
        lotes += 1
        bits, sims, xs = lote(rng)
        acum = np.zeros(sims.shape, complex)
        pot = np.zeros(sims.shape, float)
        for _ in range(n):
            h = canales(rng, BLK_POR_LOTE)
            y = fftconvolve(xs, h, axes=1)[:, :xs.shape[1]]
            g = demodular(y)
            g = g + sigma * (rng.standard_normal(g.shape) + 1j * rng.standard_normal(g.shape))
            H = respuesta(h)[:, None, :]
            acum += np.conj(H) * g
            pot += np.abs(H) ** 2
        est = acum / pot
        num += float(np.sum(np.abs(est - sims) ** 2))
        den += float(np.sum(np.abs(sims) ** 2))
        for b, e in zip(bits, est):
            br, _ = qam16_gray_demod(e.ravel())
            err += int(np.sum(b != br[:len(b)]))
            tot += len(b)
    return err / tot, 100 * np.sqrt(num / den), tot, err


def comprobar_canal(rng):
    """Sin ruido, demodular / símbolo tiene que dar exactamente H_k: confirma
    que el prefijo cubre el canal y que el orden de bins es el correcto."""
    _, sims, xs = lote(rng)
    h = canales(rng, BLK_POR_LOTE)
    g = demodular(fftconvolve(xs, h, axes=1)[:, :xs.shape[1]])
    return float(np.max(np.abs(g / sims - respuesta(h)[:, None, :])))


def main():
    rng = np.random.default_rng(2026)
    t0 = time.time()
    print(f"error máximo de H sin ruido: {comprobar_canal(rng):.2e}", flush=True)
    res = {}
    for n in DIVERSIDAD:
        res[n] = {"snr": [], "ber": [], "evm": [], "bits": [], "errores": [], "ber_teo": []}
        ceros = 0
        for snr in SNR_GRILLA:
            ber, evm, tot, err = punto(n, snr, rng)
            for k, v in zip(["snr", "ber", "evm", "bits", "errores", "ber_teo"],
                            [snr, ber, evm, tot, err, pb_mrc(snr, n)]):
                res[n][k].append(v)
            print(f"N={n} SNR {snr:4.0f} dB  BER {ber:.2e} (teórica {res[n]['ber_teo'][-1]:.2e},"
                  f" {err} errores en {tot:.2e} bits)  EVM {evm:6.2f} %", flush=True)
            ceros = ceros + 1 if err == 0 else 0
            if ceros >= 2:
                break
    h = canales(np.random.default_rng(976), 1)[0]
    ejemplo = {"h": h, "f": np.fft.fftshift(np.fft.fftfreq(NFO, 1 / PARAMS["fs"])),
               "H": np.fft.fftshift(np.fft.fft(h, n=NFO)), "H_sub": respuesta(h)}
    meta = {"L_CANAL": L_CANAL, "TAU": TAU, "NSYM_BLK": NSYM_BLK, "DIVERSIDAD": DIVERSIDAD,
            "MIN_ERRORES": MIN_ERRORES, "MAX_BITS": MAX_BITS, "MIN_LOTES": MIN_LOTES,
            "PARAMS": PARAMS,
            "fecha": time.strftime("%Y-%m-%d %H:%M"), "minutos": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(SALIDA), exist_ok=True)
    with open(SALIDA, "wb") as f:
        pickle.dump({"meta": meta, "res": res, "ejemplo": ejemplo}, f)
    print(f"Guardado en {SALIDA} ({meta['minutos']:.1f} min)")


if __name__ == "__main__":
    main()
