"""Señal de bajada LTE de 10 MHz (3GPP TS 36.211).

Numerologia: separación de 15 kHz, 50 bloques de recursos (600 portadoras),
FFT de 1024 a 15,36 MSPS, la portadora de continua apagada (300 abajo y 300
arriba), prefijo normal: 7 símbolos por ranura de 0,5 ms, el primero con
prefijo de 80 muestras y los otros 6 de 72 (a 15,36 MSPS). Se sobremuestrea
con una IFFT mas grande, igual que fixed_point/tx.py. La constelación la
pone quien llama (16QAM con tx.qam16_gray_mod). El filtro de transmisión va
aparte (filtro_tx y filtrar), después de modular.
"""
import numpy as np

NFFT, NRB, DF = 1024, 50, 15e3
NUSED = 12 * NRB
CP_RANURA = (80, 72, 72, 72, 72, 72, 72)


def params(os_factor=4):
    fs = NFFT * DF * os_factor
    return dict(Nfft=NFFT, Nused=NUSED, os_factor=os_factor, fs=fs, fs_pa=fs,
                main_bw=NUSED * DF, adj_offset=10e6, adj_bw=NUSED * DF,
                R0=50.0, pa_mode="arctan_mem", pa_nl="arctan", use_memory=True,
                sat_amp=1.0, p_soft=3.0)


def indices(os_factor):
    """Bins de la FFT sobremuestreada que ocupan las 600 portadoras."""
    n = NFFT * os_factor
    k = np.concatenate([np.arange(-NUSED // 2, 0), np.arange(1, NUSED // 2 + 1)])
    return k % n


def prefijos(n_simbolos, os_factor):
    return [CP_RANURA[m % 7] * os_factor for m in range(n_simbolos)]


def modular(qam, n_simbolos, os_factor=4):
    """qam: n_simbolos * 600 simbolos. Devuelve la senal con RMS 1 y la grilla."""
    n = NFFT * os_factor
    grilla = qam.reshape(n_simbolos, NUSED)
    idx = indices(os_factor)
    partes = []
    for m, cp in enumerate(prefijos(n_simbolos, os_factor)):
        X = np.zeros(n, complex)
        X[idx] = grilla[m]
        x = np.fft.ifft(X)
        partes.append(np.concatenate([x[-cp:], x]))
    x = np.concatenate(partes)
    return x / np.sqrt(np.mean(np.abs(x) ** 2)), grilla


def demodular(y, n_simbolos, os_factor=4, adelanto=0):
    """adelanto: cuántas muestras antes del fin del prefijo arranca la ventana
    de la FFT. Con el filtro de transmisión tiene que ser medio filtro
    (adelanto_filtro), porque con el retardo compensado el filtro mete en el
    final de cada símbolo el comienzo del siguiente. La fase lineal que agrega
    el corrimiento se corrige acá, así que la grilla no depende del adelanto."""
    n = NFFT * os_factor
    idx = indices(os_factor)
    fase = np.exp(2j * np.pi * idx * adelanto / n)
    grilla, pos = np.zeros((n_simbolos, NUSED), complex), 0
    for m, cp in enumerate(prefijos(n_simbolos, os_factor)):
        ini = pos + cp - adelanto
        grilla[m] = np.fft.fft(y[ini:ini + n])[idx] * fase
        pos += cp + n
    return grilla


def largo(n_simbolos, os_factor=4):
    return sum(prefijos(n_simbolos, os_factor)) + n_simbolos * NFFT * os_factor


def filtro_tx(os_factor=4, banda_paso=4.5e6, banda_rechazo=5.4e6, rechazo_db=70):
    """FIR de fase lineal (ventana de Kaiser) que limpia las faldas del OFDM
    antes del PA. Pasa hasta la última portadora (4,5 MHz) y rechaza desde
    5,4 MHz, antes del canal adyacente (que arranca en 5,5 MHz)."""
    from scipy.signal import firwin, kaiserord
    fs = NFFT * DF * os_factor
    n, beta = kaiserord(rechazo_db, (banda_rechazo - banda_paso) / (fs / 2))
    n |= 1  # impar: retardo entero
    return firwin(n, (banda_paso + banda_rechazo) / 2, window=("kaiser", beta), fs=fs)


def adelanto_filtro(os_factor=4):
    """Medio filtro de transmision, el adelanto de la ventana del receptor."""
    return (len(filtro_tx(os_factor)) - 1) // 2


def filtrar(x, h):
    """Filtra compensando el retardo, y vuelve a RMS 1."""
    d = (len(h) - 1) // 2
    y = np.convolve(x, h)[d:d + len(x)]
    return y / np.sqrt(np.mean(np.abs(y) ** 2))
