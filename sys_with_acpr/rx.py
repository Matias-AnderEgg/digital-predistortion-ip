import numpy as np


# 16QAM RX

def qam16_gray_demod(symbols):
    symbols = np.asarray(symbols)
    x = symbols * np.sqrt(10)

    def slicer_pam(v):
        if   v < -2: return -3
        elif v <  0: return -1
        elif v <  2: return +1
        else:        return +3

    pam2gray = {-3: (0, 0), -1: (0, 1), +1: (1, 1), +3: (1, 0)}

    bits_out = []
    sym_out  = np.zeros_like(symbols, dtype=complex)
    for k in range(len(symbols)):
        i_hat = slicer_pam(np.real(x[k]))
        q_hat = slicer_pam(np.imag(x[k]))
        bI = pam2gray[i_hat]
        bQ = pam2gray[q_hat]
        bits_out.extend([bI[0], bI[1], bQ[0], bQ[1]])
        sym_out[k] = (i_hat + 1j * q_hat) / np.sqrt(10)

    return np.array(bits_out, dtype=int), sym_out


# OFDM RX

def ofdm_demodulate(rx_signal, n_ofdm_symbols, Nfft, Nused, cp_len, os_factor):
    Nfft_os   = Nfft * os_factor
    cp_len_os = cp_len * os_factor
    block_len = Nfft_os + cp_len_os

    rx_grid = np.zeros((n_ofdm_symbols, Nused), dtype=complex)
    for m in range(n_ofdm_symbols):
        block = rx_signal[m * block_len:(m + 1) * block_len]
        if len(block) < block_len:
            break
        x = block[cp_len_os:]
        X_os_shifted = np.fft.fftshift(np.fft.fft(x, n=Nfft_os))
        center = Nfft_os // 2
        start  = center - Nused // 2
        X_active = X_os_shifted[start:start + Nused]
        rx_grid[m, :] = X_active / np.sqrt(Nfft_os / Nused)

    return rx_grid


# Métricas RX

def evm_rms_percent(ref, rx):
    """Alinea rx en ganancia/fase respecto a ref y calcula EVM RMS [%]."""
    ref = np.asarray(ref)
    rx  = np.asarray(rx)
    den = np.vdot(rx, rx)
    if np.abs(den) < 1e-30:
        return np.inf, rx.copy()
    a = np.vdot(rx, ref) / den
    rx_aligned = a * rx
    err = rx_aligned - ref
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(ref) ** 2))
    return 100 * evm, rx_aligned


def equalize_per_subcarrier(ref_grid, rx_grid):
    """Ecualizador de un coeficiente por subportadora, con canal conocido.

    Forzador de ceros (sin sesgo): para cada subportadora k,
    H_k = <ref_k, rx_k> / <ref_k, ref_k> sobre todos los símbolos OFDM, y la
    salida es rx_k / H_k. Corrige la respuesta en frecuencia lineal del PA
    (los filtros del Wiener-Hammerstein), que con un único escalar queda como
    error y no es distorsión no lineal.

    No se usa la escala de mínimo error cuadrático de evm_rms_percent,
    <rx, ref> / <rx, rx>, porque achica la constelación en SNR/(1+SNR) y
    sesga la BER hacia arriba (~8 % a 10 dB) y el EVM hacia abajo a SNR baja.
    """
    ref_grid = np.asarray(ref_grid)
    rx_grid  = np.asarray(rx_grid)
    H = np.sum(rx_grid * np.conj(ref_grid), axis=0) / (np.sum(np.abs(ref_grid) ** 2, axis=0) + 1e-30)
    return rx_grid / (H[None, :] + 1e-30)


def evm_percent(ref, eq):
    """EVM RMS [%] de una constelación ya ecualizada, sin realinear."""
    ref = np.asarray(ref).ravel()
    eq  = np.asarray(eq).ravel()
    return 100 * np.sqrt(np.mean(np.abs(eq - ref) ** 2) / np.mean(np.abs(ref) ** 2))


def compute_ber(bits_tx, bits_rx):
    n = min(len(bits_tx), len(bits_rx))
    return float(np.mean(bits_tx[:n] != bits_rx[:n]))
