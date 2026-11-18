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


def compute_ber(bits_tx, bits_rx):
    n = min(len(bits_tx), len(bits_rx))
    return float(np.mean(bits_tx[:n] != bits_rx[:n]))
