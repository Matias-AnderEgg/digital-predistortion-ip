import numpy as np


# 16QAM TX

def qam16_gray_mod(bits):
    bits = np.asarray(bits).astype(int)
    assert len(bits) % 4 == 0, "La cantidad de bits debe ser múltiplo de 4"

    b = bits.reshape(-1, 4)

    def gray2pam(b1, b0):
        if (b1, b0) == (0, 0):
            return -3
        elif (b1, b0) == (0, 1):
            return -1
        elif (b1, b0) == (1, 1):
            return +1
        elif (b1, b0) == (1, 0):
            return +3

    s = np.zeros(len(b), dtype=complex)
    for k in range(len(b)):
        i = gray2pam(b[k, 0], b[k, 1])
        q = gray2pam(b[k, 2], b[k, 3])
        s[k] = i + 1j * q

    return s / np.sqrt(10)


# OFDM TX

def map_to_oversampled_ifft(X_active, Nfft, os_factor):
    Nused = len(X_active)
    Nfft_os = Nfft * os_factor
    X_os = np.zeros(Nfft_os, dtype=complex)

    center = Nfft_os // 2
    start = center - Nused // 2
    X_os[start:start + Nused] = X_active
    return X_os


def ofdm_modulate(symbols, Nfft=64, Nused=52, cp_len=16, os_factor=4):
    assert Nused <= Nfft, "Nused debe ser <= Nfft"

    n_ofdm_symbols = int(np.ceil(len(symbols) / Nused))
    n_total = n_ofdm_symbols * Nused

    if len(symbols) < n_total:
        pad = np.zeros(n_total - len(symbols), dtype=complex)
        symbols = np.concatenate([symbols, pad])

    tx_grid = symbols.reshape(n_ofdm_symbols, Nused)

    Nfft_os = Nfft * os_factor
    cp_len_os = cp_len * os_factor

    tx_blocks = []

    for m in range(n_ofdm_symbols):
        X_active = tx_grid[m]
        X_os_shifted = map_to_oversampled_ifft(X_active, Nfft, os_factor)
        X_os = np.fft.ifftshift(X_os_shifted)

        x = np.fft.ifft(X_os, n=Nfft_os)
        x = x * np.sqrt(Nfft_os / Nused)

        x_cp = np.concatenate([x[-cp_len_os:], x])
        tx_blocks.append(x_cp)

    tx_signal = np.concatenate(tx_blocks)
    return tx_signal, tx_grid, Nfft_os, cp_len_os
