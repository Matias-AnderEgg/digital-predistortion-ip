import numpy as np
from scipy.signal import welch


def compute_psd(x, fs, nperseg=4096):
    x = x / np.sqrt(np.mean(np.abs(x)**2) + 1e-30)
    f, Pxx = welch(
        x,
        fs=fs,
        window="hann",
        nperseg=min(nperseg, len(x)),
        return_onesided=False,
        scaling="density",
    )
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)
    Pxx_dB = 10 * np.log10(Pxx + 1e-20)
    return f, Pxx_dB, Pxx


def bandpower_from_psd(f, Pxx, f1, f2):
    idx = (f >= f1) & (f <= f2)
    if np.sum(idx) < 2:
        return 0.0
    return np.trapz(Pxx[idx], f[idx])


def compute_acpr(f, Pxx, main_bw, adj_offset, adj_bw):
    p_main = bandpower_from_psd(f, Pxx, -main_bw / 2, main_bw / 2)

    p_adj_u = bandpower_from_psd(
        f, Pxx,
        adj_offset - adj_bw / 2,
        adj_offset + adj_bw / 2,
    )
    p_adj_l = bandpower_from_psd(
        f, Pxx,
        -adj_offset - adj_bw / 2,
        -adj_offset + adj_bw / 2,
    )

    acpr_u = 10 * np.log10((p_adj_u + 1e-20) / (p_main + 1e-20))
    acpr_l = 10 * np.log10((p_adj_l + 1e-20) / (p_main + 1e-20))
    return acpr_l, acpr_u
