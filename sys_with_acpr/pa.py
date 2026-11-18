import numpy as np
from fractions import Fraction
from scipy.signal import lfilter, resample_poly


EPS = 1e-30

# Frecuencia interna de referencia para la memoria W-H.
PA_MEMORY_REF_FS = 15.36e6

# Espaciado de los retardos de los filtros del W-H, en muestras. Los filtros
# están definidos muestra a muestra, y con la señal LTE a 61,44 MHz y k = 1 la
# parte no lineal de la memoria (la que no es un filtro lineal y el receptor no
# corrige) queda 10 dB por debajo de la que tiene con la OFDM de 52
# portadoras a 3,84 MHz (-48,9 contra -38,4 dB). Con k = 2 (-36,7 dB) la memoria queda en
# retardos que el DPD de retardos consecutivos no modela y el DPD ensucia el
# espectro lejos del canal, así que se deja k = 1, el modelo tal cual está
# publicado.
PA_MEMORIA_K = 1

# Escala interna del modelo arctan.
ARCTAN_INPUT_SCALE = 0.2533


# Utilidades
def _as_complex_1d(x):
    return np.asarray(x, dtype=np.complex128).reshape(-1)

def _mean_power(x):
    x = _as_complex_1d(x)
    return float(np.mean(np.abs(x) ** 2))

def power_db(x):
    return float(10.0 * np.log10(_mean_power(x) + EPS))

def normalize_rms(x, target_rms=1.0):
    x = _as_complex_1d(x)
    p = _mean_power(x)
    g = float(target_rms / np.sqrt(p + EPS))
    return g * x, g

def _rational_ratio(fs_out, fs_in, max_den=512):
    frac = Fraction(float(fs_out) / float(fs_in)).limit_denominator(max_den)
    return frac.numerator, frac.denominator

def resample_to_fs(x, fs_in, fs_out):
    x = _as_complex_1d(x)
    if fs_out is None or np.isclose(fs_in, fs_out):
        return x.copy()
    up, down = _rational_ratio(fs_out, fs_in)
    return resample_poly(x, up, down)

def _match_length(y, n):
    if len(y) > n:
        return y[:n]
    if len(y) < n:
        return np.concatenate([y, np.zeros(n - len(y), dtype=np.complex128)])
    return y


# PA configurable: memoria W-H opcional + no linealidad seleccionable

def _pa_default_state(state=None):
    if state is None:
        state = {}
    return {
        "x1_z1": complex(state.get("x1_z1", 0.0 + 0.0j)),
        "x1_z2": complex(state.get("x1_z2", 0.0 + 0.0j)),
        "y1_z1": complex(state.get("y1_z1", 0.0 + 0.0j)),
        "x2_z1": complex(state.get("x2_z1", 0.0 + 0.0j)),
        "y2_z1": complex(state.get("y2_z1", 0.0 + 0.0j)),
    }


def _pa_static_nonlinearity(x, *, pa_nl="clip", sat_amp=1.0, p_soft=2.0):
    """
      - "clip": hard clipping de amplitud.
      - "soft": soft clipping tipo Rapp.
      - "arctan": no linealidad del PA Wiener-Hammerstein original.
    """
    x = _as_complex_1d(x)
    mag = np.abs(x)
    phase = np.exp(1j * np.angle(x))

    if pa_nl == "clip":
        return np.minimum(mag, sat_amp) * phase

    if pa_nl == "soft":
        mag_out = mag / (1.0 + (mag / (sat_amp + EPS)) ** (2.0 * p_soft)) ** (1.0 / (2.0 * p_soft))
        return mag_out * phase

    if pa_nl == "arctan":
        alpha1 = 8.00335 - 1j * 4.61157
        alpha2 = -3.77167 + 1j * 12.03758
        beta1 = 2.26895
        beta2 = 0.82345
        s = 1.0

        # Normalizar para que ganancia de pequeña señal = 1:
        # Para |x| << 1: arctan(b*s*|x|) ≈ b*s*|x|
        # => output ≈ (alpha1*beta1 + alpha2*beta2)*s * x
        G0 = alpha1 * beta1 * s + alpha2 * beta2 * s      # ganancia compleja real
        alpha1_n = alpha1 / G0
        alpha2_n = alpha2 / G0                             # así G_ss = 1

        # Normalizar Psat al mismo nivel que clip/soft (sat_amp):
        # Para |x|->inf: output -> (alpha1_n+alpha2_n)*pi/2
        psat_unit = float(np.abs(alpha1_n + alpha2_n) * np.pi / 2)  # Psat con sat_amp=1
        alpha1_n = alpha1_n / psat_unit * float(sat_amp)
        alpha2_n = alpha2_n / psat_unit * float(sat_amp)

        complex_amp = (alpha1_n * np.arctan(beta1 * s * mag) + alpha2_n * np.arctan(beta2 * s * mag))
        return complex_amp * phase

    raise ValueError(f"pa_nl desconocida: {pa_nl}")


# Memoria discreta W-H a frecuencia interna

def _apply_wh_memory_discrete(
    x_pa,
    *,
    pa_nl="clip",
    sat_amp=1.0,
    p_soft=2.0,
    pa_state=None,
):
    """
    Memoria discreta original del PA Wiener-Hammerstein.

    Esta función se debe ejecutar a la frecuencia interna de referencia
    PA_MEMORY_REF_FS. No debe interpretarse como memoria fija en cualquier
    frecuencia de muestreo externa.

    Los retardos de los dos filtros van espaciados PA_MEMORIA_K muestras
    (z^-1 -> z^-k). Con k = 1 es el filtro original, que con pa_state corre
    muestra a muestra y guarda el estado. Sin estado se filtra de una vez con
    lfilter, que con k = 1 da lo mismo que el lazo (error 1e-15).
    """
    x_pa = _as_complex_1d(x_pa)
    k = PA_MEMORIA_K

    if pa_state is None:
        b1 = np.zeros(2 * k + 1); b1[0], b1[2 * k] = 1 / 1.5, 0.25 / 1.5
        a1 = np.zeros(k + 1); a1[0], a1[k] = 1.0, 0.6 / 1.5
        b2 = np.zeros(k + 1); b2[0], b2[k] = 1 / 0.52, -0.1 / 0.52
        a2 = np.zeros(k + 1); a2[0], a2[k] = 1.0, -0.104 / 0.52
        d = _pa_static_nonlinearity(lfilter(b1, a1, x_pa), pa_nl=pa_nl, sat_amp=sat_amp, p_soft=p_soft)
        return lfilter(b2, a2, d), None

    if k != 1:
        raise ValueError("el PA con estado (pa_state) sólo admite PA_MEMORIA_K = 1")

    st = _pa_default_state(pa_state)
    x1_z1 = st["x1_z1"]
    x1_z2 = st["x1_z2"]
    y1_z1 = st["y1_z1"]
    x2_z1 = st["x2_z1"]
    y2_z1 = st["y2_z1"]

    z = np.empty_like(x_pa)

    for n in range(len(x_pa)):
        x_n = x_pa[n]

        # LTI1
        y1_n = (x_n + 0.25 * x1_z2 - 0.6 * y1_z1) / 1.5
        # No linealidad seleccionable.
        d_n = _pa_static_nonlinearity(np.array([y1_n], dtype=np.complex128),pa_nl=pa_nl,sat_amp=sat_amp,p_soft=p_soft,)[0]
        # LTI2
        z_n = (d_n - 0.1 * x2_z1 + 0.104 * y2_z1) / 0.52
        z[n] = z_n

        x1_z2 = x1_z1
        x1_z1 = x_n
        y1_z1 = y1_n
        x2_z1 = d_n
        y2_z1 = z_n

    new_state = {
        "x1_z1": x1_z1,
        "x1_z2": x1_z2,
        "y1_z1": y1_z1,
        "x2_z1": x2_z1,
        "y2_z1": y2_z1,
    }
    return z, new_state


def _apply_unified_pa_core(
    x_pa,
    *,
    pa_nl="clip",
    use_memory=False,
    sat_amp=1.0,
    p_soft=2.0,
    pa_state=None,
    fs_pa=PA_MEMORY_REF_FS,
    memory_ref_fs=None,  # None -> corre a fs_pa (memoria in-band, estilo referencia)
):

    x_pa = _as_complex_1d(x_pa)

    if not use_memory:
        return _pa_static_nonlinearity(x_pa,pa_nl=pa_nl,sat_amp=sat_amp,p_soft=p_soft,), pa_state

    # Para que la memoria sea invariante al oversampling externo, ejecutamos
    # el bloque W-H discreto en una frecuencia interna fija. Luego volvemos a
    # fs_pa y hacemos match de longitud.
    fs_pa = float(fs_pa if fs_pa is not None else memory_ref_fs)
    memory_ref_fs = float(memory_ref_fs if memory_ref_fs is not None else fs_pa)

    if np.isclose(fs_pa, memory_ref_fs):
        return _apply_wh_memory_discrete(x_pa,pa_nl=pa_nl,sat_amp=sat_amp,p_soft=p_soft,pa_state=pa_state,)

    n_in = len(x_pa)
    x_mem = resample_to_fs(x_pa, fs_pa, memory_ref_fs)
    y_mem, new_state = _apply_wh_memory_discrete(x_mem,pa_nl=pa_nl,sat_amp=sat_amp,p_soft=p_soft,pa_state=pa_state,)
    y_pa = resample_to_fs(y_mem, memory_ref_fs, fs_pa)
    y_pa = _match_length(y_pa, n_in)
    return y_pa, new_state

# Potencia de entrada de referencia del PA, con BO = 0

def _compute_pa_reference_pin_db(
    *,
    pa_nl="clip",
    use_memory=False,
    sat_amp=1.0,
    p_soft=2.0,
    fs_pa=PA_MEMORY_REF_FS,
    memory_ref_fs=None,  # None -> corre a fs_pa (memoria in-band, estilo referencia)
    n_samples=4096,
    discard=512,
    tone_bin_ratio=0.037109375,
):
    """
    Referencia de BO:

      - clip: BO=0 en la rodilla, o sea Pin = sat_amp^2.
      - soft/arctan: BO=0 en P1dB, estimado desde la curva AM-AM CW.

    Devuelve Pin_ref_dB relativo a una señal de RMS=1.
    """
    if pa_nl == "clip":
        return float(20.0 * np.log10(float(sat_amp) + EPS)), None

    pin_grid_db = np.linspace(-40.0, 15.0, 276)
    n = np.arange(n_samples)
    tone = np.exp(1j * 2.0 * np.pi * tone_bin_ratio * n)

    pin_db = []
    pout_db = []

    for pin_target_db in pin_grid_db:
        amp = 10.0 ** (float(pin_target_db) / 20.0)
        x_pa = amp * tone
        y_raw, _ = _apply_unified_pa_core(x_pa,pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,pa_state=None,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)
        pin_db.append(power_db(x_pa[discard:]))
        pout_db.append(power_db(y_raw[discard:]))

    pin_db = np.asarray(pin_db)
    pout_db = np.asarray(pout_db)

    gain_small_db = float(np.median(pout_db[:20] - pin_db[:20]))
    pout_ideal = pin_db + gain_small_db
    compression = pout_ideal - pout_db

    idx = np.where(compression >= 1.0)[0]
    if not idx.size:
        return 0.0, None

    i = int(idx[0])
    if i == 0:
        pin_1db = float(pin_db[0])
        pout_1db = float(pout_db[0])
    else:
        c0, c1 = compression[i - 1], compression[i]
        t = float((1.0 - c0) / (c1 - c0 + EPS))
        pin_1db = float(pin_db[i - 1] + t * (pin_db[i] - pin_db[i - 1]))
        pout_1db = float(pout_db[i - 1] + t * (pout_db[i] - pout_db[i - 1]))

    p1db = {
        "Pin_1dB_dBm": pin_1db,
        "Pout_1dB_dBm": pout_1db,
    }
    return pin_1db, p1db


# Función principal del PA

def apply_selected_pa(
    x_in,
    *,
    pa_mode,
    pa_nl=None,
    use_memory=None,
    bo_db,
    R0=50.0,
    fs_in=1.0,
    fs_pa=None,
    pa_state=None,
    sat_amp=1.0,
    p_soft=2.0,
    memory_ref_fs=None,  # None -> corre a fs_pa (memoria in-band, estilo referencia)
    normalize_input=True,
    normalize_output=True,
    **ignored_kwargs,
):
    """
    Convención limpia de potencia:

    1) Entrada del bloque opcionalmente normalizada a RMS = 1  -> 0 dB relativo.
    2) BO escala la señal que excita al PA respecto de una referencia:
           clip        -> BO=0 en la rodilla/sat_amp
           soft/arctan -> BO=0 en P1dB

       Entonces:
           Pin_PA_dB = Pin_ref_dB - BO
    3) El PA genera y_raw, que es la salida física/cruda del modelo.
    4) La salida del bloque opcionalmente se normaliza a RMS = 1 para RX/EVM/BER/ACPR.
    """
    x_in = _as_complex_1d(x_in)
    fs_pa = fs_in if fs_pa is None else fs_pa
    # memory_ref_fs=None -> la memoria W-H corre a fs_pa (in-band, estilo referencia)
    memory_ref_fs = fs_pa if memory_ref_fs is None else memory_ref_fs

    x_hi = resample_to_fs(x_in, fs_in, fs_pa)

    if normalize_input:
        x_norm, g_in = normalize_rms(x_hi, target_rms=1.0)
    else:
        x_norm = x_hi.copy()
        g_in = 1.0

    pin_ref_db, p1db_ref = _compute_pa_reference_pin_db(pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)
    scale_bo = float(10.0 ** ((pin_ref_db - float(bo_db)) / 20.0))
    x_pa = scale_bo * x_norm

    y_raw, new_state = _apply_unified_pa_core(x_pa,pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,pa_state=pa_state,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)

    if normalize_output:
        y_hi, g_out = normalize_rms(y_raw, target_rms=1.0)
    else:
        y_hi = y_raw.copy()
        g_out = 1.0

    y_out = resample_to_fs(y_hi, fs_pa, fs_in)
    y_out = _match_length(y_out, len(x_in))

    Pin_input = _mean_power(x_in)
    Pin_norm = _mean_power(x_norm)
    Pin_pa = _mean_power(x_pa)
    Pout_raw = _mean_power(y_raw)
    Pout_out = _mean_power(y_out)

    pa_info = {
        "mode": pa_mode,
        "pa_nl": pa_nl,
        "use_memory": bool(use_memory),
        "bo_db": float(bo_db),
        "sat_amp": float(sat_amp),
        "p_soft": float(p_soft),
        "fs_in": float(fs_in),
        "fs_pa": float(fs_pa),
        "memory_ref_fs": float(memory_ref_fs),
        "R0": float(R0),
        "normalize_input": bool(normalize_input),
        "normalize_output": bool(normalize_output),
        "scale_input_norm": float(g_in),
        "scale_bo": float(scale_bo),
        "Pin_ref_BO0_dB": float(pin_ref_db),
        "p1db_ref": p1db_ref,
        "scale_output_norm": float(g_out),
        "Pin_input_dB": float(10.0 * np.log10(Pin_input + EPS)),
        "Pin_norm_dB": float(10.0 * np.log10(Pin_norm + EPS)),
        "Pin_PA_dB": float(10.0 * np.log10(Pin_pa + EPS)),
        "Pout_PA_raw_dB": float(10.0 * np.log10(Pout_raw + EPS)),
        "Pout_output_dB": float(10.0 * np.log10(Pout_out + EPS)),
        "G_PA_raw_dB": float(10.0 * np.log10((Pout_raw + EPS) / (Pin_pa + EPS))),
    }

    if ignored_kwargs:
        pa_info["ignored_kwargs"] = sorted(ignored_kwargs.keys())

    return y_out, pa_info, new_state


# Curvas AM-AM y puntos de operación

def compute_pa_power_curve(
    *,
    pa_mode,
    pa_nl=None,
    use_memory=None,
    fs_in=1.0,
    fs_pa=None,
    R0=50.0,
    pin_grid_dbm=None,
    sat_amp=1.0,
    p_soft=2.0,
    memory_ref_fs=None,  # None -> corre a fs_pa (memoria in-band, estilo referencia)
    n_samples=4096,
    discard=512,
    tone_bin_ratio=0.037109375,
    **ignored_kwargs,
):
    """
    Curva AM-AM simple en dB relativos.

    Para no mezclar unidades físicas con la normalización del sistema:
    - Pin_dB es potencia relativa de entrada al PA.
    - Pout_dB es potencia relativa raw de salida del PA.
    """
    pin_ref_db, p1db_ref = _compute_pa_reference_pin_db(
        pa_nl=pa_nl,
        use_memory=use_memory,
        sat_amp=sat_amp,
        p_soft=p_soft,
        fs_pa=fs_pa if fs_pa is not None else fs_in,
        memory_ref_fs=memory_ref_fs,
        n_samples=n_samples,
        discard=discard,
        tone_bin_ratio=tone_bin_ratio,
    )
    if pin_grid_dbm is None:
        pin_grid_dbm = np.linspace(-30.0, 30.0, 241)

    fs_pa = fs_in if fs_pa is None else fs_pa
    n = np.arange(n_samples)
    tone = np.exp(1j * 2.0 * np.pi * tone_bin_ratio * n)

    pin_db = []
    pout_db = []

    for pin_target_db in np.asarray(pin_grid_dbm, dtype=float):
        amp = 10.0 ** (pin_target_db / 20.0)
        x_pa = amp * tone
        y_raw, _ = _apply_unified_pa_core(x_pa,pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,pa_state=None,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)
        pin_db.append(power_db(x_pa[discard:]))
        pout_db.append(power_db(y_raw[discard:]))

    pin_db = np.asarray(pin_db)
    pout_db = np.asarray(pout_db)
    gain_lin_db = float(np.median(pout_db[:10] - pin_db[:10]))
    pout_ideal = pin_db + gain_lin_db

    compression = pout_ideal - pout_db
    idx_1db = np.where(compression >= 1.0)[0]
    p1db = p1db_ref
    if p1db is None and idx_1db.size:
        i = int(idx_1db[0])
        p1db = {"Pin_1dB_dBm": float(pin_db[i]), "Pout_1dB_dBm": float(pout_db[i])}

    # Psat: primer Pin donde Pout >= max(Pout) - 1 dB
    pout_max = float(np.max(pout_db))
    idx_sat = np.where(pout_db >= pout_max - 1.0)[0]
    if idx_sat.size:
        psat_point = {"Pin_sat_dBm": float(pin_db[idx_sat[0]]), "Pout_sat_dBm": float(pout_db[idx_sat[0]])}
    else:
        psat_point = {"Pin_sat_dBm": float(pin_ref_db), "Pout_sat_dBm": float(np.interp(pin_ref_db, pin_db, pout_db))}

    return {
        "pin_dbm": pin_db,
        "pout_dbm": pout_db,
        "pout_ideal_dbm": pout_ideal,
        "gain_lin_db": gain_lin_db,
        "p1db": p1db,
        "psat_point": psat_point,
        "ref_type": "psat",
        "Pin_ref_dBm": float(pin_ref_db),
        "Pout_ref_dBm": float(np.interp(pin_ref_db, pin_db, pout_db)),
    }


def get_operating_point(
    x_in,
    *,
    pa_mode,
    pa_nl=None,
    use_memory=None,
    bo_db,
    R0=50.0,
    fs_in=1.0,
    fs_pa=None,
    sat_amp=1.0,
    p_soft=2.0,
    memory_ref_fs=None,  # None -> corre a fs_pa (memoria in-band, estilo referencia)
    **ignored_kwargs,
):
    """
    Punto de operación coherente con apply_selected_pa:
    clip        -> BO=0 en rodilla/sat_amp.
    soft/arctan -> BO=0 en P1dB.
    """
    x_in = _as_complex_1d(x_in)
    fs_pa = fs_in if fs_pa is None else fs_pa
    memory_ref_fs = fs_pa if memory_ref_fs is None else memory_ref_fs
    x_hi = resample_to_fs(x_in, fs_in, fs_pa)
    x_norm, _ = normalize_rms(x_hi, target_rms=1.0)
    pin_ref_db, _ = _compute_pa_reference_pin_db(pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)
    scale_bo = float(10.0 ** ((pin_ref_db - float(bo_db)) / 20.0))
    x_pa = scale_bo * x_norm
    y_raw, _ = _apply_unified_pa_core(x_pa,pa_nl=pa_nl,use_memory=use_memory,sat_amp=sat_amp,p_soft=p_soft,pa_state=None,fs_pa=fs_pa,memory_ref_fs=memory_ref_fs,)

    return {
        "Pin_dBm": power_db(x_pa),
        "Pout_dBm": power_db(y_raw),
        "bo_db": float(bo_db),
        "pa_nl": pa_nl,
        "use_memory": bool(use_memory),
        "memory_ref_fs": float(memory_ref_fs),
        "bo_reference": "clip_knee" if pa_nl == "clip" else "p1db",
        "ref_type_pa": "clip_knee" if pa_nl == "clip" else "p1db",
        "ref_type_bo": "clip_knee" if pa_nl == "clip" else "p1db",
        "Pin_ref_pa_dBm": float(pin_ref_db),
        "Pout_ref_pa_dBm": float(np.nan),
        "Pin_ref_bo_dBm": float(pin_ref_db),
        "Pout_ref_bo_dBm": float(np.nan),
        "scale_in_signal": scale_bo,
    }
