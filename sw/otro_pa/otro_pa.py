"""El piso del DPD a back-off alto, donde el PA es casi lineal.

Entrena con el mismo C que corre la placa (configuración por defecto, como
sw/hw_test/lte_placa.py) y mide sobre los 364 símbolos LTE de la tesis, con
la medición de lte_placa.metricas. Además del PA de la tesis (el
Wiener-Hammerstein de Ahluwalia, arctan) usa en algunos puntos el de
Gregorio, Gonzalez, Schmidt y Cousseau (Rapp con p = 2, FILTROS_LIBRO en
fixed_point/pa.py), que es casi lineal en más back-off.

Uso:
  python otro_pa.py piso            C, óptimo del costo y ajuste contra el PA
                                    real, deja resultados/piso.txt
  python otro_pa.py gradiente       el gradiente medio del GD del DPD en esos
                                    puntos, deja resultados/gradiente.txt
  python otro_pa.py normalizacion   normalización por muestra contra por
                                    bloque, deja resultados/normalizacion.txt

Cada uno tarda unos minutos. sys_with_acpr/piso_grilla.py y
meseta_optimo.py importan pa() de acá.
"""
import ctypes
import functools
import os
import sys

import numpy as np
from scipy.signal import lfilter

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, "..", "hw_test"))

import lte_placa as L  # noqa: E402
from fixed_point import pa as P  # noqa: E402

T, GV, dpd = L.T, L.GV, L.dpd
BO_LIST = [-9, -6, -3, 0, 3, 6, 9]
# El barrido de la tesis, extendido para que los dos PA cubran el mismo rango
# de back-off de salida: el de la tesis hacia la compresión y el del libro
# hacia la zona lineal.
BO_PA = {"tesis": [-24, -21, -18, -15, -12] + BO_LIST, "libro": BO_LIST + [12, 15, 18]}
PAS = {
    "tesis": dict(pa_nl="arctan", p_soft=3.0, pa_mem="ahluwalia"),
    "libro": dict(pa_nl="soft", p_soft=2.0, pa_mem="libro"),
}
# La referencia de BO (P1dB con un tono) es lo mas lento del PA y no cambia
# entre llamadas.
P._compute_pa_reference_pin_db = functools.lru_cache(P._compute_pa_reference_pin_db)


def pa(u, nombre, bo, normalizar_salida):
    """El PA con fixed_bo, como P.apply_selected_pa (entrada a RMS 1, escala
    del BO y salida a RMS 1 si se pide), pero con pa_rapido: el PA de la
    tesis muestra a muestra tarda minutos por llamada. Dan igual (1e-15)."""
    kw = PAS[nombre]
    ref, _ = P._compute_pa_reference_pin_db(
        pa_nl=kw["pa_nl"], use_memory=True, p_soft=kw["p_soft"],
        sat_amp=T.pa_kwargs(L.PARAMS)["sat_amp"], fs_pa=L.PARAMS["fs_pa"],
        memory_ref_fs=L.PARAMS["fs_pa"], pa_mem=kw["pa_mem"])
    u = np.asarray(u, complex)
    y = pa_rapido(10 ** ((ref - bo) / 20) * u / np.sqrt(np.mean(np.abs(u) ** 2)), **kw)
    return y / np.sqrt(np.mean(np.abs(y) ** 2)) if normalizar_salida else y


# Los filtros W-H como (numerador, denominador), para filtrar con lfilter.
# Los de Ahluwalia son los de _apply_wh_memory_discrete reescritos.
FILTROS = {"ahluwalia": (([1.0, 0.0, 0.25], [1.5, 0.6]), ([1.0, -0.1], [0.52, -0.104])),
           "libro": P.FILTROS_LIBRO}


def pa_rapido(x_pa, pa_nl, p_soft, pa_mem):
    """El mismo W-H que P._apply_unified_pa_core, pero vectorizado: la no
    linealidad es estatica, así que se puede filtrar el bloque entero."""
    (b_a, a_a), (b_b, a_b) = FILTROS[pa_mem]
    d = P._pa_static_nonlinearity(lfilter(b_a, a_a, x_pa), pa_nl=pa_nl, p_soft=p_soft)
    return lfilter(b_b, a_b, d)


def piso():
    """De donde sale el piso del DPD con BO alto, donde el PA es casi lineal.
    Tres DPD con la misma estructura (GV.PMAX, GV.M_TAPS, w[0] = 1):
      C        el del C: modelo del PA por GD y DPD por GD a través del modelo
      optimo   el que minimiza lo mismo que el GD del DPD (h0 x contra el
               modelo del PA), resuelto con least_squares: si da mejor que
               el C, el GD del DPD no converge
      PA real  el que minimiza el NMSE a través del PA verdadero, con la
               ganancia alineada como en la medición: si da mejor que optimo,
               el piso viene del modelo del PA; si no, de la estructura
    least_squares es solo diagnostico, el DPD sigue siendo por GD. Deja
    resultados/piso.txt."""
    from scipy.optimize import least_squares
    x_ent = L.senal(123)[0][:L.N_ENT]
    x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
    x, grilla = L.senal(456)
    mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
    nmse = lambda a, b: 10 * np.log10(np.mean(np.abs(a - b) ** 2) / np.mean(np.abs(b) ** 2))  # noqa: E731
    completo = lambda r: np.concatenate(([1.0 + 0j], r[:11] + 1j * r[11:]))  # noqa: E731
    reales = lambda w: np.concatenate((w[1:].real, w[1:].imag))  # noqa: E731
    lineas = [f"{'PA':6s} {'BO':>4s}  {'NMSE mod':>8s}  {'DPD':8s} {'costo DLA':>9s} {'NMSE ent':>8s}  "
              f"{'EVM':>7s} {'ACPR':>7s}"]
    print(lineas[0], flush=True)
    for nombre, bo in (("tesis", 9), ("libro", 9), ("libro", 18)):
        y_ent = np.ascontiguousarray(pa(x_ent, nombre, bo, False))
        lib = T.cargar_lib()
        cfg = T.Cfg()
        lib.dpd_train_cfg_default(ctypes.byref(cfg))
        h = np.zeros(12, np.complex128)
        w_c = np.zeros(12, np.complex128)
        lib.dpd_train_pa_model(x_ent, y_ent, len(x_ent), ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
        lib.dpd_train_dla(x_ent, len(x_ent), h, ctypes.byref(cfg), w_c, np.zeros(cfg.ep_dpd))
        nmse_mod = nmse(mp(x_ent, h), y_ent)

        def r_dla(r):
            e = mp(mp(x_ent, completo(r)), h) - h[0] * x_ent
            return np.concatenate((e.real, e.imag))

        def r_pa(r):
            y = pa(mp(x_ent, completo(r)), nombre, bo, False)
            e = np.vdot(y, x_ent) / np.vdot(y, y) * y - x_ent
            return np.concatenate((e.real, e.imag))

        w_opt = completo(least_squares(r_dla, reales(w_c), x_scale="jac").x)
        w_pa = completo(least_squares(r_pa, reales(w_opt), x_scale="jac").x)
        for etiqueta, w in (("C", w_c), ("optimo", w_opt), ("PA real", w_pa)):
            costo = 10 * np.log10(np.mean(np.abs(mp(mp(x_ent, w), h) - h[0] * x_ent) ** 2)
                                  / np.mean(np.abs(h[0] * x_ent) ** 2))
            y = pa(mp(x_ent, w), nombre, bo, False)
            y = np.vdot(y, x_ent) / np.vdot(y, y) * y
            nm, evm, acpr = L.metricas(x, grilla, pa(mp(x, w), nombre, bo, True))[:3]
            lineas.append(f"{nombre:6s} {bo:+4d}  {nmse_mod:8.2f}  {etiqueta:8s} {costo:9.2f} "
                          f"{nmse(y, x_ent):8.2f}  {evm:6.2f}% {acpr:7.1f}")
            print(lineas[-1], flush=True)
        print("  w C      ", np.round(np.abs(w_c), 5), flush=True)
        print("  w PA real", np.round(np.abs(w_pa), 5), flush=True)
    os.makedirs(os.path.join(AQUI, "resultados"), exist_ok=True)
    with open(os.path.join(AQUI, "resultados", "piso.txt"), "w") as fh:
        fh.write("\n".join(lineas) + "\n")


def _base(v):
    """phi[n, r, j] = v[n-r] |v[n-r]|^(p-1), con las columnas del C."""
    n = len(v)
    out = np.zeros((n, 3, 4), complex)
    for r in range(3):
        vr = np.concatenate((np.zeros(r), v[:n - r]))
        for j, p in enumerate((1, 3, 5, 7)):
            out[:, r, j] = vr * np.abs(vr) ** (p - 1)
    return out.reshape(n, 12)


def _grad_medio(x, h, w, G=8.0, delta=1e-3):
    """Suma sobre el bloque del gradiente por muestra de dpd_train_dla, sin
    normalizar (el del costo sum |e|^2) y dividido por reg_energy + delta como
    en el C. El precondicionador es una constante por coeficiente y no mueve
    los ceros, así que no se aplica. Devuelve los dos y reg_energy."""
    mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
    k = np.array([0, 2, 4, 6])
    s = np.tile(G ** k.astype(float), 3)
    hs = (h * s).reshape(3, 4)
    xs = x / G
    n = len(x)
    u = mp(xs, w * s)
    e = h[0] * xs - mp(u, h * s)
    phi = _base(xs)
    g = np.zeros((n, 12), complex)
    reg = np.zeros(n)
    for r in range(3):
        ur = np.concatenate((np.zeros(r), u[:n - r]))[:, None]
        a = np.abs(ur)
        with np.errstate(divide="ignore", invalid="ignore"):
            A = np.nan_to_num(np.sum(k[1:] * hs[r, 1:] * a ** (k[1:] - 2) * ur ** 2, axis=1))
        B = np.sum((k + 2) * np.conj(hs[r]) * a ** k, axis=1)
        pr = np.concatenate((np.zeros((r, 12)), phi[:n - r]))
        g += (-np.conj(e) * A - e * B)[:, None] * np.conj(pr)
        reg += (np.abs(A) ** 2 + np.abs(B) ** 2) * np.sum(np.abs(pr * s) ** 2, axis=1)
    g *= s
    reg *= G * G
    plano, nlms = g.sum(0), (g / (reg + delta)[:, None]).sum(0)
    plano[0] = nlms[0] = 0  # w[0] queda fijo
    return plano, nlms, reg


def gradiente():
    """Si el GD del DPD apunta al mínimo de sum |e|^2. Mide el gradiente medio
    en el arranque, en el optimo (least_squares, solo diagnostico) y donde
    queda el C con paso chico y sin fuga, y resuelve el punto fijo de la
    normalización por muestra: el mínimo de sum |e|^2 / (reg_energy + delta),
    con los pesos iterados. Deja resultados/gradiente.txt."""
    from scipy.optimize import least_squares
    x_ent = L.senal(123)[0][:L.N_ENT]
    x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
    x, grilla = L.senal(456)
    mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
    completo = lambda r: np.concatenate(([1.0 + 0j], r[:11] + 1j * r[11:]))  # noqa: E731
    reales = lambda w: np.concatenate((w[1:].real, w[1:].imag))  # noqa: E731
    lib = T.cargar_lib()
    lineas = [f"{'PA':6s} {'BO':>4s}  {'w':14s} {'costo DLA':>9s}  {'|grad|':>8s} {'|grad NLMS|':>11s}  "
              f"{'EVM':>7s} {'ACPR':>7s}"]
    print(lineas[0], flush=True)
    for nombre, bo in (("tesis", 9), ("libro", 18)):
        y_ent = np.ascontiguousarray(pa(x_ent, nombre, bo, False))
        cfg = T.Cfg()
        lib.dpd_train_cfg_default(ctypes.byref(cfg))
        h = np.zeros(12, np.complex128)
        lib.dpd_train_pa_model(x_ent, y_ent, len(x_ent), ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
        cfg.mu_w, cfg.w_reg_lambda, cfg.ep_dpd = 0.005, 0.0, 160
        w_gd = np.zeros(12, np.complex128)
        lib.dpd_train_dla(x_ent, len(x_ent), h, ctypes.byref(cfg), w_gd, np.zeros(160))

        def residuo(r, peso=1.0):
            e = (mp(mp(x_ent, completo(r)), h) - h[0] * x_ent) * peso
            return np.concatenate((e.real, e.imag))

        w_opt = completo(least_squares(residuo, reales(w_gd), x_scale="jac").x)
        w_fijo = w_gd.copy()
        for _ in range(6):
            peso = 1 / np.sqrt(_grad_medio(x_ent, h, w_fijo)[2] + 1e-3)
            w_fijo = completo(least_squares(residuo, reales(w_fijo), args=(peso,), x_scale="jac").x)
        w_id = np.zeros(12, np.complex128)
        w_id[0] = 1
        ref = None
        for etiqueta, w in (("arranque", w_id), ("optimo", w_opt), ("C, mu chico", w_gd),
                            ("punto fijo", w_fijo)):
            plano, nlms, reg = _grad_medio(x_ent, h, w)
            if ref is None:
                ref = np.linalg.norm(plano), np.linalg.norm(nlms)
            costo = 10 * np.log10(np.mean(np.abs(mp(mp(x_ent, w), h) - h[0] * x_ent) ** 2)
                                  / np.mean(np.abs(h[0] * x_ent) ** 2))
            _, evm, acpr = L.metricas(x, grilla, pa(mp(x, w), nombre, bo, True))[:3]
            lineas.append(f"{nombre:6s} {bo:+4d}  {etiqueta:14s} {costo:9.2f}  "
                          f"{np.linalg.norm(plano) / ref[0]:8.1e} {np.linalg.norm(nlms) / ref[1]:11.1e}  "
                          f"{evm:6.2f}% {acpr:7.1f}")
            print(lineas[-1], flush=True)
        lineas.append(f"{'':12s} reg_energy en el arranque: maximo / mediana = "
                      f"{_grad_medio(x_ent, h, w_id)[2].max() / np.median(_grad_medio(x_ent, h, w_id)[2]):.0f}")
        print(lineas[-1], flush=True)
    with open(os.path.join(AQUI, "resultados", "gradiente.txt"), "w") as fh:
        fh.write("\n".join(lineas) + "\n")


def dla_py(x, h, mu=0.5, ep=10, lam=1e-4, norma="muestra", G=8.0, delta=1e-3, clip=1e3,
           techo=None, w_ini=None, mb=1, delta_col=None):
    """El dpd_train_dla del C en Python, con la configuración por defecto,
    para probar la normalización sin tocar el C de la placa.
      norma="muestra"  la del C: cada gradiente dividido por la energía de
                       su regresor, reg_energy, que en los picos vale 1e5
                       veces la mediana y hace que el GD casi no los vea
      norma="bloque"   dividido por la media de reg_energy en la época
                       anterior (la primera, con una pasada en el w inicial):
                       el punto fijo pasa a ser el mínimo de sum |e|^2
    Con "muestra" da igual que el C (ver normalización()).
    techo: si se da, la señal deseada es h0 x recortada a esa amplitud (en
    las unidades de la salida del PA) con la fase de x, el limitador ideal,
    en vez de h0 x.
    w_ini: arranca de ese w (warm_start del C), si no de la identidad.
    delta_col: el delta del precondicionador de columnas. Por defecto
    delta / G^2, como el C.
    norma="minibloque" junta el gradiente y la energía de mb muestras y da un
    paso cada mb, dividido por la energía del minibloque: un pico pesa como
    las otras muestras del minibloque y no 1e5 veces menos, y su golpe se
    reparte entre mb. La fuga se aplica por muestra, como en el C."""
    orden = np.array([1, 3, 5, 7])
    s = np.tile(G ** (orden - 1.0), 3)
    hs = (np.asarray(h) * s).reshape(3, 4)
    k = orden - 1
    alto = np.tile(orden >= 3, 3)
    xs = np.asarray(x) / G
    n = len(xs)
    phi = np.zeros((n + 2, 12), complex)  # dos ceros adelante: la historia
    for q in range(3):
        xq = np.concatenate((np.zeros(q), xs[:n - q]))
        phi[2:, 4 * q:4 * q + 4] = xq[:, None] * np.abs(xq[:, None]) ** k
    phi_s = phi * s
    en_phi = np.sum(np.abs(phi_s) ** 2, axis=1)
    inv_pre = 1.0 / (np.mean(np.abs(phi_s[2:]) ** 2, axis=0)
                     + (delta / (G * G) if delta_col is None else delta_col))
    if w_ini is None:
        w = np.zeros(12, complex)
        w[0] = 1.0
    else:
        w = np.array(w_ini, complex)

    def terminos(uh):
        a = np.abs(uh)[:, None]
        u = uh[:, None]
        A = np.sum(k[1:] * hs[:, 1:] * a ** (k[1:] - 2) * u ** 2, axis=1) if a.min() > 1e-30 else \
            np.array([np.sum(k[1:] * hs[r, 1:] * a[r] ** (k[1:] - 2) * u[r] ** 2) if a[r] > 1e-30 else 0
                      for r in range(3)])
        B = np.sum((k + 2) * np.conj(hs) * a ** k, axis=1)
        dh = np.sum(hs * u * a ** k)
        return A, B, dh

    def energia_media(w):
        u = phi_s[2:] @ w
        uh = np.stack([np.concatenate((np.zeros(r), u[:n - r])) for r in range(3)], axis=1)
        a = np.abs(uh)[:, :, None]
        A = np.sum(k[1:] * hs[None, :, 1:] * a ** (k[1:] - 2) * uh[:, :, None] ** 2, axis=2)
        B = np.sum((k + 2) * np.conj(hs)[None] * a ** k, axis=2)
        e_r = np.stack([en_phi[2 - r:n + 2 - r] for r in range(3)], axis=1)
        return G * G * np.mean(np.sum((np.abs(A) ** 2 + np.abs(B) ** 2) * e_r, axis=1))

    reg_media = energia_media(w) if norma == "bloque" else None
    for _ in range(ep):
        uh = np.zeros(3, complex)
        acc = 0.0
        g_mb, reg_mb = 0.0, 0.0
        for i in range(n):
            u_n = np.dot(w, phi_s[i + 2])
            uh = np.array([u_n, uh[0], uh[1]])
            A, B, dh = terminos(uh)
            d = hs[0, 0] / s[0] * xs[i]
            if techo is not None and abs(d) > techo / G:
                d *= techo / G / abs(d)
            e = d - dh
            coef = -np.conj(e) * A - e * B
            ph = phi_s[i + 2 - np.arange(3)]
            g = coef @ np.conj(ph)
            reg = G * G * np.sum((np.abs(A) ** 2 + np.abs(B) ** 2) * en_phi[i + 2 - np.arange(3)])
            acc += reg
            if norma == "minibloque":
                g_mb = g_mb + g
                reg_mb += reg
                if (i + 1) % mb:
                    if lam:
                        w[alto] *= 1 - lam
                    continue
                g, g_mb = g_mb * inv_pre / (reg_mb + delta), 0.0
                reg_mb = 0.0
            else:
                g = g * inv_pre / ((reg if norma == "muestra" else reg_media) + delta)
            g[0] = 0
            nrm = np.linalg.norm(g)
            if nrm > clip:
                g *= clip / (nrm + 1e-30)
            w = w - mu * g
            if lam:
                w[alto] *= 1 - lam
            w[0] = 1.0
        if not np.all(np.isfinite(w)):
            return None
        reg_media = acc / n
    return w


def normalizacion():
    """Normalización del GD del DPD por muestra (la del C) contra por bloque,
    en todo el barrido de los dos PA, con el modelo del PA del C. Deja
    resultados/normalizacion.txt."""
    x_ent = L.senal(123)[0][:L.N_ENT]
    x_ent = np.ascontiguousarray(x_ent / np.sqrt(np.mean(np.abs(x_ent) ** 2)))
    x, grilla = L.senal(456)
    mp = lambda v, c: dpd.apply_memory_poly(v, c, Pmax=GV.PMAX, M=GV.M_TAPS)  # noqa: E731
    lib = T.cargar_lib()
    # (mu_w, fuga): la del C, y sin fuga, que con la normalización por
    # muestra empeora todo
    variantes = [(0.5, 1e-4), (0.5, 0.0), (0.1, 0.0)]
    cab = "  ".join(f"bloque mu={mu:<3} fuga={lam:g}".ljust(15) for mu, lam in variantes)
    lineas = [f"{'PA':6s} {'BO':>4s}  {'sin DPD':14s}  {'C, por muestra':14s}  {cab}"]
    print(lineas[0], flush=True)
    for nombre in PAS:
        for bo in BO_PA[nombre]:
            y_ent = np.ascontiguousarray(pa(x_ent, nombre, bo, False))
            cfg = T.Cfg()
            lib.dpd_train_cfg_default(ctypes.byref(cfg))
            h = np.zeros(12, np.complex128)
            w_c = np.zeros(12, np.complex128)
            lib.dpd_train_pa_model(x_ent, y_ent, len(x_ent), ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
            lib.dpd_train_dla(x_ent, len(x_ent), h, ctypes.byref(cfg), w_c, np.zeros(cfg.ep_dpd))
            if nombre == "libro" and bo == 18:
                dw = np.max(np.abs(dla_py(x_ent, h) - w_c))
                print(f"  dla_py por muestra contra el C: |dw| = {dw:.1e}", flush=True)
            sin = L.metricas(x, grilla, pa(x, nombre, bo, True))
            celdas = [f"{sin[1]:5.2f}% {sin[2]:6.1f}"]
            for w in [w_c] + [dla_py(x_ent, h, mu=mu, lam=lam, norma="bloque") for mu, lam in variantes]:
                if w is None:
                    celdas.append(f"{'diverge':>13s}")
                    continue
                m = L.metricas(x, grilla, pa(mp(x, w), nombre, bo, True))
                celdas.append(f"{m[1]:5.2f}% {m[2]:6.1f}")
            lineas.append(f"{nombre:6s} {bo:+4d}  " + "   ".join(celdas))
            print(lineas[-1], flush=True)
    with open(os.path.join(AQUI, "resultados", "normalizacion.txt"), "w") as fh:
        fh.write("\n".join(lineas) + "\n")


if __name__ == "__main__":
    modos = {"piso": piso, "gradiente": gradiente, "normalizacion": normalizacion}
    if len(sys.argv) != 2 or sys.argv[1] not in modos:
        sys.exit(__doc__)
    modos[sys.argv[1]]()
