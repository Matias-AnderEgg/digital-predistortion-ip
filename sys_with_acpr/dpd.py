import numpy as np

from sys_with_acpr.pa import apply_selected_pa, normalize_rms, power_db


EPS = 1e-30


# Contador de operaciones

OPS = {"complex_mul": 0,"complex_add": 0,"real_add": 0,"abs": 0,"pow": 0,"div": 0,}

def reset_ops():
    """Reinicia el contador experimental de operaciones."""
    for k in OPS:
        OPS[k] = 0

def get_ops():
    """Devuelve una copia del contador experimental de operaciones."""
    return OPS.copy()

def total_ops(ops: dict | None = None):
    """Suma todas las operaciones contadas."""
    src = OPS if ops is None else ops
    return int(sum(src.values()))

def _count_basis_eval(n_terms: int):
    """Costo aproximado de evaluar términos x|x|^(p-1)."""
    n_terms = int(n_terms)
    OPS["abs"] += n_terms
    OPS["pow"] += n_terms
    OPS["complex_mul"] += n_terms


def _count_complex_dot(n_terms: int):
    """Costo aproximado de un producto interno complejo de longitud n_terms."""
    n_terms = int(n_terms)
    OPS["complex_mul"] += n_terms
    OPS["complex_add"] += max(n_terms - 1, 0)

def _count_complex_axpy(n_terms: int):
    """Costo aproximado de y <- y + a*x para vectores complejos."""
    n_terms = int(n_terms)
    OPS["complex_mul"] += n_terms
    OPS["complex_add"] += n_terms

def _count_nlms_update(n_terms: int):
    """Costo aproximado de un update NLMS complejo."""
    n_terms = int(n_terms)
    # vdot(phi, phi)
    OPS["complex_mul"] += n_terms
    OPS["real_add"] += max(n_terms - 1, 0)
    # e * conj(phi) / norm y suma sobre coeficientes
    OPS["complex_mul"] += n_terms
    OPS["div"] += n_terms
    OPS["complex_add"] += n_terms

# Polinomio con memoria

def _odd_orders(Pmax: int):
    """Órdenes impares para LS/ILA: 1, 3, 5, ..."""
    if Pmax < 1:
        raise ValueError("Pmax debe ser >= 1")
    return list(range(1, Pmax + 1, 2))

def _orders_from_coeffs_size(Pmax: int, M: int, n_coeffs: int):
    """Permite aplicar LS con impares y GD-DLA con todos los órdenes."""
    odd = _odd_orders(Pmax)
    full = _odd_orders(Pmax)

    if n_coeffs == M * len(full):
        return full
    if n_coeffs == M * len(odd):
        return odd

    raise ValueError(
        f"Coeficientes inválidos: llegó {n_coeffs}. "
        f"Esperaba {M * len(full)} para todos los órdenes o {M * len(odd)} para impares."
    )


def build_memory_poly_regressor(x: np.ndarray, Pmax: int, M: int):
    """
    Matriz Phi para Memory Polynomial impar:
    y[n] = sum_q sum_p c[p,q] x[n-q] |x[n-q]|^(p-1)
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if M < 1:
        raise ValueError("M debe ser >= 1")

    orders = _odd_orders(Pmax)
    N = len(x)
    N_eff = N - (M - 1)
    if N_eff <= 0:
        raise ValueError("La señal es demasiado corta para esa memoria")

    Phi = np.zeros((N_eff, M * len(orders)), dtype=np.complex128)

    col = 0
    for q in range(M):
        xq = x[(M - 1 - q):(M - 1 - q) + N_eff]
        ax = np.abs(xq)
        _count_basis_eval(N_eff * len(orders))
        for p in orders:
            Phi[:, col] = xq * ax ** (p - 1)
            col += 1

    valid = slice(M - 1, M - 1 + N_eff)
    return Phi, valid, orders


def apply_memory_poly(x: np.ndarray, coeffs: np.ndarray, Pmax: int, M: int):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    coeffs = np.asarray(coeffs, dtype=np.complex128).reshape(-1)
    orders = _orders_from_coeffs_size(Pmax, M, coeffs.size)
    y = np.zeros_like(x)
    col = 0
    for q in range(M):
        xq = np.zeros_like(x)
        xq[q:] = x[:len(x) - q]
        ax = np.abs(xq)

        for p in orders:
            _count_basis_eval(len(x))
            OPS["complex_mul"] += len(x)
            OPS["complex_add"] += len(x)
            y += coeffs[col] * xq * ax ** (p - 1)
            col += 1

    return y


def _basis_from_history(hist: np.ndarray, orders: list[int], M: int):
    """Vector de regresores para una historia [x[n], x[n-1], ...]."""
    phi = np.zeros(M * len(orders), dtype=np.complex128)
    col = 0

    for q in range(M):
        xq = hist[q]
        ax = abs(xq)
        for p in orders:
            phi[col] = xq * ax ** (p - 1)
            col += 1

    _count_basis_eval(len(phi))
    return phi


def _basis_column_power(x: np.ndarray, orders: list[int], M: int):
    """
    Potencia media por columna de la base MP, calculada offline sobre el
    training set: P_i = E[|phi_i[n]|^2].

    Los valores 1/P_i son las constantes del precondicionador diagonal
    (power-normalized LMS). Igualan la escala entre el término lineal (|x|)
    y los de orden alto (|x|^7), que es lo que un mu escalar no puede hacer.
    En FPGA son coeficientes precargados -> sin división en tiempo real.
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    N = len(x)
    powers = []
    for q in range(M):
        xq = np.zeros(N, dtype=np.complex128)
        xq[q:] = x[:N - q]
        ax = np.abs(xq)
        for p in orders:
            col = xq * ax ** (p - 1)
            powers.append(float(np.mean(np.abs(col) ** 2)))
    return np.asarray(powers, dtype=float)


def _normalize_linear_term(coeffs: np.ndarray):
    coeffs = np.asarray(coeffs, dtype=np.complex128).copy()
    if coeffs.size > 0 and abs(coeffs[0]) > EPS:
        coeffs /= coeffs[0]
    return coeffs


def _signal_report(name: str, x: np.ndarray):
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    return {
        f"{name}_P_dB": power_db(x),
        f"{name}_rms": float(np.sqrt(np.mean(np.abs(x) ** 2) + EPS)),
        f"{name}_peak": float(np.max(np.abs(x)) if len(x) else 0.0),
        f"{name}_papr_dB": float(10.0 * np.log10((np.max(np.abs(x)) ** 2 + EPS) / (np.mean(np.abs(x) ** 2) + EPS))),
    }


# Método LS, ILA
def train_dpd_ls(
    x_train: np.ndarray,
    *,
    pa_mode: str,
    pa_nl: str | None = None,
    use_memory: bool | None = None,
    bo_db: float,
    fs_in: float,
    fs_pa: float | None = None,
    R0: float = 50.0,
    Pmax: int = 5,
    M: int = 3,
    rcond: float | None = 1e-10,
    normalize_gain: bool = True,
    sat_amp: float = 1.0,
    p_soft: float = 2.0,
    **ignored_kwargs,
):
    """
    ILA clásico:
        x_train -> PA limpio -> y_train
        x_train ~= MP(y_train) b
    """
    x_train = np.asarray(x_train, dtype=np.complex128).reshape(-1)
    x_train, gx = normalize_rms(x_train, target_rms=1.0)

    y_train, pa_info, _ = apply_selected_pa(x_train,pa_mode=pa_mode,pa_nl=pa_nl,use_memory=use_memory,bo_db=bo_db,R0=R0,fs_in=fs_in,
                                            fs_pa=fs_pa,sat_amp=sat_amp,p_soft=p_soft,normalize_input=True,normalize_output=True,)

    Phi, valid, orders = build_memory_poly_regressor(y_train, Pmax=Pmax, M=M)

    # Para una matriz Phi de tamaño N_eff x K, el costo dominante de una
    # resolución por QR/SVD escala como O(N_eff*K^2 + K^3).
    N_eff, K = Phi.shape
    OPS["complex_mul"] += int(N_eff * K * K + K ** 3)
    OPS["complex_add"] += int(N_eff * K * K)

    coeffs, *_ = np.linalg.lstsq(Phi, x_train[valid], rcond=rcond)

    if normalize_gain:
        coeffs = _normalize_linear_term(coeffs)

    u_test = apply_memory_poly(x_train, coeffs, Pmax=Pmax, M=M)

    info = {"method": "LS-ILA","orders": orders,"M": M,"Pmax": Pmax,"x_norm_gain": gx,"pa_info_train": pa_info,
            **_signal_report("x_train", x_train),**_signal_report("y_train", y_train),**_signal_report("u_test", u_test),}

    if ignored_kwargs:
        info["ignored_kwargs_ls"] = sorted(ignored_kwargs.keys())

    return {
        "dpd_coeffs": coeffs,
        "orders": orders,
        "M": M,
        "Pmax": Pmax,
        "init_method": "ls",
        "method": "LS",
        "info": info,
    }


# Método GD, DLA con el gradiente de Wirtinger

def _train_pa_model_gd(
    x: np.ndarray,
    y: np.ndarray,
    *,
    Pmax: int,
    M: int,
    mu: float,
    n_epochs: int = 5,
    N_train_max: int | None = None,
    use_nlms: bool = True,
    precond_delta: float = 1e-3,
    divergence_factor: float = 1e3,
    h_ganancia_max: float | None = None,
):
    """
    Identificación forward del PA por GD complejo:

        y_hat[n] = h^T phi_x[n]
        e[n]     = y[n] - y_hat[n]
        h        = h + mu e[n] phi_x^*[n]

    Con use_nlms=True el paso se normaliza en dos niveles, igual que en el
    lazo del DPD, y mu queda adimensional con rango estable (0, 2).

    h_ganancia_max (sin NLMS): si en una muestra el paso efectivo
    mu * sum_i |phi_i|^2 / P_i pasa de este valor, se lo achica hasta ese
    valor, y debajo queda el de siempre. Es el h_ganancia_max del C de
    sw/dpd_train. None lo desactiva (ver robustez_arreglos.py).
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)

    N = min(len(x), len(y), N_train_max or len(x))
    x = x[:N]
    y = y[:N]

    orders = _odd_orders(Pmax)
    h = np.zeros(M * len(orders), dtype=np.complex128)
    # Arranca en la ganancia lineal del PA y no en 1. y va sin normalizar, así
    # que su ganancia baja con el back-off, y lo que queda del arranque en las
    # direcciones que la señal casi no excita (convergen muy lento) pesa cada
    # vez más frente a y. Con h0 = 1 el modelo empeora con el back-off y el
    # DPD empeora arriba de +7,5 dB.
    h[0] = np.vdot(x, y) / (np.vdot(x, x) + EPS)

    # Precondicionador diagonal estático (mismo principio que en el DPD):
    # paso por coeficiente mu_i = mu / P_i. Sin esto, identificar h con órdenes
    # altos (|x|^5..|x|^7) por LMS escalar es mal condicionado y h sale sesgado,
    # lo que después contamina el gradiente Wirtinger del DPD.
    inv_precond = 1.0 / (_basis_column_power(x, orders, M) + float(precond_delta))

    x_hist = np.zeros(M, dtype=np.complex128)
    mse_hist = []
    h_best = h.copy()
    mse_best = np.inf

    for _ in range(int(n_epochs)):
        x_hist[:] = 0.0
        mse_acc = 0.0

        for n in range(N):
            x_hist[1:] = x_hist[:-1]
            x_hist[0] = x[n]

            phi = _basis_from_history(x_hist, orders, M)
            _count_complex_dot(len(phi))
            y_hat = np.dot(h, phi)
            e = y[n] - y_hat

            if not np.isfinite(e) or not np.all(np.isfinite(phi)):
                mse_acc = np.inf
                break

            # El paso se normaliza en dos niveles, como en el lazo del DPD:
            #  (1) 1/P_i, fijo, iguala la escala entre el término lineal y los de
            #      orden alto.
            #  (2) NLMS por muestra: con orden 7 el pico de |x|^6 x está tres
            #      órdenes por encima de la media, y sin dividir por la energía
            #      instantánea la ganancia de lazo en los picos pasa de 2 y el
            #      lazo diverge. Dividiendo vale mu en toda muestra.
            _count_nlms_update(len(phi))
            if use_nlms:
                OPS["complex_mul"] += len(phi)
                OPS["real_add"] += max(len(phi) - 1, 0)
                OPS["div"] += 1
                reg_energy = float(np.sum((np.abs(phi) ** 2) * inv_precond))
                h += (float(mu) * e * np.conj(phi) * inv_precond
                      / (reg_energy + float(precond_delta)))
            elif h_ganancia_max is not None:
                g = float(mu) * float(np.sum((np.abs(phi) ** 2) * inv_precond))
                paso = float(mu) * float(h_ganancia_max) / g if g > h_ganancia_max else float(mu)
                h += paso * e * np.conj(phi) * inv_precond
            else:
                h += float(mu) * e * np.conj(phi) * inv_precond

            if not np.all(np.isfinite(h)):
                mse_acc = np.inf
                break

            mse_acc += abs(e) ** 2

        mse = float(mse_acc / max(N, 1))
        mse_hist.append(mse)

        if np.isfinite(mse) and mse < mse_best:
            mse_best = mse
            h_best = h.copy()

        if not np.isfinite(mse):
            h = h_best.copy()
            break

        # El lazo puede divergir sin llegar a inf: se midió un MSE de 4.8e40,
        # que es finito, así que la guarda anterior no cortaba y h_best quedaba
        # en la primera época, o sea el modelo casi sin entrenar. Cortar cuando
        # el MSE se despega del mejor deja el aviso en mse_hist.
        if mse > float(divergence_factor) * mse_best:
            h = h_best.copy()
            break

    return h_best, orders, np.asarray(mse_hist)


def _pa_wirtinger_terms_from_history(
    u_hist: np.ndarray,
    h: np.ndarray,
    orders: list[int],
    M: int,
):
    """
    Términos internos del gradiente DLA
    Modelo PA:
        d_hat[n] = sum_r sum_p h[p,r] u[n-r] |u[n-r]|^(p-1)

    Definiendo k = p - 1:
        A_r = sum_k k h_kr |u|^(k-2) u^2
        B_r = sum_k (k+2) h_kr^* |u|^k

    Estos términos propagan el error desde el modelo del PA hacia w.
    """
    A = np.zeros(M, dtype=np.complex128)
    B = np.zeros(M, dtype=np.complex128)

    col = 0
    for r in range(M):
        ur = u_hist[r]
        aur = abs(ur)

        for p in orders:
            k = p - 1
            hkr = h[col]

            OPS["abs"] += 1
            OPS["pow"] += 2
            OPS["complex_mul"] += 4
            OPS["complex_add"] += 2

            if k != 0 and aur > EPS:
                A[r] += k * hkr * aur ** (k - 2) * ur ** 2

            B[r] += (k + 2) * np.conj(hkr) * aur ** k
            col += 1

    return A, B


def _train_dpd_gd_direct_dla(
    x: np.ndarray,
    h: np.ndarray,
    *,
    Pmax: int,
    M: int,
    mu: float,
    n_epochs: int = 10,
    N_train_max: int | None = None,
    target_gain: complex | None = None,
    grad_clip: float | None = 1e3,
    freeze_w0: bool = True,
    use_nlms: bool = True,
    w_amp_clip: float | None = None,
    w_reg_lambda: float = 0.0,
    w_reg_pmin: int = 1,
    w_memory_lambda: float = 0.0,
    precond_delta: float = 1e-3,
):
    """
    Entrenamiento DLA-GD del predistorsionador con salvaguardas
    para evitar la solución degenerada del atenuador (w[0] < 1):

        1) target_gain = h[0] por defecto -> el camino lineal queda
           trivialmente satisfecho con w[0] = 1, w_resto = 0.

        2) freeze_w0=True -> proyecta el gradiente a grad[0] = 0 y reasigna
           w[0] = 1 después de cada paso. El optimizador NO PUEDE elegir
           atenuar; sólo puede mover los términos no lineales.

        3) use_nlms=True -> normaliza el gradiente por ||phi_x||^2 (NLMS).
           Hace mu_w mucho menos sensible y estabiliza el entrenamiento.

    Estructura del lazo (Wirtinger):
        x[n] -> DPD(w) -> u[n] -> PA_model(h) -> d_hat[n]
        e[n] = d[n] - d_hat[n]
        w   <- w - mu * grad_wirtinger
    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    h = np.asarray(h, dtype=np.complex128).reshape(-1)

    N = min(len(x), N_train_max or len(x))
    x = x[:N]

    orders = _odd_orders(Pmax)
    n_coeffs = M * len(orders)
    if h.size != n_coeffs:
        raise ValueError(f"h inválido: esperaba {n_coeffs}, llegó {h.size}")

    # w arranca en identidad (w[0]=1, resto 0).
    w = np.zeros(n_coeffs, dtype=np.complex128)
    w[0] = 1.0 + 0.0j

    # Precondicionador diagonal: 1/P_i por coeficiente (constantes offline).
    inv_precond = np.ones(n_coeffs, dtype=float)
    if use_nlms:
        inv_precond = 1.0 / (_basis_column_power(x, orders, M) + float(precond_delta))

    if target_gain is None:
        target_gain = h[0] if h.size > 0 else (1.0 + 0.0j)

    # Límites por orden: |w[q,p]| <= w_amp_clip / x_peak^(p-1)
    # Garantiza que el término de orden p no supere w_amp_clip veces el
    # término lineal en el pico de la señal de entrenamiento.
    # Para p=7 con x_peak=3.67: max = 1.0 / 3.67^6 ≈ 4e-4, evitando que
    # el 7mo orden invierta la señal en los picos de la OFDM.
    if w_amp_clip is not None:
        # Piso en 4.0 (≈ PAPR 12 dB) para que los bounds sean válidos incluso
        # cuando N_train_max es pequeño y el pico real del training set no llega
        # al pico teórico de la señal OFDM de evaluación.
        x_peak_est = max(float(np.max(np.abs(x))), 4.0)
        w_max_per_order = {p: float(w_amp_clip) / (x_peak_est ** (p - 1) + EPS)
                          for p in orders}

    x_hist = np.zeros(M, dtype=np.complex128)
    u_hist = np.zeros(M, dtype=np.complex128)
    phi_x_hist = np.zeros((M, n_coeffs), dtype=np.complex128)

    mse_hist = []
    # Se devuelve el w de la última época. El error de cada época se mide a
    # través del modelo del PA, un polinomio que no satura, y en la zona
    # comprimida explota en los picos que expande el DPD, así que no sirve
    # para elegir época.
    w_ultimo = w.copy()

    for _ in range(int(n_epochs)):
        x_hist[:] = 0.0
        u_hist[:] = 0.0
        phi_x_hist[:] = 0.0
        mse_acc = 0.0

        for n in range(N):
            x_hist[1:] = x_hist[:-1]
            x_hist[0] = x[n]

            phi_x = _basis_from_history(x_hist, orders, M)
            _count_complex_dot(len(phi_x))
            u_n = np.dot(w, phi_x)

            if not np.isfinite(u_n):
                mse_acc = np.inf
                break

            u_hist[1:] = u_hist[:-1]
            u_hist[0] = u_n

            phi_x_hist[1:] = phi_x_hist[:-1]
            phi_x_hist[0] = phi_x

            phi_u = _basis_from_history(u_hist, orders, M)
            _count_complex_dot(len(phi_u))
            d_hat = np.dot(h, phi_u)
            OPS["complex_mul"] += 1
            d = target_gain * x[n]
            e = d - d_hat

            A, B = _pa_wirtinger_terms_from_history(u_hist, h, orders, M)

            if (not np.all(np.isfinite(A)) or not np.all(np.isfinite(B)) or not np.isfinite(e)):
                mse_acc = np.inf
                break

            grad = np.zeros(n_coeffs, dtype=np.complex128)
            for r in range(M):
                _count_complex_axpy(n_coeffs)
                OPS["complex_mul"] += 4 * n_coeffs
                OPS["complex_add"] += 2 * n_coeffs
                grad += ( -np.conj(e) * A[r] * np.conj(phi_x_hist[r]) -e * B[r] * np.conj(phi_x_hist[r]))

            if not np.all(np.isfinite(grad)):
                mse_acc = np.inf
                break

            # El paso se normaliza en dos niveles:
            #  (1) 1/P_i, fijo, iguala la escala entre el término lineal y los de
            #      orden alto.
            #  (2) NLMS sobre el regresor del gradiente, que los términos A y B
            #      del PA (~|u|^k) agrandan en los picos. Dividir por
            #      (|A|^2+|B|^2)*||phi_x||^2 evita que un pico saque a w del óptimo.
            if use_nlms:
                reg_energy = 0.0
                for r in range(M):
                    OPS["complex_mul"] += n_coeffs
                    OPS["real_add"] += max(n_coeffs - 1, 0)
                    reg_energy += (abs(A[r]) ** 2 + abs(B[r]) ** 2) * float(
                        np.vdot(phi_x_hist[r], phi_x_hist[r]).real)
                OPS["div"] += 1
                grad = grad * inv_precond / (reg_energy + float(precond_delta))

            # Proyección: el coef. lineal queda fijo en 1+0j -> el optimizador
            # no puede atenuar la señal globalmente, sólo puede generar
            # distorsión polinómica no lineal.
            if freeze_w0:
                grad[0] = 0.0 + 0.0j

            if grad_clip is not None:
                OPS["complex_mul"] += n_coeffs
                OPS["real_add"] += max(n_coeffs - 1, 0)
                grad_norm = np.linalg.norm(grad)
                if grad_norm > float(grad_clip):
                    grad *= float(grad_clip) / (grad_norm + EPS)

            OPS["complex_mul"] += n_coeffs
            OPS["complex_add"] += n_coeffs
            w -= float(mu) * grad

            # L2 leakage selectiva: w[q,p] *= (1-lambda) para p >= w_reg_pmin.
            # Previene deriva hacia los bounds en órdenes altos (p>=5) cuando el
            # gradiente es pequeño (PA casi lineal a alto BO), sin afectar p=3
            # que es fisicamente necesario para compensar la no linealidad del PA.
            if w_reg_lambda > 0.0 or w_memory_lambda > 0.0:
                lk_order  = 1.0 - float(w_reg_lambda)
                lk_memory = 1.0 - float(w_memory_lambda)
                _col = 0
                for _q in range(M):
                    for _p in orders:
                        if _p >= w_reg_pmin and w_reg_lambda > 0.0:
                            w[_col] *= lk_order
                        if _q >= 1 and w_memory_lambda > 0.0:
                            w[_col] *= lk_memory
                        _col += 1

            if freeze_w0:
                w[0] = 1.0 + 0.0j

            # Proyección por orden: evita que |w[q,p]| > w_amp_clip / x_peak^(p-1).
            # Sin esto, el término de orden 7 en el pico (|x|≈3.7) vale
            # w[7] * |x|^6 ≈ 2450 * w[7], invirtiendo la señal si w[7] > 4e-4.
            if w_amp_clip is not None:
                col_p = 0
                for _q in range(M):
                    for _p in orders:
                        if _p > 1:
                            wabs = abs(w[col_p])
                            if wabs > w_max_per_order[_p]:
                                w[col_p] *= w_max_per_order[_p] / (wabs + EPS)
                        col_p += 1

            if not np.all(np.isfinite(w)):
                mse_acc = np.inf
                break

            mse_acc += abs(e) ** 2

        mse = float(mse_acc / max(N, 1))
        mse_hist.append(mse)

        if not np.isfinite(mse):
            w = w_ultimo.copy()
            break
        w_ultimo = w.copy()

    return w, orders, np.asarray(mse_hist), target_gain


def train_dpd_gd(
    x_train: np.ndarray,
    *,
    pa_mode: str,
    pa_nl: str | None = None,
    use_memory: bool | None = None,
    bo_db: float,
    fs_in: float,
    fs_pa: float | None = None,
    R0: float = 50.0,
    Pmax: int = 5,
    M: int = 3,
    mu_h: float = 0.001,
    mu_w: float = 0.1,
    ep_pa: int = 20,
    ep_dpd: int = 5,
    N_train_max: int | None = None,
    normalize_gain: bool = False,
    # u_clip removed
    grad_clip: float | None = 1e3,
    sat_amp: float = 1.0,
    p_soft: float = 2.0,
    freeze_w0: bool = True,
    use_nlms_h: bool = False,
    use_nlms_w: bool = True,
    w_amp_clip: float | None = None,
    w_reg_lambda: float = 0.0,
    w_reg_pmin: int = 1,
    w_memory_lambda: float = 0.0,
    mu_h_respaldo: float | None = 0.05,
    h_ganancia_max: float | None = None,
    **ignored_kwargs,
):
    """
    DPD-GD con estructura DLA bibliográfica + salvaguardas que evitan
    la solución degenerada del atenuador (w[0]<1):

    Modelo del PA, el mismo esquema que el ARM del SoC: LMS
    precondicionado con mu_h = 0.001 y sin NLMS por muestra, que es el que da
    el mejor DPD en el peor caso (BO = -9 dB), y respaldo con NLMS y
    `mu_h_respaldo` si la guarda de divergencia corta el lazo. `None`
    desactiva el respaldo.

    h_ganancia_max limita el paso efectivo del modelo del PA en cada muestra
    (ver _train_pa_model_gd). La tesis usa 1, con 16000 muestras: sin límite
    el modelo diverge con algunos registros y el DPD depende mucho del
    registro (Robustez del entrenamiento).

    Pasos:
        1) x_train se normaliza a RMS=1.
        2) x_train pasa por el PA limpio sin normalizar la salida -> y_train.
        3) Se identifica el modelo h del PA (mismo orden Pmax/M que el DPD).
        4) Se entrena w por GD-Wirtinger con NLMS y w[0]=1 fijo, desde identidad.
    """
    x_train = np.asarray(x_train, dtype=np.complex128).reshape(-1)
    x_train, gx = normalize_rms(x_train, target_rms=1.0)

    y_train, pa_info, _ = apply_selected_pa(x_train,pa_mode=pa_mode,pa_nl=pa_nl,use_memory=use_memory,bo_db=bo_db,R0=R0,fs_in=fs_in,
                                            fs_pa=fs_pa,sat_amp=sat_amp,p_soft=p_soft,normalize_input=True,normalize_output=False,)

    target_gain_raw = np.vdot(x_train, y_train) / (np.vdot(x_train, x_train) + EPS)

    # El modelo h del PA usa el mismo orden que el DPD (Pmax/M).
    h_model, orders_h, mse_pa = _train_pa_model_gd(x_train, y_train, Pmax=Pmax, M=M,
                                                   mu=mu_h, n_epochs=ep_pa, N_train_max=N_train_max,
                                                   use_nlms=use_nlms_h, h_ganancia_max=h_ganancia_max)
    pa_respaldo = False
    if mu_h_respaldo is not None and len(mse_pa) < ep_pa:
        h_model, orders_h, mse_pa = _train_pa_model_gd(x_train, y_train, Pmax=Pmax, M=M,
                                                       mu=mu_h_respaldo, n_epochs=ep_pa,
                                                       N_train_max=N_train_max, use_nlms=True)
        pa_respaldo = True

    w, orders_w, mse_dpd, target_gain = _train_dpd_gd_direct_dla(x_train, h_model, Pmax=Pmax, M=M, mu=mu_w,
                                                                  n_epochs=ep_dpd, N_train_max=N_train_max,
                                                                  target_gain=None, grad_clip=grad_clip,
                                                                  freeze_w0=freeze_w0, use_nlms=use_nlms_w,
                                                                  w_amp_clip=w_amp_clip, w_reg_lambda=w_reg_lambda,
                                                                  w_reg_pmin=w_reg_pmin, w_memory_lambda=w_memory_lambda)

    if normalize_gain:
        w = _normalize_linear_term(w)

    u_test = apply_memory_poly(x_train, w, Pmax=Pmax, M=M)
    y_test, pa_info_test, _ = apply_selected_pa(u_test,pa_mode=pa_mode,pa_nl=pa_nl,use_memory=use_memory,bo_db=bo_db,R0=R0,fs_in=fs_in,
                                                fs_pa=fs_pa,sat_amp=sat_amp,p_soft=p_soft,normalize_input=False,normalize_output=True,)
    direct_mse = float(np.mean(np.abs(x_train - y_test) ** 2))

    alpha_eval = np.vdot(y_test, x_train) / (np.vdot(y_test, y_test) + EPS)
    direct_mse_gain_aligned = float(np.mean(np.abs(x_train - alpha_eval * y_test) ** 2))
    direct_nmse_gain_aligned = float(
        direct_mse_gain_aligned / (np.mean(np.abs(x_train) ** 2) + EPS)
    )

    info = {
            "method": "DLA-GD-Wirtinger","orders": orders_w,"orders_h": orders_h,"M": M,"Pmax": Pmax,"x_norm_gain": gx,
            "pa_info_train": pa_info,"pa_info_test": pa_info_test,"direct_mse_real_pa": direct_mse,"target_gain": target_gain,
            "target_gain_raw": target_gain_raw,"direct_mse_gain_aligned": direct_mse_gain_aligned,
            "direct_nmse_gain_aligned": direct_nmse_gain_aligned,"mse_pa": mse_pa,"mse_dpd": mse_dpd,
            **_signal_report("x_train", x_train),**_signal_report("y_train", y_train),**_signal_report("u_test", u_test),
            }

    if ignored_kwargs:
        info["ignored_kwargs_gd"] = sorted(ignored_kwargs.keys())

    return {
        "dpd_coeffs": w,
        "h_model": h_model,
        "target_gain": target_gain,
        "target_gain_raw": target_gain_raw,
        "orders": orders_w,
        "orders_h": orders_h,
        "M": M,
        "Pmax": Pmax,
        "mse_pa": mse_pa,
        "mse_dpd": mse_dpd,
        "pa_respaldo": pa_respaldo,
        "direct_mse_real_pa": direct_mse,
        "direct_mse_gain_aligned": direct_mse_gain_aligned,
        "direct_nmse_gain_aligned": direct_nmse_gain_aligned,
        "init_method": "gd_dla",
        "method": "DLA-GD-Wirtinger",
        "info": info,
    }


# Aplicación DPD -> PA

def apply_dpd_and_pa(
    x: np.ndarray,
    dpd_coeffs: np.ndarray,
    *,
    pa_mode: str,
    pa_nl: str | None = None,
    use_memory: bool | None = None,
    bo_db: float,
    fs_in: float,
    fs_pa: float | None = None,
    R0: float = 50.0,
    Pmax: int = 5,
    M: int = 3,
    sat_amp: float = 1.0,
    p_soft: float = 2.0,
    normalize_dpd_input: bool = True,
    normalize_dpd_output: bool = False,
    normalize_pa_input: bool = False,
    **ignored_kwargs,
):
    """
    Cadena limpia:
        x -> normalización opcional -> DPD -> normalización opcional -> PA limpio

    El PA controla:
    - Pin_PA según BO
    - Pout de bloque normalizada para RX/EVM/BER/ACPR

    """
    x = np.asarray(x, dtype=np.complex128).reshape(-1)

    if normalize_dpd_input:
        x_dpd, gx = normalize_rms(x, target_rms=1.0)
    else:
        x_dpd = x.copy()
        gx = 1.0

    u = apply_memory_poly(x_dpd, dpd_coeffs, Pmax=Pmax, M=M)

    if normalize_dpd_output:
        u_pa, gu = normalize_rms(u, target_rms=1.0)
    else:
        u_pa = u.copy()
        gu = 1.0

    y, pa_info, _ = apply_selected_pa(u_pa,pa_mode=pa_mode,pa_nl=pa_nl,use_memory=use_memory,bo_db=bo_db,R0=R0,fs_in=fs_in,fs_pa=fs_pa,
                                      sat_amp=sat_amp,p_soft=p_soft,normalize_input=normalize_pa_input,normalize_output=True,)

    pa_info["dpd_scale_x_norm"] = float(gx)
    pa_info["dpd_scale_u_norm"] = float(gu)
    pa_info["dpd_normalize_pa_input"] = bool(normalize_pa_input)
    pa_info.update(_signal_report("x_dpd_in", x_dpd))
    pa_info.update(_signal_report("u_dpd_raw", u))
    pa_info.update(_signal_report("u_pa_in", u_pa))
    pa_info.update(_signal_report("y_pa_out", y))

    if ignored_kwargs:
        pa_info["ignored_kwargs_dpd"] = sorted(ignored_kwargs.keys())

    return u_pa, y, pa_info
