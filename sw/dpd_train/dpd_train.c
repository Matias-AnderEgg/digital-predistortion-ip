/*
 * Ver dpd_train.h. Los nombres y el orden de las operaciones siguen a
 * fixed_point/dpd.py para que se puedan comparar linea por linea: la
 * verificacion es contra ese modelo corriendo en flotante.
 */
#include "dpd_train.h"

#include <math.h>
#include <string.h>

#define EPS 1e-30

static const int ORDENES[DPD_NORD] = {1, 3, 5, 7};

void dpd_train_cfg_default(dpd_train_cfg *cfg)
{
    cfg->G                 = 8.0;
    cfg->mu_h              = 0.001;
    cfg->ep_pa             = 40;
    cfg->use_nlms_h        = 0;
    cfg->divergence_factor = 1e3;
    cfg->quantize_h        = 0;
    cfg->mu_w              = 0.5;
    cfg->ep_dpd            = 5;
    cfg->use_nlms_w        = 1;
    cfg->freeze_w0         = 1;
    cfg->grad_clip         = 1e3;
    cfg->w_reg_lambda      = 1e-4;
    cfg->w_reg_pmin        = 3;
    cfg->precond_delta     = 1e-3;
    cfg->mu_h_respaldo     = 0.05;
    cfg->warm_start        = 0;
    cfg->h_ganancia_max    = 1.0;
    cfg->h_umbral_pico     = 0.0;
}

/* G^(p-1) por coeficiente: el corrimiento por orden de la convencion
 * normalizada. */
static void escala_por_orden(double G, double s[DPD_NCOEF])
{
    int col = 0;
    for (int q = 0; q < DPD_M; q++)
        for (int k = 0; k < DPD_NORD; k++)
            s[col++] = pow(G, ORDENES[k] - 1);
}

/* Base del Memory Polynomial para una historia [x[n], x[n-1], x[n-2]]. */
static void base(const double complex hist[DPD_M], double complex phi[DPD_NCOEF])
{
    int col = 0;
    for (int q = 0; q < DPD_M; q++) {
        double complex xq = hist[q];
        double ax = cabs(xq);
        for (int k = 0; k < DPD_NORD; k++)
            phi[col++] = xq * pow(ax, ORDENES[k] - 1);
    }
}

static void empujar(double complex hist[DPD_M], double complex v)
{
    for (int r = DPD_M - 1; r > 0; r--)
        hist[r] = hist[r - 1];
    hist[0] = v;
}

/*
 * Potencia media de cada columna de la base sobre todo el registro,
 * P_i = E[|phi_i|^2], ya en escala de phi_eff. Es el precondicionador
 * diagonal: iguala la escala entre el termino lineal y el de orden 7.
 * La senal entra ya dividida por G.
 */
static void potencia_columnas(const double complex *x, double xs, int n,
                              const double s[DPD_NCOEF], double P[DPD_NCOEF])
{
    double complex hist[DPD_M] = {0};
    double complex phi[DPD_NCOEF];

    memset(P, 0, DPD_NCOEF * sizeof P[0]);
    for (int i = 0; i < n; i++) {
        empujar(hist, x[i] * xs);
        base(hist, phi);
        for (int c = 0; c < DPD_NCOEF; c++) {
            double a = cabs(phi[c]);
            P[c] += a * a;
        }
    }
    for (int c = 0; c < DPD_NCOEF; c++)
        P[c] = (P[c] / n) * s[c] * s[c];
}

/* Redondeo a Q0.15 con saturacion, al par mas cercano como numpy. */
static double q15(double v)
{
    double c = nearbyint(v * 32768.0);
    if (c > 32767.0)  c = 32767.0;
    if (c < -32768.0) c = -32768.0;
    return c / 32768.0;
}

static double complex q15c(double complex v)
{
    return q15(creal(v)) + I * q15(cimag(v));
}

static int finito(double complex v)
{
    return isfinite(creal(v)) && isfinite(cimag(v));
}

static int todos_finitos(const double complex *v, int n)
{
    for (int i = 0; i < n; i++)
        if (!finito(v[i]))
            return 0;
    return 1;
}

/* precond_delta en la escala de la potencia de las columnas. Esa potencia
 * se mide sobre la senal dividida por G y queda G^2 mas chica que la fisica,
 * asi que el delta se divide por G^2 para que el entrenamiento no dependa de
 * G. Sin esto, con G = 8 el delta pesaba un 6 % de la potencia del termino
 * lineal y h salia hasta 180 LSB distinto que con G = 1. */
static double delta_columnas(const dpd_train_cfg *cfg)
{
    return cfg->precond_delta / (cfg->G * cfg->G);
}

static int pa_model(const double complex *x, const double complex *y, int n,
                    const dpd_train_cfg *cfg,
                    double complex h[DPD_NCOEF], double *mse_hist)
{
    const double xs = 1.0 / cfg->G;     /* G es potencia de 2: exacto */
    double s[DPD_NCOEF], inv_precond[DPD_NCOEF];
    double complex h_best[DPD_NCOEF];
    double mse_best = INFINITY;
    int ep;

    escala_por_orden(cfg->G, s);
    potencia_columnas(x, xs, n, s, inv_precond);
    for (int c = 0; c < DPD_NCOEF; c++)
        inv_precond[c] = 1.0 / (inv_precond[c] + delta_columnas(cfg));

    if (!cfg->warm_start) {
        /* Arranca en la ganancia lineal <x, y> / <x, x> y no en 1. y viene
         * sin normalizar y su ganancia baja con el back-off. Lo que queda
         * del arranque en las direcciones que la senal casi no excita, que
         * convergen muy lento, pesa cada vez mas frente a y: con h0 = 1 el
         * modelo empeoraba con el back-off. */
        double complex xy = 0.0;
        double xx = 0.0;
        for (int i = 0; i < n; i++) {
            xy += conj(x[i]) * y[i];
            xx += creal(x[i] * conj(x[i]));
        }
        memset(h, 0, DPD_NCOEF * sizeof h[0]);
        h[0] = xy / (xx + 1e-30);
    }
    memcpy(h_best, h, sizeof h_best);

    double lim_pico = cfg->h_umbral_pico > 0.0 ? cfg->h_umbral_pico : INFINITY;

    for (ep = 0; ep < cfg->ep_pa; ep++) {
        double complex hist[DPD_M] = {0};
        double mse_acc = 0.0;

        for (int i = 0; i < n; i++) {
            double complex phi[DPD_NCOEF], phi_eff[DPD_NCOEF];
            double complex y_hat = 0.0, e;

            empujar(hist, x[i] * xs);
            base(hist, phi);
            for (int c = 0; c < DPD_NCOEF; c++) {
                phi_eff[c] = phi[c] * s[c];
                y_hat += (cfg->quantize_h ? q15c(h[c]) : h[c]) * phi_eff[c];
            }
            e = y[i] * xs - y_hat;

            if (!finito(e) || !todos_finitos(phi, DPD_NCOEF)) {
                mse_acc = INFINITY;
                break;
            }

            /*
             * Paso normalizado en dos niveles. El precondicionador usa la
             * potencia MEDIA de cada columna, y la columna de orden 7 tiene
             * picos tres ordenes por encima de su media: sin dividir ademas
             * por la energia instantanea del regresor, la ganancia de lazo en
             * los picos pasa de 2 y el lazo diverge.
             */
            double norm = 1.0;
            if (cfg->use_nlms_h || cfg->h_ganancia_max > 0.0) {
                double reg_energy = 0.0;
                for (int c = 0; c < DPD_NCOEF; c++) {
                    double a = cabs(phi_eff[c]);
                    reg_energy += a * a * inv_precond[c];
                }
                if (cfg->use_nlms_h)
                    norm = reg_energy + cfg->precond_delta;
                else if (cfg->mu_h * reg_energy > cfg->h_ganancia_max)
                    norm = cfg->mu_h * reg_energy / cfg->h_ganancia_max;
            }
            for (int r = 0; r < DPD_M; r++)  /* el pico entra por cualquier tap */
                if (cabs(hist[r]) > lim_pico * xs)
                    norm = INFINITY;    /* paso cero: no actualiza */
            for (int c = 0; c < DPD_NCOEF; c++) {
                double paso = cfg->mu_h * inv_precond[c] / norm;
                h[c] += e * conj(phi_eff[c]) * paso;
            }

            if (!todos_finitos(h, DPD_NCOEF)) {
                mse_acc = INFINITY;
                break;
            }
            double ae = cabs(e);
            mse_acc += ae * ae;
        }

        double mse = mse_acc / n;
        if (mse_hist)
            mse_hist[ep] = mse;
        if (isfinite(mse) && mse < mse_best) {
            mse_best = mse;
            memcpy(h_best, h, sizeof h_best);
        }
        /* Un lazo que diverge puede no llegar nunca a inf: se midio un MSE
         * de 4.8e40. Por eso se corta tambien cuando se despega del mejor. */
        if (!isfinite(mse) || mse > cfg->divergence_factor * mse_best) {
            ep++;
            break;
        }
    }

    for (int c = 0; c < DPD_NCOEF; c++)
        h[c] = cfg->quantize_h ? q15c(h_best[c]) : h_best[c];
    return ep;
}

/*
 * El esquema por defecto (mu_h = 0.001 sin NLMS) da el mejor DPD cuando
 * converge, pero con algunos registros diverge. El NLMS por muestra
 * converge siempre pero da un DPD peor, asi que queda de respaldo.
 */
int dpd_train_pa_model(const double complex *x, const double complex *y, int n,
                       const dpd_train_cfg *cfg,
                       double complex h[DPD_NCOEF], double *mse_hist)
{
    int ep = pa_model(x, y, n, cfg, h, mse_hist);
    if (ep < cfg->ep_pa && cfg->mu_h_respaldo > 0.0) {
        dpd_train_cfg r = *cfg;
        r.mu_h = cfg->mu_h_respaldo;
        r.use_nlms_h = 1;
        r.warm_start = 0;               /* desde cero, no desde lo que divergio */
        ep = -pa_model(x, y, n, &r, h, mse_hist);
    }
    return ep;
}

/*
 * Terminos que propagan el error desde el modelo del PA hacia w. Con
 * k = p - 1, y h_eff = h * G^(p-1):
 *     A_r = sum_k k h_kr |u|^(k-2) u^2
 *     B_r = sum_k (k+2) h_kr^* |u|^k
 * Valen el doble de la derivada de Wirtinger, que lleva un 1/2. El factor
 * es comun a los dos y queda absorbido en mu_w, igual que en el modelo.
 */
static void terminos_wirtinger(const double complex u_hist[DPD_M],
                               const double complex h[DPD_NCOEF],
                               const double s[DPD_NCOEF],
                               double complex A[DPD_M], double complex B[DPD_M])
{
    int col = 0;
    for (int r = 0; r < DPD_M; r++) {
        double complex ur = u_hist[r];
        double aur = cabs(ur);
        A[r] = 0.0;
        B[r] = 0.0;
        for (int j = 0; j < DPD_NORD; j++) {
            int k = ORDENES[j] - 1;
            double complex hkr = h[col] * s[col];
            if (k != 0 && aur > EPS)
                A[r] += k * hkr * pow(aur, k - 2) * (ur * ur);
            B[r] += (k + 2) * conj(hkr) * pow(aur, k);
            col++;
        }
    }
}

int dpd_train_dla(const double complex *x, int n, const double complex h[DPD_NCOEF],
                  const dpd_train_cfg *cfg,
                  double complex w[DPD_NCOEF], double *mse_hist)
{
    const double xs = 1.0 / cfg->G;
    double s[DPD_NCOEF], inv_precond[DPD_NCOEF];
    /* Ultimo w de una epoca terminada: si el lazo diverge se vuelve a el. El
     * error de cada epoca se mide a traves del modelo del PA, que no satura,
     * y no sirve para elegir epoca, asi que se devuelve la ultima. */
    double complex w_ultimo[DPD_NCOEF];
    /* El camino lineal queda satisfecho con w[0] = 1 y el resto en cero. */
    const double complex target_gain = h[0];
    int ep;

    escala_por_orden(cfg->G, s);
    for (int c = 0; c < DPD_NCOEF; c++)
        inv_precond[c] = 1.0;
    if (cfg->use_nlms_w) {
        potencia_columnas(x, xs, n, s, inv_precond);
        for (int c = 0; c < DPD_NCOEF; c++)
            inv_precond[c] = 1.0 / (inv_precond[c] + delta_columnas(cfg));
    }

    if (!cfg->warm_start) {
        memset(w, 0, DPD_NCOEF * sizeof w[0]);
        w[0] = 1.0;
    }
    memcpy(w_ultimo, w, sizeof w_ultimo);

    for (ep = 0; ep < cfg->ep_dpd; ep++) {
        double complex x_hist[DPD_M] = {0}, u_hist[DPD_M] = {0};
        double complex phi_x_hist[DPD_M][DPD_NCOEF] = {{0}};
        double mse_acc = 0.0;

        for (int i = 0; i < n; i++) {
            double complex phi_x[DPD_NCOEF], phi_u[DPD_NCOEF], grad[DPD_NCOEF];
            double complex A[DPD_M], B[DPD_M];
            double complex u_n = 0.0, d_hat = 0.0, d, e;

            empujar(x_hist, x[i] * xs);
            base(x_hist, phi_x);
            for (int c = 0; c < DPD_NCOEF; c++)
                u_n += w[c] * (phi_x[c] * s[c]);
            if (!finito(u_n)) {
                mse_acc = INFINITY;
                break;
            }
            empujar(u_hist, u_n);
            memmove(phi_x_hist[1], phi_x_hist[0], (DPD_M - 1) * sizeof phi_x_hist[0]);
            memcpy(phi_x_hist[0], phi_x, sizeof phi_x_hist[0]);

            base(u_hist, phi_u);
            for (int c = 0; c < DPD_NCOEF; c++)
                d_hat += h[c] * (phi_u[c] * s[c]);
            d = target_gain * (x[i] * xs);
            e = d - d_hat;

            terminos_wirtinger(u_hist, h, s, A, B);
            if (!todos_finitos(A, DPD_M) || !todos_finitos(B, DPD_M) || !finito(e)) {
                mse_acc = INFINITY;
                break;
            }

            for (int c = 0; c < DPD_NCOEF; c++) {
                double complex g = 0.0;
                for (int r = 0; r < DPD_M; r++) {
                    double complex cphi = conj(phi_x_hist[r][c]);
                    g += -conj(e) * A[r] * cphi - e * B[r] * cphi;
                }
                grad[c] = g * s[c];
            }

            /*
             * Normalizacion en dos niveles: el precondicionador iguala los
             * ordenes entre si y la division por la energia del regresor
             * real del gradiente (que pesa phi_x por A y B, ~|u|^k) evita
             * que un pico de la OFDM patee w lejos del optimo.
             */
            if (cfg->use_nlms_w) {
                double reg_energy = 0.0;
                for (int r = 0; r < DPD_M; r++) {
                    double e_phi = 0.0;
                    for (int c = 0; c < DPD_NCOEF; c++) {
                        double a = cabs(phi_x_hist[r][c] * s[c]);
                        e_phi += a * a;
                    }
                    double aA = cabs(A[r]), aB = cabs(B[r]);
                    reg_energy += (aA * aA + aB * aB) * e_phi;
                }
                /* Con la convencion normalizada el paso sale G^2 mas grande
                 * que el fisico para el mismo mu: se compensa aca. */
                reg_energy *= cfg->G * cfg->G;
                for (int c = 0; c < DPD_NCOEF; c++)
                    grad[c] = grad[c] * inv_precond[c] / (reg_energy + cfg->precond_delta);
            }

            if (cfg->freeze_w0)
                grad[0] = 0.0;

            if (cfg->grad_clip > 0.0) {
                double norma = 0.0;
                for (int c = 0; c < DPD_NCOEF; c++) {
                    double a = cabs(grad[c]);
                    norma += a * a;
                }
                norma = sqrt(norma);
                if (norma > cfg->grad_clip)
                    for (int c = 0; c < DPD_NCOEF; c++)
                        grad[c] *= cfg->grad_clip / (norma + EPS);
            }

            for (int c = 0; c < DPD_NCOEF; c++)
                w[c] -= cfg->mu_w * grad[c];

            /* Fuga sobre los ordenes altos: evita que deriven hacia valores
             * grandes cuando el gradiente es chico. */
            if (cfg->w_reg_lambda > 0.0) {
                int col = 0;
                for (int q = 0; q < DPD_M; q++)
                    for (int j = 0; j < DPD_NORD; j++, col++)
                        if (ORDENES[j] >= cfg->w_reg_pmin)
                            w[col] *= 1.0 - cfg->w_reg_lambda;
            }

            if (cfg->freeze_w0)
                w[0] = 1.0;

            if (!todos_finitos(w, DPD_NCOEF)) {
                mse_acc = INFINITY;
                break;
            }
            double ae = cabs(e);
            mse_acc += ae * ae;
        }

        double mse = mse_acc / n;
        if (mse_hist)
            mse_hist[ep] = mse;
        if (!isfinite(mse)) {
            memcpy(w, w_ultimo, sizeof w_ultimo);
            ep++;
            break;
        }
        memcpy(w_ultimo, w, sizeof w_ultimo);
    }

    return ep;
}

double dpd_rms_salida(const double complex *x, int n, const double complex w[DPD_NCOEF],
                      const dpd_train_cfg *cfg)
{
    const double xs = 1.0 / cfg->G;
    double s[DPD_NCOEF], p = 0.0;
    double complex hist[DPD_M] = {0};

    escala_por_orden(cfg->G, s);
    for (int i = 0; i < n; i++) {
        double complex phi[DPD_NCOEF], u = 0.0;
        empujar(hist, x[i] * xs);
        base(hist, phi);
        for (int c = 0; c < DPD_NCOEF; c++)
            u += w[c] * phi[c] * s[c];
        p += creal(u) * creal(u) + cimag(u) * cimag(u);
    }
    return cfg->G * sqrt(p / n);
}

static int16_t a_q15(double v, int *saturo)
{
    double c = nearbyint(v * 32768.0);  /* al par mas cercano, como numpy */
    if (c > 32767.0)  { (*saturo)++; return 32767; }
    if (c < -32768.0) { (*saturo)++; return -32768; }
    return (int16_t)c;
}

int dpd_coef_to_q15(const double complex w[DPD_NCOEF],
                    int16_t re[DPD_NCOEF], int16_t im[DPD_NCOEF])
{
    int saturo = 0;
    for (int c = 0; c < DPD_NCOEF; c++) {
        re[c] = a_q15(creal(w[c]), &saturo);
        im[c] = a_q15(cimag(w[c]), &saturo);
    }
    return saturo;
}
