/*
 * Entrenamiento del predistorsionador por gradiente, para el ARM del SoC.
 *
 * Es un port de fixed_point/dpd.py: identificacion del modelo del PA por
 * LMS precondicionado y entrenamiento del DPD por aprendizaje directo (DLA)
 * con gradiente de Wirtinger. Corre en punto flotante de doble precision,
 * porque el ARM tiene FPU, y recien al final se pasan los coeficientes a
 * Q0.15 para escribirlos por AXI-Lite. Lo unico en punto fijo es lo que va
 * a la FPGA.
 *
 * Usa la convencion normalizada del IP: la senal fisica se divide por G
 * antes de entrar y cada termino de orden p lleva un factor G^(p-1). Asi
 * los coeficientes que salen son los mismos que espera el datapath.
 *
 * Las dimensiones son las de la instancia del IP (M=3, ordenes 1, 3, 5 y 7)
 * y quedan fijas en compilacion: no hay memoria dinamica.
 */
#ifndef DPD_TRAIN_H
#define DPD_TRAIN_H

#include <complex.h>
#include <stdint.h>

#define DPD_M     3                     /* taps de memoria */
#define DPD_NORD  4                     /* ordenes impares 1, 3, 5, 7 */
#define DPD_NCOEF (DPD_M * DPD_NORD)    /* 12, en orden tap-mayor: w[q][p] */

typedef struct {
    double G;                   /* escala de entrada del IP */

    /* Modelo del PA */
    double mu_h;                /* adimensional, estable en (0, 2) con NLMS */
    int    ep_pa;
    int    use_nlms_h;
    double divergence_factor;   /* corta si el MSE supera este factor del mejor */
    int    quantize_h;          /* 1: predice con h en Q0.15, como el entrenamiento
                                   en punto fijo. Apagado */

    /* Predistorsionador */
    double mu_w;
    int    ep_dpd;
    int    use_nlms_w;
    int    freeze_w0;           /* w[0] fijo en 1: no puede elegir atenuar */
    double grad_clip;           /* 0 lo desactiva */
    double w_reg_lambda;        /* fuga sobre los ordenes >= w_reg_pmin */
    int    w_reg_pmin;

    double precond_delta;       /* evita dividir por cero en las normalizaciones.
                                   En la potencia de las columnas va dividido
                                   por G^2 */

    /* Si el modelo del PA diverge, se lo reentrena con NLMS y este paso.
     * 0 lo desactiva. */
    double mu_h_respaldo;

    /* 0: cada lazo arranca de cero (h en la ganancia lineal de x a y, w en
     * paso directo). 1: arranca del h o w que se le pasa, para seguir
     * entrenando de a pocas epocas en un lazo cerrado (ver
     * sw/lazo_simulado). */
    int    warm_start;

    /* Proteccion del modelo del PA contra los picos de la entrada. 0 las
     * desactiva.
     * h_ganancia_max: si la ganancia de lazo de una muestra (paso por
     *   energia del regresor) pasa de este valor, el paso se achica hasta
     *   ese valor. Debajo, el paso es el de siempre. Vale 1 por defecto.
     * h_umbral_pico:
     *   no actualiza h en las muestras cuyo regresor tiene
     *   algun tap por encima de esta amplitud, en las unidades de x. Al
     *   reidentificar con el DPD prendido, va el pico de la entrada con que
     *   se identifico el PA con el DPD apagado: fuera de ese rango el
     *   polinomio extrapola. */
    double h_ganancia_max;
    double h_umbral_pico;
} dpd_train_cfg;

/* Los valores con que se generan los vectores dorados. */
void dpd_train_cfg_default(dpd_train_cfg *cfg);

/*
 * Identifica el modelo del PA, y ~ h^T phi(x).
 *
 * x e y son n muestras en escala fisica: x con RMS 1 e y la salida del PA
 * alineada en tiempo con x. mse_hist recibe el MSE de cada epoca (puede ser
 * NULL) y tiene que tener lugar para cfg->ep_pa valores.
 *
 * Devuelve la cantidad de epocas corridas. Si es menor que ep_pa, el lazo
 * se corto por divergencia y h tiene los mejores coeficientes vistos.
 *
 * Respaldo: si el lazo se corta por divergencia y cfg->mu_h_respaldo > 0,
 * se lo vuelve a entrenar desde cero con NLMS y mu_h_respaldo, y se
 * devuelven las epocas de ese segundo intento con signo negativo.
 */
int dpd_train_pa_model(const double complex *x, const double complex *y, int n,
                       const dpd_train_cfg *cfg,
                       double complex h[DPD_NCOEF], double *mse_hist);

/*
 * Entrena el predistorsionador contra el modelo h del PA.
 *
 * x son las mismas n muestras de entrada. w recibe los coeficientes de la
 * ultima epoca (o de la ultima finita, si el lazo diverge), en la escala que
 * usa el IP.
 */
int dpd_train_dla(const double complex *x, int n, const double complex h[DPD_NCOEF],
                  const dpd_train_cfg *cfg,
                  double complex w[DPD_NCOEF], double *mse_hist);

/*
 * RMS de la salida del predistorsionador para la entrada x, en escala
 * fisica, calculado en flotante.
 *
 * Es la renormalizacion de potencia del lazo: el ARM escribe en el IP w
 * dividido por este valor, y asi el PA recibe la misma potencia media con
 * DPD que sin DPD (la convencion fixed_bo, pero hecha por el sistema y no
 * escondida en el modelo del PA). Como la salida es lineal en w, dividir
 * los coeficientes equivale a dividir la senal.
 */
double dpd_rms_salida(const double complex *x, int n, const double complex w[DPD_NCOEF],
                      const dpd_train_cfg *cfg);

/*
 * Pasa los coeficientes a los codigos Q0.15 de W_COEFF_RE / W_COEFF_IM,
 * redondeando al mas cercano y saturando. Devuelve cuantos saturaron, que
 * es normal en w[0]: vale 1 y Q0.15 llega hasta 1 - 2^-15.
 */
int dpd_coef_to_q15(const double complex w[DPD_NCOEF],
                    int16_t re[DPD_NCOEF], int16_t im[DPD_NCOEF]);

#endif
