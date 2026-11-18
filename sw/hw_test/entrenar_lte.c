/*
 * La cadena completa en el chip con la senal LTE de 10 MHz de la tesis, con
 * los datos de lte_placa.py:
 * 1. Lee la entrada y la salida del PA de entrenamiento de lte_ent.bin, y
 *    la subtrama de evaluacion, ya en codigos del IP, de lte_eval_in.bin.
 * 2. Entrena con sw/dpd_train y compara h, w y los codigos Q0.15 con lo que
 *    dio el mismo C en la Mac (lte_ref.h).
 * 3. Carga los coeficientes y pasa la subtrama por el IP de a trozos.
 * 4. Deja la salida en lte_salida.bin. La comparacion contra el modelo de
 *    Python y las metricas se hacen en la Mac (lte_placa.py medir), para no
 *    traer la salida esperada por la consola.
 *
 * Por que de a trozos: la subtrama tiene 61440 muestras y la BRAM del
 * reproductor 8192. Cada corrida del reproductor arranca con la memoria del
 * DPD en cero (los ceros que mete al final de la anterior), y la salida de
 * la muestra n depende de x(n), x(n-1) y x(n-2). Entonces cada trozo, menos
 * el primero, repite al principio las DPD_M - 1 muestras anteriores, y sus
 * salidas se descartan. Asi la salida pegada es la misma que si la subtrama
 * entera pasara de una vez, que es lo que compara lte_placa.py.
 *
 * Se compila en la placa:
 *   gcc -O2 -std=gnu99 -Wall -ffp-contract=off -o entrenar_lte entrenar_lte.c ip.c dpd_train.c -lm
 */
#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#include "dpd_train.h"
#include "ip.h"
#include "lte_ref.h"

#define PREVIAS (DPD_M - 1)

static double complex x[N_ENT], y[N_ENT];
static uint32_t entrada[N_EVAL], salida[N_EVAL];
static uint32_t trozo_in[IP_MAX_MUESTRAS], trozo_out[IP_MAX_MUESTRAS];

static double segundos(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int leer(const char *ruta, void *a, size_t tam, size_t n)
{
    FILE *f = fopen(ruta, "rb");
    if (!f) {
        perror(ruta);
        return -1;
    }
    size_t leidos = fread(a, tam, n, f);
    fclose(f);
    if (leidos != n) {
        printf("%s: %zu de %zu elementos\n", ruta, leidos, n);
        return -1;
    }
    return 0;
}

static int leer_entrenamiento(void)
{
    FILE *f = fopen("lte_ent.bin", "rb");
    int32_t n = 0;
    if (!f) {
        perror("lte_ent.bin");
        return -1;
    }
    int ok = fread(&n, sizeof n, 1, f) == 1 && n == N_ENT
          && fread(x, sizeof x[0], N_ENT, f) == N_ENT
          && fread(y, sizeof y[0], N_ENT, f) == N_ENT;
    fclose(f);
    if (!ok)
        printf("lte_ent.bin no tiene %d muestras de x y de y\n", N_ENT);
    return ok ? 0 : -1;
}

/* Maxima diferencia contra la referencia de la Mac, en LSB de Q0.15. */
static double dif_lsb(const double complex c[DPD_NCOEF], const double ref[2 * DPD_NCOEF])
{
    double m = 0.0;
    for (int i = 0; i < DPD_NCOEF; i++) {
        double d = cabs(c[i] - (ref[2 * i] + I * ref[2 * i + 1]));
        if (d > m)
            m = d;
    }
    return m * 32768.0;
}

int main(void)
{
    dpd_train_cfg cfg;
    double complex h[DPD_NCOEF], w[DPD_NCOEF];
    double mse_h[64], mse_w[64];
    int16_t re[DPD_NCOEF], im[DPD_NCOEF];
    uint32_t coefs[DPD_NCOEF];

    if (leer_entrenamiento() != 0
        || leer("lte_eval_in.bin", entrada, sizeof entrada[0], N_EVAL) != 0
        || ip_abrir() != 0)
        return 2;

    printf("\n=== LTE 10 MHz: entrenamiento en el ARM, aplicacion en el IP ===\n");
    printf("FCLK0 %.3f MHz, ID 0x%08x, BO %g dB\n", ip_fclk0_mhz(), ip_id(), (double)BO_DB);
    if (ip_id() != IP_ID_ESPERADO) {
        printf("no hay IP en esa direccion\n");
        return 1;
    }

    dpd_train_cfg_default(&cfg);
    printf("%d muestras, %d epocas del modelo del PA y %d del DPD\n", N_ENT, cfg.ep_pa, cfg.ep_dpd);
    double t0 = segundos();
    int ep_h = dpd_train_pa_model(x, y, N_ENT, &cfg, h, mse_h);
    double t1 = segundos();
    int ep_w = dpd_train_dla(x, N_ENT, h, &cfg, w, mse_w);
    double t2 = segundos();
    int respaldo = ep_h < 0;
    if (respaldo)
        ep_h = -ep_h;
    printf("modelo del PA:     %5.2f s, %d epocas%s, MSE final %.3e\n", t1 - t0, ep_h,
           respaldo ? " (con el respaldo)" : "", mse_h[ep_h - 1]);
    printf("predistorsionador: %5.2f s, %d epocas, MSE final %.3e\n", t2 - t1, ep_w, mse_w[ep_w - 1]);
    printf("total:             %5.2f s\n", t2 - t0);
    printf("contra el C en la Mac: h %.1e LSB, w %.1e LSB\n", dif_lsb(h, H_REF), dif_lsb(w, W_REF));

    dpd_coef_to_q15(w, re, im);
    int distintos = 0;
    for (int i = 0; i < DPD_NCOEF; i++) {
        coefs[i] = ((uint32_t)(uint16_t)im[i] << 16) | (uint16_t)re[i];
        distintos += coefs[i] != COEFS_REF[i];
    }
    printf("codigos Q0.15: %d de %d iguales a los de la Mac\n", DPD_NCOEF - distintos, DPD_NCOEF);

    ip_cargar_coeficientes(coefs);
    int trozos = 0;
    double t3 = segundos();
    for (int s = 0; s < N_EVAL; trozos++) {
        int previas = s == 0 ? 0 : PREVIAS;
        int nuevas = N_EVAL - s;
        if (nuevas > (int)IP_MAX_MUESTRAS - previas)
            nuevas = IP_MAX_MUESTRAS - previas;
        memcpy(trozo_in, entrada + s - previas, (previas + nuevas) * sizeof entrada[0]);
        if (ip_correr(trozo_in, trozo_out, previas + nuevas) != 0) {
            printf("el reproductor no termino en el trozo %d\n", trozos);
            return 1;
        }
        memcpy(salida + s, trozo_out + previas, nuevas * sizeof salida[0]);
        s += nuevas;
    }
    double t4 = segundos();
    printf("subtrama: %d muestras en %d trozos, %.1f ms con las copias a la BRAM\n",
           N_EVAL, trozos, 1e3 * (t4 - t3));

    FILE *f = fopen("lte_salida.bin", "wb");
    if (!f || fwrite(salida, sizeof salida[0], N_EVAL, f) != N_EVAL) {
        perror("lte_salida.bin");
        return 2;
    }
    fclose(f);
    printf(distintos == 0 ? "coeficientes OK, la salida se compara en la Mac\n" : "HAY FALLAS\n");
    return distintos == 0 ? 0 : 1;
}
