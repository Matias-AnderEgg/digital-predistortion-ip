/*
 * Lado de la placa del lazo con el PA simulado y la senal LTE de la tesis.
 *
 * Cada vuelta usa una subtrama LTE (14 simbolos, 61440 muestras). No entra
 * en la BRAM del reproductor, asi que pasa por el IP de a trozos, como en
 * sw/hw_test/entrenar_lte.c: cada trozo menos el primero repite al principio
 * las DPD_M - 1 muestras anteriores y sus salidas se descartan, y la salida
 * pegada es la misma que si la subtrama pasara de una vez.
 *
 *   ./lazo_lte inicio        w en paso directo, cargado en el IP
 *   ./lazo_lte aplicar k     pasa la subtrama k por el IP y deja la salida en u.bin
 *   ./lazo_lte modelo n      identifica el PA con las primeras n muestras de
 *                            u.bin y de y.bin, deja h.bin
 *   ./lazo_lte entrenar k n  una pasada del DPD sobre las primeras n muestras
 *                            de la subtrama k, partiendo del w actual, y carga
 *                            en el IP el nuevo w dividido por el RMS de la
 *                            salida del DPD sobre la subtrama entera
 *
 * Archivos: x_bloques.bin (las subtramas, en codigos del IP, las manda la Mac
 * al principio), y.bin (salida del PA, la manda la Mac), u.bin, h.bin y
 * w.bin (los escribe este programa).
 *
 * Se compila en la placa:
 *   gcc -O2 -std=gnu99 -Wall -ffp-contract=off -o lazo_lte lazo_lte.c ip.c dpd_train.c -lm
 */
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dpd_train.h"
#include "ip.h"

#define N_BLOQUE  61440         /* una subtrama LTE de 1 ms a 61,44 MHz */
#define G_X       8.0           /* escala de entrada del IP */
#define G_U       128.0         /* escala de salida del IP */
#define G_Y       8.0           /* escala con que la Mac manda y */
#define PREVIAS   (DPD_M - 1)

static uint32_t codigos[N_BLOQUE], salida[N_BLOQUE];
static uint32_t trozo_in[IP_MAX_MUESTRAS], trozo_out[IP_MAX_MUESTRAS];
static double complex x[N_BLOQUE], u[N_BLOQUE], y[N_BLOQUE];

static double segundos(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static int leer(const char *ruta, void *dst, size_t tam, long desde)
{
    FILE *f = fopen(ruta, "rb");
    int ok = f && fseek(f, desde, SEEK_SET) == 0 && fread(dst, tam, 1, f) == 1;
    if (f)
        fclose(f);
    if (!ok)
        fprintf(stderr, "no se pudo leer %s\n", ruta);
    return ok ? 0 : -1;
}

static int escribir(const char *ruta, const void *src, size_t tam)
{
    FILE *f = fopen(ruta, "wb");
    int ok = f && fwrite(src, tam, 1, f) == 1;
    if (f)
        fclose(f);
    if (!ok)
        fprintf(stderr, "no se pudo escribir %s\n", ruta);
    return ok ? 0 : -1;
}

/* {im, re} en 16 bits cada uno, Q0.15 sobre la escala g. */
static void a_complejos(const uint32_t *c, double complex *v, int n, double g)
{
    for (int i = 0; i < n; i++)
        v[i] = g / 32768.0 * ((int16_t)(c[i] & 0xFFFF) + I * (int16_t)(c[i] >> 16));
}

/* Pasa w a Q0.15, lo carga en el IP e imprime los codigos para la Mac. */
static void cargar(const double complex w[DPD_NCOEF])
{
    int16_t re[DPD_NCOEF], im[DPD_NCOEF];
    uint32_t c[DPD_NCOEF];
    dpd_coef_to_q15(w, re, im);
    printf("w");
    for (int i = 0; i < DPD_NCOEF; i++) {
        c[i] = ((uint32_t)(uint16_t)im[i] << 16) | (uint16_t)re[i];
        printf(" %08x", c[i]);
    }
    printf("\n");
    ip_cargar_coeficientes(c);
}

/* La subtrama entera por el IP, de a trozos. */
static int por_el_ip(const uint32_t *entrada, uint32_t *sal)
{
    int trozos = 0;
    for (int s = 0; s < N_BLOQUE; trozos++) {
        int previas = s == 0 ? 0 : PREVIAS;
        int nuevas = N_BLOQUE - s;
        if (nuevas > (int)IP_MAX_MUESTRAS - previas)
            nuevas = IP_MAX_MUESTRAS - previas;
        memcpy(trozo_in, entrada + s - previas, (previas + nuevas) * sizeof entrada[0]);
        if (ip_correr(trozo_in, trozo_out, previas + nuevas) != 0) {
            fprintf(stderr, "el reproductor no termino en el trozo %d\n", trozos);
            return -1;
        }
        memcpy(sal + s, trozo_out + previas, nuevas * sizeof sal[0]);
        s += nuevas;
    }
    return 0;
}

int main(int argc, char **argv)
{
    dpd_train_cfg cfg;
    double complex h[DPD_NCOEF], w[DPD_NCOEF];
    double mse[64];
    const char *orden = argc > 1 ? argv[1] : "";
    int k = argc > 2 ? atoi(argv[2]) : 0;
    int n = argc > 3 ? atoi(argv[3]) : N_BLOQUE;

    if (ip_abrir() != 0 || ip_id() != IP_ID_ESPERADO) {
        fprintf(stderr, "no hay IP, falta cargar el bitstream\n");
        return 2;
    }
    dpd_train_cfg_default(&cfg);

    if (strcmp(orden, "inicio") == 0) {
        memset(w, 0, sizeof w);
        w[0] = 1.0;
        cargar(w);
        return escribir("w.bin", w, sizeof w);
    }

    if (strcmp(orden, "aplicar") == 0) {
        if (leer("x_bloques.bin", codigos, sizeof codigos, (long)k * sizeof codigos) != 0)
            return 2;
        double t = segundos();
        if (por_el_ip(codigos, salida) != 0)
            return 1;
        printf("aplicar %.1f ms\n", 1e3 * (segundos() - t));
        return escribir("u.bin", salida, sizeof salida);
    }

    if (strcmp(orden, "modelo") == 0) {
        int nm = k;
        if (nm < 1 || nm > N_BLOQUE) {
            fprintf(stderr, "entre 1 y %d muestras\n", N_BLOQUE);
            return 2;
        }
        size_t tam = (size_t)nm * sizeof codigos[0];
        if (leer("u.bin", codigos, tam, 0) != 0)
            return 2;
        a_complejos(codigos, u, nm, G_U);
        if (leer("y.bin", codigos, tam, 0) != 0)
            return 2;
        a_complejos(codigos, y, nm, G_Y);
        double t = segundos();
        int ep = dpd_train_pa_model(u, y, nm, &cfg, h, mse);
        /* Con epocas negativas se uso el respaldo, ver dpd_train.h. */
        int epa = ep < 0 ? -ep : ep;
        printf("modelo %.3f s, %d muestras, %d epocas%s, MSE %.4e\n", segundos() - t, nm, epa,
               ep < 0 ? " (con el respaldo)" : "", mse[epa - 1]);
        return escribir("h.bin", h, sizeof h);
    }

    if (strcmp(orden, "entrenar") == 0) {
        if (leer("h.bin", h, sizeof h, 0) != 0 || leer("w.bin", w, sizeof w, 0) != 0
            || leer("x_bloques.bin", codigos, sizeof codigos, (long)k * sizeof codigos) != 0)
            return 2;
        a_complejos(codigos, x, N_BLOQUE, G_X);
        cfg.warm_start = 1;
        cfg.ep_dpd = 1;
        if (n < 1 || n > N_BLOQUE)
            n = N_BLOQUE;
        double t = segundos();
        dpd_train_dla(x, n, h, &cfg, w, mse);
        /* Al IP va w / c, para que el PA reciba la misma potencia media que
         * sin DPD. c se mide con la subtrama entera. w.bin guarda el w sin
         * escalar, que es del que sigue entrenando. */
        double c = dpd_rms_salida(x, N_BLOQUE, w, &cfg), t_ent = segundos() - t;
        double complex wc[DPD_NCOEF];
        for (int i = 0; i < DPD_NCOEF; i++)
            wc[i] = w[i] / c;
        printf("entrenar %.3f s, MSE %.4e, escala %.6f\n", t_ent, mse[0], c);
        cargar(wc);
        return escribir("w.bin", w, sizeof w);
    }

    fprintf(stderr, "uso: lazo_lte inicio | aplicar k | modelo n | entrenar k [n]\n");
    return 2;
}
