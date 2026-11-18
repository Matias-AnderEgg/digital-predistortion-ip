/*
 * Acceso al IP de DPD y al reproductor desde el Linux de la Red Pitaya.
 *
 * Lo usan los programas de la placa: main.c, que pasa los golden vectors,
 * entrenar_lte.c, que entrena en el ARM y aplica en el IP, y lazo_lte.c de
 * sw/lazo_simulado.
 */
#ifndef IP_H
#define IP_H

#include <stdint.h>

#define IP_ID_ESPERADO  0x44504431U        /* "DPD1" */
#define IP_N_COEF       12U
#define IP_MAX_MUESTRAS 8192U              /* 32 KB por BRAM */

/* Mapea los perifericos por /dev/mem. Devuelve 0 si salio bien. */
int ip_abrir(void);

/* Frecuencia de FCLK0 segun los registros del PS, en MHz. */
double ip_fclk0_mhz(void);

uint32_t ip_id(void);
uint32_t ip_config(void);

/* Escribe los 12 coeficientes, {im, re} en 32 bits, y pide commit. El
 * commit se aplica en el primer pulso de `en`, o sea al arrancar la
 * proxima corrida. */
void ip_cargar_coeficientes(const uint32_t coefs[IP_N_COEF]);

/* Pasa n muestras por el IP con el reproductor y deja las salidas en
 * `salida`. Devuelve 0 si salio bien, -1 si el reproductor no termino. */
int ip_correr(const uint32_t *entrada, uint32_t *salida, uint32_t n);

#endif
