/*
 * Prueba en hardware del IP de aplicacion del DPD, en el ARM de la Zynq.
 *
 * Hace en la placa lo mismo que rtl/hw_test/tb/test_dpd_hw.py en
 * simulacion: carga los coeficientes de los golden vectors por AXI-Lite,
 * carga la BRAM de entrada, arranca el reproductor por el GPIO, espera
 * done y compara la BRAM de salida contra el golden. Primero los 4096
 * vectores funcionales, despues los 52 dirigidos.
 *
 * Corre bajo el Linux de la Red Pitaya, como root: llega a los perifericos
 * del PL mapeando /dev/mem (ver ip.c). Se compila en la misma placa:
 *   gcc -O2 -o prueba main.c ip.c
 * Necesita al lado:
 *   placa_mapa.h   copiado de rtl/synth/reportes/
 *   vectores.h     generado con gen_vectores.py
 *
 * El bitstream se carga antes, con correr.sh. Imprime el resultado y
 * termina.
 */
#include <stdint.h>
#include <stdio.h>

#include "ip.h"
#include "vectores.h"

static uint32_t salida[IP_MAX_MUESTRAS];

/* Corre n muestras y devuelve cuantas salidas difieren del esperado. */
static uint32_t correr(const char *nombre, const uint32_t *entrada,
                       const uint32_t *esperado, uint32_t n)
{
    uint32_t malas = 0;

    if (ip_correr(entrada, salida, n) != 0) {
        printf("%s: el reproductor no termino\n", nombre);
        return n;
    }
    for (uint32_t k = 0; k < n; k++) {
        if (salida[k] != esperado[k]) {
            if (malas < 5)
                printf("  %s[%u]: 0x%08x, se esperaba 0x%08x\n",
                       nombre, k, salida[k], esperado[k]);
            malas++;
        }
    }
    printf("%s: %u de %u muestras iguales al golden\n", nombre, n - malas, n);
    return malas;
}

int main(void)
{
    if (ip_abrir() != 0)
        return 2;

    printf("\n=== prueba del IP de DPD ===\n");
    printf("FCLK0 %.3f MHz\n", ip_fclk0_mhz());

    uint32_t id = ip_id(), cfg = ip_config();
    printf("ID 0x%08x, CONFIG 0x%08x\n", id, cfg);
    if (id != IP_ID_ESPERADO) {
        printf("el ID no es el esperado (0x%08x), no hay IP en esa direccion\n",
               IP_ID_ESPERADO);
        return 1;
    }

    ip_cargar_coeficientes(COEFS);

    uint32_t malas = 0;
    malas += correr("funcionales", FUNCTIONAL_IN, FUNCTIONAL_OUT,
                    sizeof FUNCTIONAL_IN / sizeof FUNCTIONAL_IN[0]);
    malas += correr("dirigidos", DIRECTED_IN, DIRECTED_OUT,
                    sizeof DIRECTED_IN / sizeof DIRECTED_IN[0]);

    printf(malas == 0 ? "TODO OK\n" : "HAY FALLAS\n");
    return malas == 0 ? 0 : 1;
}
