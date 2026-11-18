#include "ip.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <unistd.h>

#include "placa_mapa.h"

/* Banco de registros del IP, rtl/dpd_regs/rdl/dpd_apply_regs.rdl */
#define REG_ID        0x000U
#define REG_CONFIG    0x004U
#define REG_CTRL      0x008U
#define REG_STATUS    0x00CU
#define REG_COEF(i)   (0x040U + 4U * (i))
#define CTRL_BYPASS   (1U << 0)
#define CTRL_COMMIT   (1U << 8)

/* AXI GPIO: canal 1 es la salida (ctrl), canal 2 la entrada (estado). */
#define GPIO_CTRL     0x000U
#define GPIO_ESTADO   0x008U
#define START         (1U << 31)
#define ESTADO_DONE   (1U << 0)

/* Reloj del PL: FPGA0_CLK_CTRL e IO_PLL_CTRL, en el SLCR del PS. */
#define SLCR_BASE        0xF8000000U
#define IO_PLL_CTRL      0x108U
#define FPGA0_CLK_CTRL   0x170U
#define PS_CLK_MHZ       33.333333

#define REG(p, off)  ((p)[(off) / 4U])

static int fd = -1;
static volatile uint32_t *dpd, *gpio, *bram_in, *bram_out;

static volatile uint32_t *mapear(uint32_t base, uint32_t rango)
{
    void *p = mmap(NULL, rango, PROT_READ | PROT_WRITE, MAP_SHARED, fd, base);
    if (p == MAP_FAILED) {
        perror("mmap");
        return NULL;
    }
    return (volatile uint32_t *)p;
}

int ip_abrir(void)
{
    fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        perror("/dev/mem (hay que correrlo como root)");
        return -1;
    }
    dpd      = mapear(SEG_DPD_REG0_BASE, SEG_DPD_REG0_RANGO);
    gpio     = mapear(SEG_GPIO_REG_BASE, SEG_GPIO_REG_RANGO);
    bram_in  = mapear(SEG_BRAM_CTRL_IN_MEM0_BASE, SEG_BRAM_CTRL_IN_MEM0_RANGO);
    bram_out = mapear(SEG_BRAM_CTRL_OUT_MEM0_BASE, SEG_BRAM_CTRL_OUT_MEM0_RANGO);
    return (dpd && gpio && bram_in && bram_out) ? 0 : -1;
}

double ip_fclk0_mhz(void)
{
    volatile uint32_t *slcr = mapear(SLCR_BASE, 0x1000);
    if (!slcr)
        return 0.0;
    uint32_t pll = REG(slcr, IO_PLL_CTRL), clk = REG(slcr, FPGA0_CLK_CTRL);
    munmap((void *)slcr, 0x1000);
    if (((clk >> 4) & 3U) != 0)          /* solo se contempla el IO PLL */
        return 0.0;
    double fdiv = (pll >> 12) & 0x7FU;
    double d0 = (clk >> 8) & 0x3FU, d1 = (clk >> 20) & 0x3FU;
    return PS_CLK_MHZ * fdiv / d0 / d1;
}

uint32_t ip_id(void)     { return REG(dpd, REG_ID); }
uint32_t ip_config(void) { return REG(dpd, REG_CONFIG); }

void ip_cargar_coeficientes(const uint32_t coefs[IP_N_COEF])
{
    for (uint32_t i = 0; i < IP_N_COEF; i++)
        REG(dpd, REG_COEF(i)) = coefs[i];
    REG(dpd, REG_CTRL) = CTRL_COMMIT;
}

int ip_correr(const uint32_t *entrada, uint32_t *salida, uint32_t n)
{
    uint32_t espera = 0;

    for (uint32_t k = 0; k < n; k++)
        bram_in[k] = entrada[k];

    REG(gpio, GPIO_CTRL) = START | n;
    while ((REG(gpio, GPIO_ESTADO) & ESTADO_DONE) == 0) {
        if (++espera > 10000000U) {
            REG(gpio, GPIO_CTRL) = 0;
            return -1;
        }
    }
    REG(gpio, GPIO_CTRL) = 0;

    for (uint32_t k = 0; k < n; k++)
        salida[k] = bram_out[k];
    return 0;
}
