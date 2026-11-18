#!/bin/sh
# Corre las pruebas en la Red Pitaya. Va en la placa, en la misma carpeta que
# placa.bit.bin y las fuentes (ver README).
#
#   sh correr.sh             reloj, bitstream y golden vectors
#   sh correr.sh sin_cargar  solo la prueba, con el bitstream ya cargado
#
# 1. Pone FCLK0 en 100 MHz. El Linux de la Red Pitaya lo deja en 125 MHz,
#    y el diseno cierra tiempos a 100 MHz. FPGA0_CLK_CTRL = IO PLL (1000 MHz)
#    / 10 / 1. Se hace antes de cargar el bitstream para que la logica
#    arranque ya con el reloj final.
# 2. Carga el bitstream.
# 3. Compila y corre la prueba con los golden vectors.
#
# La cadena completa con la LTE es entrenar_lte.c (ver README).
set -e
cd "$(dirname "$0")"
MON=/opt/redpitaya/bin/monitor

if [ "$1" != "sin_cargar" ]; then
    $MON 0xF8000008 0xDF0D            # desbloquea el SLCR (Linux lo deja asi)
    $MON 0xF8000170 0x00100A00
    echo "FPGA0_CLK_CTRL = $($MON 0xF8000170)"
    /opt/redpitaya/bin/fpgautil -b placa.bit.bin -f Full
fi

gcc -O2 -Wall -o prueba main.c ip.c
./prueba
