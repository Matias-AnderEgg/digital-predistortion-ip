# Prueba en la placa

Programas del ARM y herramientas de la computadora para probar el núcleo en
una Red Pitaya STEMlab 125-14 (Zynq-7020, `xc7z020clg400-1`, la parte de
`rtl/synth/placa.tcl`).

| archivo | dónde corre | qué es |
|---|---|---|
| `main.c` | placa | `prueba`: los vectores de referencia por el núcleo |
| `entrenar_lte.c` | placa | entrena en el ARM y pasa la subtrama LTE por el núcleo |
| `ip.c`, `ip.h` | placa | acceso al núcleo y al reproductor por `/dev/mem` |
| `correr.sh` | placa | pone el reloj, carga el bitstream, compila y corre `prueba` |
| `consola.py` | computadora | comandos y archivos por la consola serie |
| `gen_vectores.py` | computadora | arma `vectores.h` con los vectores de referencia |
| `lte_placa.py` | computadora | prepara los datos de `entrenar_lte`, compara su salida y mide |
| `entrenamiento.py` | computadora | empaquetado de muestras y salida esperada del núcleo, lo usan `lte_placa.py`, el lazo y las figuras |

`vectores.h`, `placa_mapa.h` y lo que deja `lte_placa.py preparar` en
`lte/` no se versionan, porque salen de un script. De lo que graba la placa
se versionan `lte/salida_bo-9.bin` y `lte/salida_bo3.bin`, que lee
`figuras/figuras_placa.py`.

## La placa

Arranca el Linux de Red Pitaya (versión 2.07) desde la tarjeta SD. Los
programas corren como programas comunes de Linux, llegan a los periféricos
de la lógica mapeando `/dev/mem` y se compilan en la misma placa, que trae
gcc.

Todo pasa por la consola serie del micro USB (CONSOLE), que entra como root
sin contraseña. `consola.py` corre comandos por ahí y pasa archivos
comprimidos y en base64. Usa el primer `/dev/cu.usbserial-*`, o el puerto
que diga `RP_PUERTO`. La sesión se inicia una vez a mano, con `screen`.

El Linux deja el reloj de la lógica (FCLK0) en 125 MHz, y el diseño cierra
tiempos a 100 MHz. `correr.sh` escribe `FPGA0_CLK_CTRL = 0x00100A00`
(1000 MHz / 10 / 1) antes de cargar el bitstream, y `prueba` vuelve a leer el
registro e imprime la frecuencia.

## Prueba con los vectores de referencia

1. `vivado -mode batch -source rtl/synth/placa.tcl` deja en
   `rtl/synth/reportes/` el `placa_mapa.h` y el bitstream en
   `synth/placa/placa.runs/impl_1/sistema_wrapper.bit`.
2. El bitstream se pasa al formato de `fpgautil` con bootgen: un `bin.bif`
   con `all: { sistema_wrapper.bit }` y
   `bootgen -image bin.bif -arch zynq -process_bitstream bin -w on`. Deja
   `sistema_wrapper.bit.bin`, que se copia como
   `rtl/synth/reportes/placa.bit.bin`.
3. Se copia `placa_mapa.h` acá y se corre `python gen_vectores.py`.
4. Con la placa encendida y el micro USB en la computadora:

   ```bash
   python consola.py cmd "mkdir -p /root/dpd_hw"
   python consola.py enviar main.c /root/dpd_hw/main.c
   python consola.py enviar ip.c /root/dpd_hw/ip.c
   python consola.py enviar ip.h /root/dpd_hw/ip.h
   python consola.py enviar correr.sh /root/dpd_hw/correr.sh
   python consola.py enviar placa_mapa.h /root/dpd_hw/placa_mapa.h
   python consola.py enviar vectores.h /root/dpd_hw/vectores.h
   python consola.py enviar ../../rtl/synth/reportes/placa.bit.bin /root/dpd_hw/placa.bit.bin
   python consola.py cmd "sh /root/dpd_hw/correr.sh"
   ```

`prueba` carga los coeficientes por AXI-Lite y las muestras en la BRAM de
entrada, arranca el reproductor, espera `done` y compara la BRAM de salida
contra los vectores de referencia: las 4096 muestras funcionales y los 52
casos dirigidos. Con el bitstream ya cargado, `correr.sh sin_cargar` saltea
el reloj y el bitstream.

## Prueba con la señal LTE

Repite en la placa la cadena completa con la señal de las simulaciones:
entrena con las primeras 16000 muestras de la señal de semilla 123 y evalúa
sobre la primera subtrama (14 símbolos, 61440 muestras) de la de semilla
456. Se mide como en `sys_with_acpr/main.py`: PA simulado con back-off fijo,
NMSE directo, EVM con el ecualizador por subportadora y ACPR con Welch de
8192 muestras.

La subtrama no entra en la BRAM del reproductor (8192 muestras), así que
`entrenar_lte.c` la pasa en 8 trozos. Cada trozo, menos el primero, repite
al principio las 2 muestras anteriores y descarta sus salidas, así que la
salida pegada es la misma que si la subtrama pasara de una vez.

Para cada back-off, después de `make -C sw/dpd_train`:

```bash
python lte_placa.py preparar --bo -9
python consola.py enviar entrenar_lte.c /root/dpd_hw/entrenar_lte.c
python consola.py enviar ../dpd_train/dpd_train.c /root/dpd_hw/dpd_train.c
python consola.py enviar ../dpd_train/dpd_train.h /root/dpd_hw/dpd_train.h
python consola.py enviar lte/lte_ref.h /root/dpd_hw/lte_ref.h
python consola.py enviar lte/lte_ent.bin /root/dpd_hw/lte_ent.bin
python consola.py enviar lte/lte_eval_in.bin /root/dpd_hw/lte_eval_in.bin
python consola.py cmd "cd /root/dpd_hw && gcc -O2 -std=gnu99 -Wall -ffp-contract=off -o entrenar_lte entrenar_lte.c ip.c dpd_train.c -lm && ./entrenar_lte"
python consola.py traer /root/dpd_hw/lte_salida.bin lte/lte_salida.bin
python lte_placa.py medir --bo -9
python lte_placa.py separar --bo -9
```

`entrenar_lte` compara los 12 códigos Q0.15 que entrena el ARM con los del
mismo C en la computadora, y `medir` compara la salida del núcleo con el
modelo de punto fijo y mide con el PA simulado. `separar` aplica los mismos
coeficientes en flotante, en Q0.15 y en punto fijo, sobre la subtrama y
sobre los 364 símbolos, y es la comparación de la tabla de la prueba LTE.
`medir` y `separar` miden la EVM con el ecualizador. La EVM de la tesis,
con el receptor escalar, sale de `figuras/cap7_evm_escalar.py`.

Las salidas de las corridas de la tesis están en `resultados/lte/`:
`corrida_bo-9.txt`, `corrida_bo3.txt` y `separar.txt`. En los dos back-off
los 12 códigos coinciden con los de la computadora y la salida del núcleo da
las 61440 muestras iguales al modelo de Python. El entrenamiento tarda unos
9,8 s en el ARM.

## Mapa de direcciones

Lo asigna Vivado y queda en `placa_mapa.h`:

| periférico | base | tamaño |
|---|---|---|
| registros del núcleo | 0x40000000 | 4 KB |
| AXI GPIO (control del reproductor) | 0x41200000 | 64 KB |
| BRAM de entrada | 0x42000000 | 32 KB |
| BRAM de salida | 0x44000000 | 32 KB |
