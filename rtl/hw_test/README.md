# Núcleo más reproductor, para la placa

En la placa, el ARM no puede pasarle al núcleo una muestra por ciclo. Lo
hace este reproductor, que lee de una BRAM de entrada y escribe en una de
salida.

| archivo | qué es |
|---|---|
| `hdl/dpd_test_player.sv` | el reproductor y grabador |
| `hdl/dpd_hw_wrap.v` | envoltorio en Verilog del núcleo más el reproductor, para el diseño de bloques de Vivado |
| `tb/` | el mismo flujo que corre el ARM, en cocotb |
| `run.py` | lint y bancos de prueba |

## Cómo funciona

El ARM carga los coeficientes por AXI-Lite y la BRAM de entrada, escribe la
cantidad de muestras con `start` en 1 por un AXI GPIO, espera `done` y lee
la BRAM de salida. El reproductor lee una muestra por ciclo, le da `en` al
núcleo y guarda cada salida en la misma posición que su entrada. Las
muestras ocupan una palabra de 32 bits, `{im, re}`, en las dos BRAM.

El reproductor no mira `valid` sino que cuenta los pulsos de `en`. El
núcleo conserva su estado entre corridas, y resetearlo volvería los
coeficientes al paso directo. La salida de la muestra `s` aparece después
del pulso `s + LATENCY - 1`, así que al final se agregan `LATENCY - 1`
ceros, que además dejan la línea de retardo en cero para la corrida
siguiente.

## Verificación

```bash
python rtl/hw_test/run.py
```

| prueba | qué comprueba |
|---|---|
| vectores en corridas seguidas | los funcionales (4096), los dirigidos (52) y los funcionales otra vez, sin reset en el medio, iguales bit a bit a los de referencia |
| duración | 4096 muestras en 4127 ciclos: una por ciclo más la latencia |
| cero muestras | `done` enseguida |

## En Vivado

`rtl/synth/placa.tcl` arma el diseño de bloques para la Zynq-7020
(`xc7z020clg400-1`), lo implementa y genera el bitstream. Los recursos y
los tiempos están en `rtl/synth/README.md`.
