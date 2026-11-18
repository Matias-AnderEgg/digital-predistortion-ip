# Reproducción de los resultados

Cómo se vuelve a generar cada figura y cada tabla de la tesis. Todo usa
semillas fijas, así que cada corrida tiene que dar lo mismo. La organización
del repositorio y las herramientas están en el Apéndice D de la tesis. Cada
directorio del RTL y del software tiene además su propio README.

Las secciones se nombran por título y no por número, porque la numeración de
la tesis puede cambiar.

## Simulaciones (Caps. 2, 3 y 6)

Los modelos y los scripts están en Python y usan NumPy, SciPy y Matplotlib.

Cada script de `sys_with_acpr/` corre sus simulaciones y guarda los resultados en un subdirectorio `resultados_*/`:
- un `.pkl` con los datos;
- un `.log` con lo que imprimió la corrida.

Los que tienen muchos puntos independientes, como el barrido, los reparten entre varios procesos con la opción `--jobs`.

| Sección de la tesis | Script | Resultados |
|---|---|---|
| Validación de la cadena | `validacion_canal.py` | `resultados_validacion/` |
| Efectos de memoria, y Consecuencias de la distorsión no lineal sobre el enlace | `efecto_pa.py` | `resultados_efecto_pa/` |
| Elección del método, el orden y la memoria | `barrido_mp.py` | `resultados_barrido/` |
| Condicionamiento y sobreajuste | `rango_ila.py` | `resultados_identificacion/` |
| Configuración elegida | `identificacion_pa.py` | `resultados_identificacion/` |
| Desempeño de la configuración elegida | `desempeno.py` | `resultados_desempeno/` |
| Convergencia del entrenamiento | `epocas_dpd.py`, `epocas_pa.py`, `piso_grilla.py` | `resultados_identificacion/` |
| Robustez del entrenamiento | `robustez.py`, `robustez_arreglos.py` | `resultados_robustez/` |

### El piso a back-off alto

El análisis del piso del predistorsionador a back-off alto (Desempeño de la configuración elegida) no sale de `sys_with_acpr/`.
- **Dónde está:** `sw/otro_pa/otro_pa.py`, con las opciones `piso`, `gradiente` y `normalizacion`. Deja sus resultados en `sw/otro_pa/resultados/`.
- **Qué entrenamiento usa:** el C de `sw/dpd_train/`, el mismo que corre en el procesador.
- **El ajuste contra el amplificador real** (Convergencia del entrenamiento) usa ese mismo análisis en toda la grilla de back-off, con `piso_grilla.py`. Ese script importa `sw/otro_pa/otro_pa.py`, así que necesita esa parte del repositorio.

## Figuras

Las figuras no las dibujan los scripts de simulación. Las dibuja `figuras/genera_figuras.py`, que lee esos `.pkl` y deja cada figura en PDF en `figuras/`.
- **Las figuras de los capítulos teóricos** no dependen de una corrida larga. Las calcula en el momento, con los mismos módulos del modelo de referencia.
- **Sin argumentos** genera todas. Con `--only` y una lista de nombres genera sólo esas.

Separar la simulación del dibujo permite cambiar una figura sin volver a correr la simulación.

Otras figuras y tablas salen de scripts propios, todos en `figuras/`:
- **`cuantizacion_zf.py`:** las dos figuras de cuantización del Cap. 7. Usa el modelo de punto fijo.
- **`tablas_barrido.py`:** las tablas del Apéndice A. Lee el mismo resultado del barrido que sus figuras.
- **`figuras_placa.py`:** las figuras de la validación en la placa y su tabla. Lee lo que dejaron las corridas en la placa.
- **`cap7_evm_escalar.py`:** la EVM del Cap. 7 con el receptor escalar, recalculada desde las mismas salidas de la placa y del lazo. Repite el lazo en la computadora y comprueba que el NMSE de cada vuelta sea el de la placa.

## Vectores de referencia y verificación del RTL

El script `generate.py` de `fixed_point/golden_vectors/` genera los vectores de referencia de la verificación funcional:
- las 4096 muestras funcionales;
- las 52 dirigidas, con sus coeficientes.

Los deja en el subdirectorio `out/` en dos formatos:
- archivos hexadecimales, que son los que leen los bancos de prueba;
- CSV, para mirarlos.

La simulación del RTL usa Verilator, y los bancos de prueba están escritos en Python con cocotb.
- Cada directorio de `rtl/` tiene un `run.py` que compila el diseño, corre su banco de pruebas y reporta el resultado.
- El de `rtl/dpd_regs/` además regenera el RTL del banco de registros a partir del archivo SystemRDL, con PeakRDL.
- El RTL generado también está en el repositorio, así que el núcleo se puede simular y sintetizar sin tener PeakRDL instalado.

## Síntesis

La síntesis se hizo con Vivado 2026.1, en modo por lotes y sin proyecto. Así cada corrida empieza de cero y sólo deja reportes en `rtl/synth/reportes/`. Los scripts están en `rtl/synth/`:
- **`implementacion.tcl`:** sintetiza e implementa el núcleo solo, fuera de contexto, sobre la Zynq-7020 de la placa.
- **`placa.tcl`:** arma el sistema completo de la placa, con el procesador, el núcleo y el reproductor, y genera su bitstream.
- **`tabla_recursos.py`:** lee los reportes y escribe en LaTeX las tablas de recursos y de DSP48. También comprueba que los DSP48 de cada función sumen el total.

La variante que calcula las potencias una sola vez para las tres ramas (Compromisos entre recursos y desempeño) está en `rtl/variantes/potencias_compartidas/`, y la del cierre de tiempos sin registros intermedios en `rtl/variantes/sin_registros_intermedios/`. Las dos se simulan y se sintetizan con los mismos scripts (ver `rtl/variantes/README.md`).

## Software del procesador

El entrenamiento en C de `sw/dpd_train/` se compila con `make`, y `make test` lo compara contra el modelo en Python:
- entrena los dos ajustes con los dos códigos sobre la misma señal;
- compara cada uno por separado;
- verifica que los coeficientes en Q0.15 que salen del C sean los de los vectores de referencia.

Para que la comparación sea exacta, el compilador no debe fusionar multiplicaciones y sumas. El Makefile lo impide con `-ffp-contract=off`.

## Prueba en la placa

La placa arranca el Linux de Red Pitaya, en su versión 2.07, desde la tarjeta SD. Todo se maneja desde la computadora por la consola serie del puerto USB de la placa:
- `consola.py`, en `sw/hw_test/`, corre comandos en la placa y le pasa archivos por esa consola;
- los programas se compilan en la misma placa, que trae el compilador de C.

### Prueba con los vectores de referencia

1. El bitstream que deja `placa.tcl` se convierte con `bootgen` al formato que carga la placa.
2. En la computadora, `gen_vectores.py` arma `vectores.h` con los vectores de referencia, y `consola.py enviar` pasa a la placa `main.c`, `ip.c`, `ip.h`, `placa_mapa.h`, `vectores.h`, `correr.sh` y el bitstream.
3. En la placa, `correr.sh` (que se lanza desde la computadora con `consola.py cmd`) lleva el reloj de la lógica de 125 a 100 MHz, carga el bitstream, compila `prueba` y la corre. `prueba` carga los vectores de referencia en el reproductor y compara la salida del núcleo contra ellos. Con `sin_cargar` saltea el reloj y el bitstream.

### Prueba con la señal LTE

1. En la computadora, `lte_placa.py` prepara el registro de entrenamiento y la subtrama de evaluación.
2. En la placa, `entrenar_lte` entrena y pasa la subtrama por el núcleo en ocho trozos.
3. Después, el mismo `lte_placa.py` compara la salida con el modelo de punto fijo y mide con el amplificador simulado.

Con la opción `separar`, `lte_placa.py` aplica los mismos coeficientes en flotante. Es la comparación de la tabla de la prueba LTE. `lte_placa.py` mide la EVM con el ecualizador por subportadora. La de la tesis, con el receptor escalar, sale de `figuras/cap7_evm_escalar.py`.

### Lazo vuelta a vuelta

Está en `sw/lazo_simulado/`:
- **Del lado de la computadora:** lo maneja `lazo_lte.py`, con el back-off, el número de vueltas y el de muestras por pasada como opciones.
- **Del lado de la placa:** lo maneja `lazo_lte.c`.

En cada vuelta, el script:
1. le pide a la placa que aplique y que entrene;
2. pasa la salida por el amplificador simulado;
3. repite cada paso con el mismo código en la computadora, para comparar.

Con la opción `--sin-placa` corre todo en la computadora, lo que permite probar el lazo sin el hardware.

La prueba de reidentificación del trabajo futuro es `experimentos/reid_lte.py`, que usa ese mismo lazo en la computadora (ver `experimentos/README.md`).
