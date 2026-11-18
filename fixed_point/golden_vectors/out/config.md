# Golden vectors — bloque de aplicación del DPD (Memory Polynomial)

Generado por `generate.py`. Verifica el datapath de INFERENCIA del DPD
(coeficientes ya entrenados y congelados), NO el lazo de entrenamiento GD.

## Convención de punto fijo

| Campo        | Formato | Rango                | Nota |
|--------------|---------|-----------------------|------|
| Señal de entrada | s16/15 (Q0.15) | [-1, 1)     | shift externo G=8 (x_phys = x_norm * 8) |
| Señal de salida  | s16/15 (Q0.15) | [-1, 1)     | shift externo G_out=128 (u_phys = u_norm * 128) |
| Coeficiente  | s16/15 (Q0.15) | [-1, 0.999969482...] | antes Q1.14; w[1,0] satura a 0.999969482 (no llega a 1.0 exacto) |
| Redondeo / overflow | around / saturate | | |

### Por qué la salida usa otra escala que la entrada

No tienen por qué ser iguales. El DPD expande los picos, y con G_out = G = 8
la salida recortaría. Como el corrimiento es potencia de dos, cambiarlo son
bits de alineación: cero compuertas.

Los números de esta sección son de la señal LTE, con el predistorsionador
del barrido de la tesis (M = 3, P = 7, BO -9 dB, entrenado en Python con el
paso del modelo limitado a 1 y 16000 muestras), no con los 12 códigos de
estos vectores. Los da `experimentos/analisis_salida_zf.py`, con la corrida copiada en su
encabezado, y son los del Apéndice B de la tesis. Se miden sobre la señal de evaluación (semilla 456,
364 símbolos, 1597440 muestras):

| `G_out` | recortan | error del núcleo contra el flotante | margen sobre el pico |
|---|---|---|---|
| 8 | 1 de cada 236 | -14.7 dB | -17.0 dB |
| **128** | **ninguna** | **-54.9 dB** | **7.1 dB** |

Con 8 el recorte domina el error del núcleo por unos 40 dB.

**El pico no converge.** Con 10 señales de 364 símbolos (semillas 456 a
465), el pico de la componente más grande de `u_phys` crece con la cantidad
de muestras observadas y no se estabiliza:

| muestras | pico | `u_norm` con G_out = 128 | margen |
|---|---|---|---|
| 16000 | 18.9 | 0.148 | 16.6 dB |
| 160000 | 33.4 | 0.261 | 11.7 dB |
| 1597440 | 56.7 | 0.443 | 7.1 dB |
| 15974400 (260 ms) | 110.8 | 0.866 | 1.3 dB |

Sube unos 10 dB cada dos décadas de muestras. El motivo es que el término
de orden 7 amplifica la cola de una señal OFDM, que es casi gaussiana y no
tiene pico máximo. **No existe un formato que garantice cero recorte**: lo
que se elige es una probabilidad de recorte, no un margen seguro. Con 128,
en los 16 millones de muestras no recorta ninguna.

Por eso el bloque **satura** en vez de dar la vuelta. Un pico raro que se
pase se recorta y degrada suave; con wrap aparecería con el signo cambiado,
que es catastrófico.

Con 128 el ruido de cuantización de la salida queda 61 dB por debajo
de la señal (-61.2 dB). La resolución sobra y el rango no, así que conviene
gastar bits de resolución en margen de pico.

### Margen de la entrada

Con `G = 8`, el pico de `x_norm` sobre 320k muestras de una OFDM de 52
portadoras llegó a 0.444, o sea 7.0 dB de margen, y baja con el tiempo de
observación. Con la señal LTE no se midió. La entrada viene de un conversor
y está acotada por él.

### Cómo se calcula la magnitud (sin raíz cuadrada)

`|x|^(p-1)` se calcula a partir de `|x|^2 = I^2 + Q^2`, guardado en el
acumulador ancho, y las potencias salen por multiplicaciones sucesivas por
`|x|^2`. Como todos los órdenes son impares, los exponentes de la magnitud
son pares y la raíz cuadrada no hace falta en ningún lado. Cuesta 0.02 dB
de precisión contra el camino con raíz y ahorra una unidad de raíz
cuadrada (CORDIC o tabla más Newton-Raphson) en el hardware.

En el RTL son, por tap: dos multiplicadores para `I^2` y `Q^2`, un sumador,
y dos multiplicadores más para `|x|^4` y `|x|^6`.

- Pmax = 7 (órdenes impares [1, 3, 5, 7]), M = 3 taps -> 12 coeficientes complejos.
- BO de diseño usado para entrenar: -9 dB (peor caso, mismo corner que justifica M=3/P=7).
- Orden de columnas de coeficientes (para w y para h): `for q in range(M): for p in orders:`
  i.e. col 0..3 = tap q=0 (p=1,3,5,7), col 4..7 = tap q=1, col 8..11 = tap q=2.
- w[1,0] (orden 1, tap 0) = 0.99996948 + 0.00000000j -> código 7FFF (I) — el
  1.0 matemático satura al máximo representable de Q0.15.

## Codificación de los .hex

Un valor por línea, entero de 16 bits en complemento a 2, 4 dígitos hex
mayúsculas (compatible con `$readmemh` de SystemVerilog). I y Q en archivos
separados, mismo índice de línea = misma muestra.

## Archivos

- `coeffs_w_re.hex` / `coeffs_w_im.hex` (12 líneas): coeficientes del DPD
  ya congelados en Q0.15 — los valores exactos que se escriben en
  W_COEFF_RE[12] / W_COEFF_IM[12] del mapa de registros.
- `coeffs_w.csv`: misma info en texto, con el float de entrenamiento al lado
  del valor cuantizado, para ver el error de cuantización por coeficiente.
- `functional_input_i/q.hex`, `functional_output_i/q.hex` (4096 líneas):
  el comienzo de la subtrama LTE de evaluación de la tesis (semilla 456,
  la de `sw/hw_test/lte_placa.py`) aplicado al MP congelado. Caso de uso
  representativo, para simulación de forma de onda / regresión.
- `golden_vectors_functional.csv`: misma info en texto (float físico + hex).
- `directed_input_i/q.hex`, `directed_output_i/q.hex` (52 líneas):
  casos de esquina. **Son una SECUENCIA, no muestras sueltas:** la línea de
  retardo de M=3 taps arrastra estado, así que el orden importa. Cada
  grupo empieza y termina vaciando la línea con 3 ceros (`flush_*`),
  para que la salida de cada caso sea interpretable por sí sola.

  Los casos están armados alrededor del pico REAL de la señal, no del fondo
  de escala del formato. Con fondo de escala la salida satura casi siempre,
  y un vector que satura no distingue nada: un RTL con un signo dado vuelta
  o una columna de coeficientes mal indexada satura al mismo valor y pasa
  igual. Los grupos son:

  | grupo | qué verifica |
  |---|---|
  | `zero`, `lsb_*` | que el cero da cero y que el bit menos significativo no se pierde |
  | `imp_*` | respuesta al impulso: una muestra sola y luego ceros, así la salida recorre los taps uno por uno. Verifica el indexado de taps y el orden de las columnas de coeficientes |
  | `ramp_*_de_8` | escalera de amplitud. La contribución de cada orden crece como una potencia distinta de la magnitud, así que barrer la amplitud separa los cuatro órdenes |
  | `alt_burst_*` | ráfaga alternada: ejercita los 3 taps a la vez con valores opuestos |
  | `sat_*` | saturación deliberada, al final para no contaminar lo anterior |
- `golden_vectors_directed.csv`: misma info, con etiqueta de caso por fila.

## Cómo se generó `u` (salida esperada)

`u_phys = apply_memory_poly_fxp(x_dpd, coeffs_q, Pmax=7, M=3, cfg)`,
que hace internamente: `x_norm = quantize(x_dpd/8)`, evalúa el MP con
acumulador ancho, y devuelve `u_norm * 128`. El valor que ve el registro
interno de salida del RTL (antes del shift externo) es
`u_norm = quantize(u_phys/128)`, que es lo que está en los .hex `*_output_*`.

Los `*_input_*` están en la escala de entrada y los `*_output_*` en la de
salida. El RTL no necesita saber ninguna de las dos: trabaja siempre con
códigos de 16 bits, y el cambio de escala entre la entrada y la salida es
el corrimiento fijo de `G_out/G = 16` que aplica dentro del bloque
antes de redondear al registro de salida.

## Reproducibilidad

Señal LTE de la tesis (`fixed_point/lte.py`, 16-QAM, FFT de 1024,
sobremuestreo 4, filtro de transmisión), con semillas fijas: las primeras
16000 muestras de la de semilla 123 para entrenar y la de semilla 456
para el vector funcional. El entrenamiento corre en
flotante, como en el ARM del SoC, y los coeficientes se pasan a Q0.15 al
final. Hiperparámetros GD: mu_h=0.001 sin NLMS, con respaldo NLMS
mu_h=0.05 si el modelo del PA diverge,
mu_w=0.5, ep_pa=40, ep_dpd=5, N_train_max=16000, w_reg_lambda=1e-4,
w_reg_pmin=3, y el paso efectivo del modelo del PA limitado a 1
(h_ganancia_max=1). Son los mismos 12 códigos que da el C con
`dpd_train_cfg_default` (prueba 3 de `sw/dpd_train/test/test_dpd_train.py`).

Correr `generate.py` de nuevo debería reproducir bit-exacto estos mismos
archivos.
