# Entrenamiento del DPD en el ARM

Los dos lazos de gradiente de `fixed_point/dpd.py`, en C, para correr en el
ARM de la Zynq:

1. `dpd_train_pa_model` identifica el modelo del PA, `h`, por LMS
   precondicionado a partir de la entrada `x` y la salida del PA `y`.
2. `dpd_train_dla` entrena el predistorsionador, `w`, por aprendizaje
   directo con el gradiente de Wirtinger, usando `h` como modelo del PA.
3. `dpd_coef_to_q15` pasa `w` a los códigos que se escriben en `W_COEFF`.

`dpd_train_cfg_default` llena la configuración por defecto, que es la de la
tesis.

## Criterios

- Todo en doble precisión, porque el Cortex-A9 tiene FPU. Sólo `w` se pasa
  a Q0.15, que es lo que va a la lógica.
- Dimensiones fijas (`M = 3`, órdenes 1, 3, 5 y 7), las del núcleo, sin
  memoria dinámica.
- La misma convención normalizada que el núcleo: la señal entra dividida por
  `G = 8` y cada orden lleva su `G^(p-1)`, así que los coeficientes que salen
  son los que espera el camino de datos.
- El código sigue a `fixed_point/dpd.py` con los mismos nombres, para poder
  compararlos.

## Verificación

```bash
make -C sw/dpd_train test       # 16000 muestras, 40 épocas
make -C sw/dpd_train rapido     # 2000 muestras, 5 épocas
python sw/dpd_train/test/escala_g.py
```

`make test` entrena cada lazo con el C y con el modelo de Python en
flotante, sobre la misma señal, y los compara por separado. La diferencia
es del orden del último bit del double (2,5e-15 en `h` y 3,6e-16 en `w`),
porque numpy suma en otro orden. Después entrena la cadena completa en C y
comprueba que los 12 códigos Q0.15 sean los de los vectores de referencia.

`escala_g.py` corre el mismo C con `G = 1` y con `G = 8` y comprueba que
`h` y `w` salgan iguales.

El Makefile compila con `-ffp-contract=off`: si el compilador fusiona
multiplicación y suma, redondea distinto que Python. El Cortex-A9 no tiene
FMA en doble, así que en la placa no cambia nada.

## Modelo del PA

- LMS con precondicionador diagonal, `mu_h = 0,001`, sin NLMS por muestra,
  40 épocas y 16000 muestras.
- `h` arranca con `h0` en la ganancia lineal de `x` a `y`, `<x, y> / <x, x>`,
  y el resto en cero. La salida del PA va sin normalizar y su ganancia baja
  con el back-off. Con `h0 = 1`, lo que queda del arranque en las direcciones
  que la señal casi no excita pesa cada vez más, y el modelo empeora con el
  back-off.
- `h_ganancia_max = 1`: si en una muestra el paso efectivo (`mu_h` por la
  energía del regresor precondicionado) pasa de 1, se lo achica a 1. Actúa
  sólo en los picos, donde sin el límite el modelo diverge con algunos
  registros.
- Respaldo: si la guarda de divergencia corta el lazo, se reentrena desde
  cero con NLMS y `mu_h_respaldo = 0,05`. `dpd_train_pa_model` devuelve en
  ese caso las épocas con signo negativo.
- El precondicionador divide por la potencia de cada columna más
  `precond_delta / G^2`. La potencia se mide sobre la señal dividida por
  `G`, así que el delta se escala igual para que el resultado no dependa de
  `G`.

## Predistorsionador

- `w[0]` queda fijo en 1, para que no pueda elegir atenuar.
- El gradiente de cada muestra se divide por la energía de su regresor
  (`use_nlms_w = 1`). Eso estabiliza el entrenamiento en compresión. Como en
  los picos esa energía vale cientos de miles de veces la mediana, el GD casi
  no ve los picos, y a back-off alto queda un piso (ver `sw/otro_pa/`).
- Fuga `w_reg_lambda` sobre los órdenes altos y recorte del gradiente
  (`grad_clip`).

## Otras opciones

- `warm_start = 1`: cada lazo arranca del `h` o del `w` que se le pasa. Lo
  usa el lazo vuelta a vuelta (`sw/lazo_simulado/lazo_lte.c`).
- `h_umbral_pico`: no actualiza `h` en las muestras cuyo regresor pasa de esa
  amplitud. Vale 0, apagado.
- `quantize_h`: predice con `h` redondeado a Q0.15. Vale 0, apagado.
