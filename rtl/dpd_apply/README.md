# Camino de datos del DPD

El polinomio con memoria con los coeficientes fijos:

    u[n] = sum_q sum_p  w[q,p] * x[n-q] * |x[n-q]|^(p-1) * G^(p-1)

con q en 0..M-1 y p en {1, 3, 5, 7}: doce términos complejos.

```bash
python rtl/dpd_apply/run.py
```

## Estructura

`hdl/dpd_apply_core.sv` es paralelo: una muestra de salida por ciclo.

    S0    línea de retardo             x[n-q]
    S1    a2 = I^2 + Q^2
    S2    a4 = a2*a2
    S3    a6 = a4*a2
    S4    base = x * |x|^(p-1)
    S5    prod = w * base
    S6+   acumulación, ACC_PER_STAGE términos por etapa
    últ.  cambio de escala y vuelta a Q0.15

S1 a S5 multiplican, y cada una ocupa `mp_pkg::MUL_PIPE` = 3 ciclos: dos
registros entre la multiplicación y el redondeo, y uno después. Con
`ACC_PER_STAGE = 1` la latencia es de 29 ciclos. `en` funciona como
habilitación de reloj de todo el camino de datos: si baja, se congela
entero.

| pieza | qué hace |
|---|---|
| `mp_delay_line` | guarda x[n], x[n-1] y x[n-2] |
| `delay_reg` | alinea los caminos de distinta profundidad |
| `mp_magnitude` | por retardo: a2, a4 y a6 |
| `fxp_round_sat` | base = x * a2^k, reducida al formato del acumulador |
| `cmul_round_sat` | prod = w[q,p] * base |
| `mp_acc_step` | suma cada término con su corrimiento G^(p-1) |
| `fxp_round_sat` | salida: cambio de escala y vuelta a Q0.15 |

El orden de los coeficientes es `col = q*NORD + j`, el mismo del modelo y de
`coeffs_w_re.hex` y `coeffs_w_im.hex`.

La acumulación es una cadena y no un árbol porque el modelo satura después
de cada suma. En los casos dirigidos las sumas parciales pasan de 5000
contra un límite de 256, y sumar en otro orden daría otro resultado.

El acumulador está en la escala de entrada (G = 8) y la salida en la de
salida (G_out = 128). Pasar de una a otra es multiplicar por una potencia de
dos, y el bloque de reducción final lo hace leyendo el acumulador con 4 bits
fraccionarios de más.

## Verificación

`tb/test_dpd_apply_core.py` alimenta los vectores de referencia de
`fixed_point/golden_vectors/out/` y exige coincidencia bit a bit:

- 52 casos dirigidos: impulsos, escalera de amplitud, ráfaga alternada y
  saturación deliberada;
- 4096 muestras de la señal LTE de evaluación.

Los dos son una secuencia: la línea de retardo arrastra estado, así que se
alimentan en orden y con la línea en cero al empezar. La comparación no
depende de la latencia: se toman las salidas de los ciclos con `valid` en
alto.

`run.py` corre la verificación con `ACC_PER_STAGE` en 1, 2, 3, 4 y 12, y las
cinco tienen que dar lo mismo. Si alguna difiere, una alineación depende del
parámetro.

El banco detecta errores en el diseño: con las columnas de coeficientes
transpuestas, con el corrimiento por orden a la mitad o con M = 2, falla en
el primer caso dirigido de impulso.
