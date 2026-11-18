# Bloques comunes

Los bloques aritméticos del camino de datos. Cada uno tiene su propio banco
de pruebas.

```bash
python rtl/common/run.py
```

Para cada bloque corre el lint (`verilator --lint-only -Wall`) y después su
banco de pruebas en cocotb, que termina con `TESTS=n PASS=n FAIL=0` y deja la
forma de onda en `tb/<toplevel>.vcd`. Los bancos comparan el RTL contra
`tb/fxp_ref.py`, un espejo en Python escrito con enteros, y contra la misma
cuenta hecha con fxpmath, la biblioteca del modelo de punto fijo. Los
bancos de los otros directorios importan el espejo desde acá.

Los parámetros de un módulo se fijan al elaborar, así que cada banco
(`tb/tb_<bloque>.sv`) instancia el bloque varias veces, una por combinación
de anchos que interesa probar.

## `fxp_round_sat`

Reduce el formato de punto fijo: redondeo al más cercano con empate al par
y saturación en los extremos del formato de salida. Es la función
`quantize` del modelo (fxpmath con `rounding='around'` y
`overflow='saturate'`), y aparece en cada punto donde un valor pasa a un
formato más angosto. Sólo saca bits fraccionarios: si la entrada tiene
menos que la salida, corta la elaboración.

El redondeo sale de tres bits:

    guard   el más significativo de los descartados (medio LSB)
    sticky  OR de los demás descartados
    lsb     el bit menos significativo del piso

    sube = guard AND (sticky OR lsb)

El banco prueba cinco combinaciones de anchos (de s32/30, s37/28, s20/15,
s24/22 y s17/16 a 16 bits), con los bordes del rango, los empates exactos y
sus vecinos, y valores al azar.

## `cmul_round_sat`

Multiplicador complejo con reducción de formato a la salida: el producto de
un coeficiente por un término de la base. Los productos parciales y las
sumas van al ancho completo y se redondea una sola vez, como el modelo, que
calcula el producto en doble precisión. Con un coeficiente de 16 bits y una
base de 37, la suma ocupa 53 bits, justo la mantisa de un double, así que el
modelo es exacto en este paso y el RTL coincide bit a bit.

El banco prueba el uso real (s16/15 por s37/28), todo en Q0.15, anchos
distintos en los tres lados, una salida angosta que redondea y satura a la
vez, y el uso real con los registros del camino de datos.

## `mp_magnitude`

Magnitud al cuadrado y sus potencias pares: `a2 = I^2 + Q^2`,
`a4 = a2*a2` y `a6 = a4*a2`, los `|x|^(p-1)` de los órdenes 3, 5 y 7. Como
los órdenes son impares, no hace falta raíz cuadrada.

`PIPE` es la cantidad de registros por potencia: uno después del redondeo y
los demás entre la multiplicación y el redondeo. Las potencias salen
alineadas entre sí. Con `PIPE = 0` es combinacional, y todos los modos dan
los mismos valores. Con la entrada en Q0.15, `a6 <= 8` y el acumulador llega
a ±256, así que estas etapas nunca saturan.

El banco prueba el uso real, una sola potencia, un acumulador angosto para
que la saturación sea alcanzable, una entrada más angosta y los modos con
registros. Además corre el bloque sobre 400 muestras de los vectores de
referencia contra la función `quantize` del modelo.

Cada etapa declara su señal dentro de su propio bloque `generate` y la
siguiente la lee por nombre jerárquico. Con un arreglo compartido entre
etapas, el lint de Verilator 5.052 termina con un error interno
(`V3DfgBreakCycles.cpp: Unhandled vertex type: MULS`).

## `mp_acc_step`

Un paso de la acumulación: `acc_out = sat(acc_in + (term << shift))`. El
corrimiento es el `G^(p-1)` de la convención normalizada, 0, 6, 12 y 18
bits con G = 8. Es combinacional, y `dpd_apply_core` encadena doce.

No redondea: correr a la izquierda no agrega bits fraccionarios, así que la
suma es exacta y sólo puede saturar. El término se corre a 55 bits
(`A_W + MAX_SHIFT`) antes de sumar, para no perder los bits altos.

El banco prueba el uso real, un acumulador angosto que satura seguido y el
caso sin corrimiento (G = 1).

## `mp_delay_line`

La memoria del polinomio: guarda las últimas M muestras, porque el término
del retardo q usa `x[n-q]`. Reset sincrónico activo en alto, y `en` congela
la línea.

El banco prueba el reset, el orden de los retardos con el mismo impulso que
los casos `imp_*` de los vectores de referencia, que `en` congele la línea y
una secuencia larga al azar contra una cola en Python, con M = 3, 1 y 5.

## `delay_reg`

Retarda un vector N ciclos, para alinear caminos de distinta profundidad.
Con `N = 0` es un cable, y el banco prueba ese caso y que `en` congele la
cadena.

## `mp_pkg`

La latencia del camino de datos (`mp_latency`) y la cantidad de registros
por multiplicación (`MUL_PIPE`), en un solo lugar, porque `dpd_apply_top` la
usa para alinear el paso directo.
