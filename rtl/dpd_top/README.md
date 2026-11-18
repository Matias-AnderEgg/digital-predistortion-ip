# El núcleo completo

`hdl/dpd_apply_top.sv` junta el camino de datos (`rtl/dpd_apply/`) con el
mapa de registros (`rtl/dpd_regs/`).

```bash
python rtl/dpd_top/run.py
```

Regenera el bloque de registros desde el `.rdl`, corre el lint del núcleo
entero y sus bancos de prueba.

## Qué agrega este nivel

- **El banco activo.** Es el que lee el camino de datos. Arranca en paso
  directo (`w[q=0,p=1]` al máximo positivo y el resto en cero), así que al
  salir del reset la cadena transmite sin predistorsión.
- **La copia del sombra al activo**, entera, con `COMMIT` y entre dos
  muestras (con `en` en alto). En el camino de datos paralelo los doce
  productos leen el banco en el mismo ciclo, así que una muestra nunca usa un
  juego mezclado.
- **El armado del `COMMIT`.** El sombra arranca en cero, y un `COMMIT` antes
  de la primera carga dejaría la salida en cero. Por eso `COMMIT` no hace nada
  hasta que el host escribe algún coeficiente.
- **El paso directo (`BYPASS`).** La entrada está en la escala `G` y la
  salida en `G_out` (8 y 128), así que copiar `x` a `u` subiría el nivel
  24 dB. El paso directo corre la muestra `LOG2_GOUT - LOG2_G` bits a la
  derecha con el mismo redondeo que la salida del camino de datos, y la
  retrasa `LATENCY` ciclos, así que conmutarlo no cambia el nivel ni mete un
  salto. La latencia sale de `rtl/common/hdl/mp_pkg.sv`, la misma que usa el
  camino de datos.
- **La escritura directa del banco activo** por los puertos `lazo_*`, que
  manda con `CTRL.COEF_SRC` en 1. Está prevista para un lazo de adaptación en
  la lógica, que no se construyó. En la placa esos puertos van en cero.

`dpd_apply_top` corta la elaboración si sus parámetros no coinciden con los
del mapa generado (`dpd_apply_regs_pkg`), porque el mapa tiene un registro
por coeficiente y `CONFIG` informa las escalas.

## Verificación

| grupo | qué comprueba |
|---|---|
| estado seguro | el banco activo arranca en paso directo, y la salida es la entrada cambiada de escala |
| doble banco | escribir el sombra no llega al camino de datos hasta el `COMMIT`, y la lectura devuelve el sombra |
| armado | un `COMMIT` antes de la primera carga queda ignorado, y cualquier escritura de coeficiente lo arma |
| entre muestras | con `en` bajo el `COMMIT` queda pendiente, `STATUS` lo informa y se aplica cuando `en` sube |
| paso directo | cambia de escala, queda dentro de un LSB del camino normal con el banco en paso directo y está alineado en latencia |
| escritura directa | con `COEF_SRC` en 1 los puertos `lazo_*` escriben el activo y el `COMMIT` no hace nada, con `COEF_SRC` en 0 no escriben |
| vectores de referencia | las 4096 muestras funcionales, con los coeficientes cargados por AXI-Lite |

El paso directo no da bit a bit lo mismo que el camino de datos con el banco
en paso directo, porque ese coeficiente vale `0x7FFF` y no uno exacto.
