# Lazo vuelta a vuelta con el PA simulado

La cadena completa funcionando vuelta a vuelta: la placa aplica el DPD en el
núcleo y lo entrena en el ARM, y la computadora hace de PA. La placa no
tiene un conversor rápido para manejar un PA real, así que la salida del PA
se simula y viaja por la consola serie.

| archivo | dónde corre | qué hace |
|---|---|---|
| `lazo_lte.c` | placa | un paso por llamada: `inicio`, `aplicar k`, `modelo n`, `entrenar k n` |
| `lazo_lte.py` | computadora | maneja las vueltas, simula el PA y mide |

Usa `sw/hw_test/ip.c` para el núcleo, `sw/hw_test/consola.py` para la
consola y `sw/dpd_train` para entrenar. El bitstream es el de `sw/hw_test` y
tiene que estar cargado (`correr.sh` lo carga).

## Cómo funciona

1. En la vuelta 0, sin DPD, el ARM identifica el modelo del PA desde cero
   con las primeras `--modelo` muestras (16000) de la primera subtrama, como
   el entrenamiento fuera de línea.
2. En cada vuelta k, la placa pasa por el núcleo la subtrama k de la señal
   de evaluación (14 símbolos, 61440 muestras, semilla 456), en 8 trozos
   como `sw/hw_test/entrenar_lte.c`, y devuelve la salida.
3. La computadora pasa esa salida por el PA simulado, con ganancia fija, y
   mide NMSE, EVM con el ecualizador por subportadora y ACPR, como
   `lte_placa.py`.
4. El ARM hace una pasada del DPD sobre las primeras `--muestras` muestras
   de la subtrama, partiendo de los coeficientes vigentes (`warm_start`).
   Después calcula el RMS `c` de la salida del DPD sobre la subtrama entera
   y carga en el núcleo `w / c`, para que el PA reciba la misma potencia
   media que sin DPD. Como `c` sale de la subtrama anterior, la potencia
   varía unas décimas de dB de una vuelta a otra.

Con la placa, la computadora repite cada paso con el mismo C (por ctypes) y
el modelo de Python del núcleo, y comprueba en cada vuelta que la salida
del núcleo y los 12 códigos que entrenó el ARM sean los mismos.

## Uso

Después de `make -C sw/dpd_train` (la biblioteca de la computadora no se
recompila sola):

```bash
python lazo_lte.py --bo -9 --muestras 2000               # con la placa
python lazo_lte.py --sin-placa --bo -9 --muestras 2000   # todo en la computadora
```

`--bo` es el back-off del PA en la convención del proyecto
(`fixed_point/pa.py`). Deja `resultados/lte_bo<BO>_n<muestras>/vueltas.csv`,
con una fila por vuelta, y una línea por vuelta en la terminal. La primera
corrida con la placa le manda las subtramas (unos 4 MB, 400 s). Después
cada vuelta tarda unos 30 s.

## Resultados de la tesis

- `resultados/lte_bo-9_n2000/` y `lte_bo3_n2000/`: las corridas con la
  placa, 15 vueltas con pasadas de 2000 muestras (`corrida.txt` es lo que
  imprimió la terminal). En todas las vueltas la salida del núcleo y los
  coeficientes dieron igual que en la computadora. Las leen
  `figuras/figuras_placa.py` y `cap7_evm_escalar.py`.
- `resultados/largo_pasada/`: el mismo lazo con `--sin-placa` y pasadas de
  400, 2000 y 8000 muestras, para elegir el largo de la pasada.
