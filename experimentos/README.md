# Experimentos

Scripts que respaldan resultados puntuales de la tesis. Cada uno se corre desde la raíz del repositorio, con `PYTHONDONTWRITEBYTECODE=1` para
no dejar `__pycache__`. Los dos que usan el C de `sw/dpd_train` necesitan
antes `make -C sw/dpd_train`.

| script | qué respalda | resultado |
|---|---|---|
| `analisis_salida_zf.py` | Apéndice B y Sec. 7.2.3: la escala de salida del núcleo con la señal LTE. Con `G_out = 8` recorta una de cada 236 muestras, con 128 quedan 7,1 dB de margen y en 16 millones de muestras no recorta ninguna | lo imprime, y la corrida está copiada en su encabezado |
| `analisis_dsp.py` | Compromisos entre recursos y desempeño: la cuenta de en cuántos DSP48 se reparte cada multiplicación según el ancho interno, 174 con 37 bits, 84 con 25 y 78 con 18 | lo imprime |
| `convergencia_saturacion.py` | Sec. 6.2.1: tener en cuenta la saturación no hace útil el error del entrenamiento del predistorsionador. Con el objetivo del limitador ideal sigue explotando en compresión, y recortando además la salida del modelo indica que el predistorsionador empeora la señal (de −9 a −2 dB a −9 dB) cuando con el PA real la mejora (de −11 a −24 dB) | `resultados/convergencia_saturacion.txt` |
| `formato_interno.py` | Sec. 7.2.3: de dónde sale el formato interno Q8.28. Los 28 bits fraccionarios hacen falta (con 25 el error interno sube de −55 a −42 dB respecto de la salida) y de los 8 enteros alcanza con 3, porque la cuenta más grande, una suma parcial del acumulador, llega a 7,4 a −9 dB | `resultados/formato_interno.txt` |
| `h_q15_lte.py` | Sec. 7.2.6: entrenar el modelo del PA en Q0.15 empeora la EVM (6,8 contra 3,7 % con el ecualizador por subportadora que usa el script; con el receptor escalar de la tesis, 7,1 contra 4,5 %, que calcula `figuras/cap7_evm_escalar.py`) y el ACPR adyacente (−28,2 contra −31,2 dB) | `resultados/h_q15_lte.txt` |
| `reid_lte.py` | Trabajo futuro: en una prueba preliminar, reidentificar el modelo del PA con la señal predistorsionada empeoró el resultado | `resultados/reid_lte.txt`, y un CSV por back-off y caso que no se versiona |

```bash
PYTHONDONTWRITEBYTECODE=1 python experimentos/analisis_dsp.py         # al instante
PYTHONDONTWRITEBYTECODE=1 python experimentos/analisis_salida_zf.py   # unos 3 min
PYTHONDONTWRITEBYTECODE=1 python experimentos/formato_interno.py      # 1 min
PYTHONDONTWRITEBYTECODE=1 python experimentos/convergencia_saturacion.py  # 1 min
PYTHONDONTWRITEBYTECODE=1 python experimentos/h_q15_lte.py            # unos 3 min
PYTHONDONTWRITEBYTECODE=1 python experimentos/reid_lte.py             # unos 6 min, los 12 casos en paralelo
```

`h_q15_lte.py` deja además `salida/h_q15_lte.png`, que no se versiona.
