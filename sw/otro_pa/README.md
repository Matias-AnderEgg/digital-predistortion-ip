# El piso y la meseta del DPD

Análisis de por qué el DPD no llega más lejos en los dos extremos del
back-off: el piso a back-off alto, donde el PA es casi lineal, y la meseta
de la EVM en compresión. Entrenan con el mismo C que corre la placa
(`sw/dpd_train`, configuración por defecto) y miden sobre los 364 símbolos
LTE de la tesis, con la medición de `sw/hw_test/lte_placa.py`.

Además del PA de la tesis (el Wiener-Hammerstein de Ahluwalia, con la no
linealidad arctan), `otro_pa.py` usa en algunos puntos el de Gregorio,
González, Schmidt y Cousseau (Rapp con p = 2, `FILTROS_LIBRO` en
`fixed_point/pa.py`), que es casi lineal en un rango de back-off más amplio
y deja ver mejor el piso.

Todo necesita antes `make -C sw/dpd_train`. Cada corrida tarda unos minutos.

## El piso a back-off alto

```bash
python sw/otro_pa/otro_pa.py piso
python sw/otro_pa/otro_pa.py gradiente
python sw/otro_pa/otro_pa.py normalizacion
```

- `piso` compara tres DPD con la misma estructura (12 coeficientes,
  `w[0] = 1`) y el modelo del PA del C: el del C, el óptimo del costo que
  minimiza el GD del DPD, `|h0 x - modelo(DPD(x))|^2`, y el ajuste directo
  contra el PA real. Los dos óptimos se resuelven con `least_squares`, sólo
  como diagnóstico. Si el óptimo con el modelo es mejor que el C, el GD no
  llega. Si el ajuste contra el PA real es mejor que el óptimo con el modelo,
  el piso viene del modelo. Deja `resultados/piso.txt`.
- `gradiente` mide el gradiente medio del GD del DPD en el arranque, en el
  óptimo y donde queda el C. El C divide el gradiente de cada muestra por la
  energía de su regresor, que en los picos vale más de cien mil veces la
  mediana, así que su punto fijo es el mínimo de `sum |e|^2 / energía` y no
  el de `sum |e|^2`. Deja `resultados/gradiente.txt`.
- `normalizacion` compara esa normalización por muestra con una por bloque
  (dividir por la energía media de la época anterior), con `dla_py`, una
  copia en Python de `dpd_train_dla`. Por bloque se corrige el sesgo a
  back-off alto, pero en compresión el DPD no aprende o diverge. Deja
  `resultados/normalizacion.txt`.

## La meseta en compresión

```bash
python sys_with_acpr/piso_grilla.py
python sw/otro_pa/meseta_optimo.py
```

`sys_with_acpr/piso_grilla.py` hace el análisis de `piso` en toda la grilla
de back-off de la tesis, solo con el PA de la tesis, y deja
`sys_with_acpr/resultados_identificacion/piso_grilla.txt`: el C, el óptimo
del costo y los mismos 12 coeficientes ajustados contra el PA real.

`meseta_optimo.py` repite el óptimo del costo con el modelo del PA saturado:
su salida recortada al pico de la salida del PA en la identificación, o su
entrada recortada al pico de la entrada, o la entrada llevada a RMS 1, y con
distintos objetivos (`h0 x`, `h0 x` recortado, o una ganancia libre). Si el
óptimo con el modelo saturado no se acerca al ajuste contra el PA real, el
GD con ese modelo tampoco puede. Deja `resultados/meseta_optimo.txt`.

`experimentos/reid_lte.py` usa `dla_py` con la opción `techo`, que recorta
la señal deseada del DPD al pico de la salida del PA.
