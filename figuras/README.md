# Figuras y tablas de la tesis

Los scripts que dibujan las figuras y escriben las tablas de la tesis. No
corren simulaciones largas: leen los resultados que dejan los scripts de
`sys_with_acpr/`, `sw/` y `rtl/synth/`, o calculan en el momento con los
mismos módulos del modelo. Dejan cada figura en PDF, y cada tabla en LaTeX,
en este mismo directorio. Esas salidas no se versionan.

| script | qué deja | de dónde lee |
|---|---|---|
| `genera_figuras.py` | las figuras de Matplotlib de los Caps. 3 a 6, la lista está en `FIGS` | `sys_with_acpr` y los resultados de sus scripts (`resultados_barrido/`, `resultados_efecto_pa/`, `resultados_validacion/`, `resultados_desempeno/`, `resultados_identificacion/`, `resultados_robustez/`) |
| `tablas_barrido.py` | `tabla_barrido_evm.tex`, `tabla_barrido_evm_escalar.tex` y `tabla_barrido_acpr.tex`, del Apéndice A | `sys_with_acpr/resultados_barrido/barrido_mp.pkl` |
| `cuantizacion_zf.py` | `cuantizacion_psd.pdf` y `cuantizacion_const.pdf`, del Cap. 7 | el modelo de punto fijo de `fixed_point/` y `sys_with_acpr` |
| `figuras_placa.py` | `placa_psd.pdf`, `placa_amam.pdf`, `lazo_m9_nmse.pdf` y `lazo_m9_psd.pdf`, de la validación en la placa | `sw/hw_test/lte/` y `sw/lazo_simulado/resultados/lte_*` |
| `cap7_evm_escalar.py` | imprime la EVM del Cap. 7 con el receptor escalar | las salidas de la placa y del lazo |
| `gd_superficie_gen.py` | `gd_superficie.tex`, la superficie del costo para la figura del descenso por gradiente | nada, lo calcula |

```bash
python figuras/genera_figuras.py                          # todas
python figuras/genera_figuras.py --only conversion,desempeno_bo
python figuras/tablas_barrido.py
python figuras/gd_superficie_gen.py
PYTHONDONTWRITEBYTECODE=1 python figuras/cuantizacion_zf.py
PYTHONDONTWRITEBYTECODE=1 python figuras/figuras_placa.py
PYTHONDONTWRITEBYTECODE=1 python figuras/cap7_evm_escalar.py
```

`figuras_placa.py` y `cap7_evm_escalar.py` necesitan
`sw/hw_test/lte/esperada_bo-9.npz` y `esperada_bo3.npz`, que no se
versionan. Los dejan `python sw/hw_test/lte_placa.py preparar --bo -9` y
`--bo 3`, después de `make -C sw/dpd_train`. `cap7_evm_escalar.py` repite
el lazo en la computadora y comprueba que el NMSE de cada vuelta sea el de
la placa.
