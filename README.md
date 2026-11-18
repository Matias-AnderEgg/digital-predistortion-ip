# IP de predistorsión digital

Proyecto Integrador de la carrera de Ingeniería en Telecomunicaciones del
Instituto Balseiro. Es un núcleo de predistorsión digital (DPD) para
linealizar un amplificador de potencia, con el modelo de polinomio con
memoria (M = 3, órdenes 1, 3, 5 y 7). El núcleo aplica el predistorsionador
en la lógica programable, y el entrenamiento por descenso de gradiente corre
en el ARM de una Zynq-7020 (Red Pitaya STEMlab 125-14). El amplificador se
simula, porque la placa no tiene un conversor rápido para manejar uno real.

Este repositorio tiene el código que acompaña a la tesis: los modelos, el
RTL, el software de la placa y los scripts de sus figuras y tablas.

## Directorios

| directorio | contenido |
|---|---|
| `sys_with_acpr/` | modelo de referencia en flotante (transmisor, amplificador, predistorsionador y receptor) y los scripts de los Caps. 2, 3 y 6 |
| `fixed_point/` | modelo de punto fijo del núcleo y generador de los vectores de referencia (`golden_vectors/`) |
| `rtl/common/` | los seis módulos básicos, cada uno con su banco de pruebas |
| `rtl/dpd_apply/` | camino de datos del polinomio con memoria |
| `rtl/dpd_regs/` | mapa de registros en SystemRDL y el RTL que sale de él |
| `rtl/dpd_top/` | el núcleo completo, con el doble banco y el paso directo |
| `rtl/hw_test/` | reproductor que alimenta al núcleo desde una memoria interna, para la prueba en la placa |
| `rtl/synth/` | scripts de síntesis e implementación para Vivado |
| `rtl/variantes/` | dos versiones del camino de datos que dan resultados de la tesis |
| `sw/dpd_train/` | entrenamiento en C que corre en el ARM |
| `sw/hw_test/` | programas de la placa y herramientas para comunicarse con ella |
| `sw/lazo_simulado/` | lazo vuelta a vuelta con el amplificador simulado |
| `sw/otro_pa/` | análisis del piso del predistorsionador a back-off alto |
| `figuras/` | los scripts que dibujan las figuras y escriben las tablas de la tesis |
| `experimentos/` | scripts que respaldan resultados puntuales de la tesis, ver su README |

## Dependencias

- Python 3 con NumPy, SciPy, Matplotlib y fxpmath.
- Verilator y cocotb, para los bancos de prueba del RTL.
- Vivado 2026.1, para la síntesis y la implementación.
- PeakRDL, opcional: el RTL del mapa de registros ya está generado en
  `rtl/dpd_regs/gen/`.
- Un compilador de C, para `sw/dpd_train`. Las comparaciones contra Python
  son exactas con `-ffp-contract=off`, que ya pone el Makefile.

## Pruebas

```bash
python rtl/common/run.py          # y lo mismo con dpd_apply, dpd_regs, dpd_top y hw_test
make -C sw/dpd_train test         # el C contra el modelo de Python
python fixed_point/golden_vectors/generate.py   # tiene que dejar out/ sin cambios
```

El Makefile usa `python3`. Con otro intérprete, `make -C sw/dpd_train PY=<ruta> test`.

## Reproducción

Cómo se vuelve a generar cada figura y cada tabla está en
`REPRODUCCION.md`. La organización del repositorio y las herramientas están
en el Apéndice D de la tesis.
