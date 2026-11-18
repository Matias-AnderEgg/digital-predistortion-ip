# RTL del IP de DPD

Implementación en SystemVerilog del núcleo de predistorsión digital: el
polinomio con memoria con M = 3 y órdenes 1, 3, 5 y 7, en paralelo, una
muestra por ciclo a 100 MHz, con un mapa de registros AXI-Lite para cargar
los coeficientes.

## Estructura

| directorio | contenido |
|---|---|
| `common/` | los bloques aritméticos, cada uno con su banco de pruebas |
| `dpd_apply/` | el camino de datos del polinomio con memoria, armado con los de `common/` |
| `dpd_regs/` | el mapa de registros en SystemRDL y el RTL que genera PeakRDL |
| `dpd_top/` | el núcleo completo: camino de datos, registros, doble banco y paso directo |
| `hw_test/` | el reproductor que alimenta al núcleo desde una BRAM, para la prueba en la placa |
| `synth/` | scripts de síntesis e implementación para Vivado |
| `variantes/` | versiones del camino de datos que dan resultados de la tesis: potencias compartidas y sin registros intermedios |

Cada directorio tiene su README, y los del núcleo su `run.py`, que corre el
lint de Verilator y los bancos de prueba de cocotb.

```bash
python rtl/common/run.py
python rtl/dpd_apply/run.py
python rtl/dpd_regs/run.py
python rtl/dpd_top/run.py
python rtl/hw_test/run.py
```

## Interfaz

El núcleo consume una muestra por pulso de `en` y no tiene `ready`, así que
no se lo puede frenar desde afuera. Alcanza cuando la fuente y el destino van
a ritmo fijo, como en la placa. Para colgarlo de algo que necesite frenar el
flujo, como un DMA a memoria, hay que agregar control de flujo.

Los módulos son parametrizables (M, órdenes, anchos de palabra). El mapa de
registros se genera con M, el número de órdenes y las dos escalas fijas, y
`dpd_apply_top` corta la elaboración si se lo instancia con otros valores.

Las multiplicaciones de la cadena de potencias son de 37 por 37 bits y las
de la base de 37 por 16, y ninguna entra en un solo DSP48. Por eso las 78
multiplicaciones reales del camino de datos ocupan 162 DSP48 (ver
`synth/README.md`).
