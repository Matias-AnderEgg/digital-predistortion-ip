# Síntesis e implementación

Scripts de Vivado 2026.1, en modo por lotes. Vivado no corre en macOS, así
que se ejecutan en una máquina Windows o Linux.

| script | qué hace |
|---|---|
| `sintesis.tcl` | sintetiza el núcleo solo, fuera de contexto, sin proyecto |
| `implementacion.tcl` | ubica y rutea lo que dejó `sintesis.tcl`, con el mismo dispositivo y reloj |
| `placa.tcl` | arma el sistema de la placa (procesador, núcleo, reproductor, dos BRAM y un AXI GPIO), lo implementa y genera el bitstream y `placa_mapa.h` |
| `tabla_recursos.py` | lee los reportes y escribe las tablas de la tesis en `resultados/` |

```bash
vivado -mode batch -source rtl/synth/sintesis.tcl
vivado -mode batch -source rtl/synth/implementacion.tcl
vivado -mode batch -source rtl/synth/placa.tcl
python rtl/synth/tabla_recursos.py
```

Sin argumentos usan la Zynq-7020 de la placa (`xc7z020clg400-1`) y 10 ns.
`sintesis.tcl` acepta el dispositivo y el período en ns como argumentos,
por ejemplo `-tclargs xc7a100tcsg324-1 10.0` para una Artix-7 100T.

Los reportes quedan en `reportes/`. Sólo se versionan los cinco que lee
`tabla_recursos.py`. El RTL del mapa de registros está en
`../dpd_regs/gen/`, así que no hace falta PeakRDL para sintetizar.

## Tablas de la tesis

`tabla_recursos.py` deja en `resultados/`:

- `recursos.tex`: LUT, flip-flops, DSP48 y BRAM del núcleo solo
  (`implementacion.tcl`) y del sistema de la placa (`placa.tcl`), con el
  margen a 10 ns y la frecuencia máxima que da ese margen.
- `dsp.tex`: de dónde salen los 162 DSP48, sumando la jerarquía del núcleo
  sintetizado: 30 de las potencias de la magnitud (10 por rama), 36 de la base
  `x |x|^(p-1)` (4 por cada uno de los 9 términos no lineales) y 96 del
  producto por el coeficiente (8 por cada uno de los 12). El script verifica
  que la suma dé 162.

| | Núcleo solo | Sistema de la placa |
|---|---|---|
| LUT | 6724 (12,6 %) | 7357 (13,8 %) |
| Flip-flops | 9267 (8,7 %) | 10564 (9,9 %) |
| DSP48E1 | 162 (73,6 %) | 162 (73,6 %) |
| BRAM de 36 kb | 0 | 16 (11,4 %), las del reproductor |
| Margen a 10 ns | 0,21 ns, 102,2 MHz | 0,11 ns, 101,1 MHz |

El peor camino del núcleo solo es el de la potencia `a6`, un DSP48 seguido
de una suma en lógica.

## Cierre de tiempos

La tabla del cierre de tiempos de la tesis compara tres versiones del RTL:

- sin registros intermedios (latencia 13): la variante de
  `rtl/variantes/sin_registros_intermedios/`;
- con `MUL_PIPE = 2` (latencia 24): el núcleo con `MUL_PIPE = 2` en
  `rtl/common/hdl/mp_pkg.sv`;
- con `MUL_PIPE = 3` (latencia 29): el núcleo tal cual.

Los márgenes de la tabla son de `sintesis.tcl` sin la línea que fija
`HD.CLK_SRC`, que estima la demora del reloj fuera de contexto. Con esa
línea los números cambian un poco. Margen de la síntesis a 10 ns:

| registros | latencia | Zynq-7020 | Artix-7 100T |
|---|---|---|---|
| ninguno intermedio | 13 | −7,48 ns | −7,06 ns |
| `MUL_PIPE = 2` | 24 | −0,75 ns | −0,35 ns |
| `MUL_PIPE = 3` | 29 | +0,38 ns | +0,47 ns |

## Potencias compartidas

La variante que calcula las potencias una sola vez para las tres ramas está
en `rtl/variantes/potencias_compartidas/` y se sintetiza con los mismos
scripts (ver `rtl/variantes/README.md`). En la Zynq-7020 usa 142 DSP48, 6149
LUT y 8193 flip-flops, y cierra 10 ns con 0,31 ns de margen.
