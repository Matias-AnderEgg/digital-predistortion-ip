# Variantes del núcleo

Versiones del camino de datos que dan resultados de la tesis pero no son el
núcleo que se probó en la placa. Dan bit a bit lo mismo que el núcleo con
los vectores de referencia: cambian los recursos y los tiempos, no lo que se
calcula.

Para usar una, se copian sus archivos sobre los del núcleo, con el mismo
nombre, y se corren los mismos scripts:

```bash
cp rtl/variantes/potencias_compartidas/dpd_apply_core.sv rtl/dpd_apply/hdl/
python rtl/dpd_apply/run.py
python rtl/dpd_top/run.py
vivado -mode batch -source rtl/synth/sintesis.tcl
vivado -mode batch -source rtl/synth/implementacion.tcl
git checkout rtl/                    # vuelve al núcleo
```

## `potencias_compartidas/`

`dpd_apply_core.sv` calcula la magnitud y sus potencias una sola vez, sobre
la muestra del retardo 0, y las lleva a los retardos 1 y 2 por una cadena
de registros, en lugar de calcularlas en cada rama. En la Zynq-7020,
implementada fuera de contexto a 10 ns, usa 142 DSP48 (contra 162), 6149
LUT y 8193 flip-flops, y cierra con 0,31 ns de margen.

## `sin_registros_intermedios/`

La primera fila de la tabla del cierre de tiempos (`rtl/synth/README.md`):
cada etapa que multiplica tiene un solo registro, después del redondeo, así
que la multiplicación y el redondeo quedan en el mismo ciclo, y la
acumulación suma dos términos por etapa. La latencia es de 13 ciclos y no
cierra 100 MHz. Cambia cinco archivos: `mp_pkg.sv`, `cmul_round_sat.sv` y
`mp_magnitude.sv` de `rtl/common/hdl/`, `dpd_apply_core.sv` y
`dpd_apply_top.sv`. Con esta variante, `cmul_round_sat.sv` no tiene el
parámetro `PIPE`, así que los bancos de `rtl/common/` no corren: se prueba
con los de `rtl/dpd_apply/` y `rtl/dpd_top/`.
