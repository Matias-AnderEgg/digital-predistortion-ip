# Mapa de registros AXI-Lite

Por acá entran los coeficientes y los bits de control del núcleo.

El mapa está en `rdl/dpd_apply_regs.rdl`, en SystemRDL. De ahí sale, con
PeakRDL, el RTL de `gen/` y la documentación HTML. El RTL generado no se
edita a mano.

```bash
python rtl/dpd_regs/run.py
```

Regenera `gen/`, imprime el mapa, corre el lint y los bancos de prueba. El
HTML queda en `gen/html/index.html` y no se versiona. Los dos `.sv` de
`gen/` sí, para que el núcleo se pueda simular y sintetizar sin PeakRDL.

## El mapa

    0x00-0x03  ID           'DPD1' en ASCII, constante
    0x04-0x07  CONFIG       M, NORD, LOG2_G y LOG2_GOUT de esta instancia
    0x08-0x0b  CTRL         BYPASS, COEF_SRC y COMMIT
    0x0c-0x0f  STATUS       COMMIT_PENDING
    0x40-0x6f  W_COEFF[12]  banco sombra, IM[31:16] y RE[15:0]

`CONFIG` informa los parámetros de la instancia, para que el software los
lea en vez de asumirlos. Cada coeficiente ocupa un registro de 32 bits, las
dos componentes juntas, así que se carga con una sola escritura. El índice
es `col = q*NORD + j`, el mismo orden de `dpd_apply_core` y de los `.hex`.
El hueco entre `STATUS` y `W_COEFF` deja lugar para más registros de control
sin mover los coeficientes.

Los `W_COEFF` son el banco sombra. El banco activo, el paso directo y la
copia de un banco al otro están en `rtl/dpd_top/`. Este bloque le da a
`dpd_apply_top` los valores del sombra, el pulso de `COMMIT`, los bits de
control y un `swmod` por coeficiente que avisa cuando el host lo escribió.

## Verificación

Los bancos de prueba verifican que el mapa sea el que se quiere:

- `ID` y `CONFIG` tienen las constantes de esta instancia;
- lo escrito en el sombra se lee igual, por el bus y por el lado del hardware;
- el strobe de byte separa las componentes;
- `swmod` pulsa en el coeficiente escrito y en ningún otro;
- `COMMIT` es un pulso de un ciclo;
- `BYPASS` y `COEF_SRC` son estado y salen al hardware;
- `STATUS` lo maneja el hardware y el host no lo puede escribir;
- un acceso fuera del mapa contesta `SLVERR`;
- el reset limpia el sombra;
- ninguna dirección se solapa con otra.

El maestro AXI-Lite está escrito a mano en `tb/axil.py`, sin
`cocotbext-axi`.
