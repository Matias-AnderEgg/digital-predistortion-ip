# Sintesis fuera de contexto del IP de aplicacion del DPD.
#
# Uso:
#   vivado -mode batch -source rtl/synth/sintesis.tcl
#   vivado -mode batch -source rtl/synth/sintesis.tcl -tclargs xc7a100tcsg324-1 10.0
#
# Dos argumentos opcionales: el dispositivo y el periodo de reloj en ns. Sin
# argumentos, la Zynq-7020 de la placa y 10 ns.
#
# Flujo sin proyecto: se corre entero de cero cada vez y todo lo que produce
# son reportes y el checkpoint que usa implementacion.tcl.

set AQUI  [file dirname [file normalize [info script]]]
set RTL   [file dirname $AQUI]
set SALIDA [file join $AQUI reportes]

set PARTE   "xc7z020clg400-1"     ;# la Zynq-7020 de la Red Pitaya, la de placa.tcl
set PERIODO 10.0
if {[llength $argv] >= 1} { set PARTE   [lindex $argv 0] }
if {[llength $argv] >= 2} { set PERIODO [lindex $argv 1] }

file mkdir $SALIDA

# El orden importa: los paquetes antes que quien los usa.
set FUENTES [list \
    [file join $RTL common hdl mp_pkg.sv] \
    [file join $RTL common hdl fxp_round_sat.sv] \
    [file join $RTL common hdl delay_reg.sv] \
    [file join $RTL common hdl cmul_round_sat.sv] \
    [file join $RTL common hdl mp_magnitude.sv] \
    [file join $RTL common hdl mp_acc_step.sv] \
    [file join $RTL common hdl mp_delay_line.sv] \
    [file join $RTL dpd_apply hdl dpd_apply_core.sv] \
    [file join $RTL dpd_regs gen dpd_apply_regs_pkg.sv] \
    [file join $RTL dpd_regs gen dpd_apply_regs.sv] \
    [file join $RTL dpd_top hdl dpd_apply_top.sv] \
]

puts "=== dispositivo: $PARTE, periodo: $PERIODO ns ==="

foreach f $FUENTES {
    if {![file exists $f]} {
        error "no existe $f. Si falta el bloque de registros, correr rtl/dpd_regs/run.py"
    }
    read_verilog -sv $f
}

# El reloj va ANTES de sintetizar, no despues: si se declara despues, la
# herramienta ya optimizo sin ningun objetivo de tiempo y el reporte mide algo
# que nadie trato de cumplir. Como el periodo es parametro, el XDC se escribe
# en cada corrida en vez de ser un archivo fijo.
#
# No se declaran retardos de entrada y salida, porque dependen de donde se
# instancie el IP. El reporte mide los caminos internos.
set XDC [file join $SALIDA reloj.xdc]
#
# HD.CLK_SRC dice de que buffer global sale el reloj. Fuera de contexto no
# hay ningun buffer, y sin esto Vivado no puede estimar la demora del reloj.
set fh [open $XDC w]
puts $fh "create_clock -name clk -period $PERIODO \[get_ports clk\]"
puts $fh "set_property HD.CLK_SRC BUFGCTRL_X0Y0 \[get_ports clk\]"
close $fh
read_xdc $XDC

# Fuera de contexto: no inserta buffers de entrada y salida ni pide asignacion
# de pines, como corresponde a un IP que se instancia dentro de otro diseno.
synth_design -top dpd_apply_top -part $PARTE -mode out_of_context

report_utilization            -file [file join $SALIDA utilizacion.rpt]
report_utilization -hierarchical \
                              -file [file join $SALIDA utilizacion_jerarquica.rpt]
report_timing_summary -max_paths 10 \
                              -file [file join $SALIDA tiempos.rpt]
# Un camino por destino: sin -unique_pins los 20 son el mismo camino
# terminando en los 20 bits de un mismo registro.
report_timing -max_paths 20 -nworst 1 -unique_pins -sort_by slack \
                              -file [file join $SALIDA caminos.rpt]

# El diseno sintetizado, para poder analizarlo sin volver a sintetizar.
write_checkpoint -force [file join $SALIDA sintetizado.dcp]

# Resumen por consola, para no tener que abrir los reportes para lo basico.
# Se lee del reporte de utilizacion y no contando celdas, para que de los
# mismos numeros que el reporte.
puts "\n=== resumen ==="
set util [report_utilization -return_string]
foreach {etiqueta fila} {DSP48 {DSPs} LUT {Slice LUTs} FF {Slice Registers} BRAM {Block RAM Tile}} {
    set n "?"
    regexp -line "^\\|\\s*${fila}\\*?\\s*\\|\\s*(\[0-9.\]+)" $util -> n
    puts [format "  %-6s %s" $etiqueta $n]
}

set wns [get_property SLACK [get_timing_paths -delay_type max]]
puts "  camino critico (WNS): $wns ns  con periodo $PERIODO ns"
puts "\nreportes en $SALIDA"
