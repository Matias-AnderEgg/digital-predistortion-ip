# Diseno de bloques para la prueba en hardware sobre una Zynq-7020.
#
# Uso:
#   vivado -mode batch -source rtl/synth/placa.tcl
#   vivado -mode batch -source rtl/synth/placa.tcl -tclargs xc7z020clg400-1
#
# Arma, implementa y genera el bitstream de:
#
#   ARM (PS7) --GP0--> interconexion --+--> IP (banco de registros, AXI-Lite)
#                                       +--> BRAM de entrada (puerto A)
#                                       +--> BRAM de salida  (puerto A)
#                                       +--> AXI GPIO (control del reproductor)
#
#   BRAM de entrada (puerto B) --> reproductor --> IP --> BRAM de salida (B)
#
# Todo a 100 MHz desde FCLK_CLK0. No usa pines del PL: lo unico que sale del
# chip es lo del PS (DDR y MIO), asi que el mismo diseno sirve para cualquier
# placa con este encapsulado. La configuracion del PS queda en los valores
# por defecto (sin preset de placa): en la Red Pitaya el bitstream se carga
# desde su Linux, que ya configuro el PS (ver sw/hw_test/correr.sh).
#
# Usa flujo con proyecto porque el diseno de bloques lo necesita. El proyecto
# queda en synth/placa/ y se rehace de cero en cada corrida.
#
# Deja en synth/reportes/:
#   placa_tiempos.rpt, placa_utilizacion.rpt
#   placa.xsa        plataforma con el bitstream, para Vitis
#   placa_mapa.h     direcciones de cada periferico, para el programa del ARM

set AQUI   [file dirname [file normalize [info script]]]
set RTL    [file dirname $AQUI]
set SALIDA [file join $AQUI reportes]
set PROY   [file join $AQUI placa]

set PARTE "xc7z020clg400-1"
if {[llength $argv] >= 1} { set PARTE [lindex $argv 0] }

file mkdir $SALIDA
create_project placa $PROY -part $PARTE -force

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
    [file join $RTL hw_test hdl dpd_test_player.sv] \
    [file join $RTL hw_test hdl dpd_hw_wrap.v] \
]
add_files -norecurse $FUENTES
set_property file_type SystemVerilog [get_files *.sv]
update_compile_order -fileset sources_1

# La ultima version de cada IP que trae esta instalacion.
proc ip {nombre} { return [lindex [lsort [get_ipdefs -all "xilinx.com:ip:${nombre}:*"]] end] }

create_bd_design sistema

# --- ARM ----------------------------------------------------------------
set ps [create_bd_cell -type ip -vlnv [ip processing_system7] ps]
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "0"} $ps
set_property CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} $ps

# --- IP mas reproductor ---------------------------------------------------
set dpd [create_bd_cell -type module -reference dpd_hw_wrap dpd]

# --- BRAM de entrada y de salida -------------------------------------------
foreach nombre {in out} {
    set ctrl [create_bd_cell -type ip -vlnv [ip axi_bram_ctrl] bram_ctrl_$nombre]
    set_property -dict {CONFIG.SINGLE_PORT_BRAM {1} CONFIG.PROTOCOL {AXI4LITE}} $ctrl
    set mem [create_bd_cell -type ip -vlnv [ip blk_mem_gen] bram_$nombre]
    set_property -dict {
        CONFIG.Memory_Type {True_Dual_Port_RAM}
        CONFIG.Enable_B {Use_ENB_Pin}
        CONFIG.Use_RSTB_Pin {true}
        CONFIG.Register_PortA_Output_of_Memory_Primitives {false}
        CONFIG.Register_PortB_Output_of_Memory_Primitives {false}
    } $mem
    connect_bd_intf_net [get_bd_intf_pins bram_ctrl_$nombre/BRAM_PORTA] [get_bd_intf_pins bram_$nombre/BRAM_PORTA]
}
connect_bd_intf_net [get_bd_intf_pins dpd/BRAM_IN]  [get_bd_intf_pins bram_in/BRAM_PORTB]
connect_bd_intf_net [get_bd_intf_pins dpd/BRAM_OUT] [get_bd_intf_pins bram_out/BRAM_PORTB]

# --- GPIO de control: canal 1 sale (ctrl), canal 2 entra (estado) ----------
set gpio [create_bd_cell -type ip -vlnv [ip axi_gpio] gpio]
set_property -dict {
    CONFIG.C_IS_DUAL {1}
    CONFIG.C_ALL_OUTPUTS {1} CONFIG.C_GPIO_WIDTH {32}
    CONFIG.C_ALL_INPUTS_2 {1} CONFIG.C_GPIO2_WIDTH {32}
} $gpio
connect_bd_net [get_bd_pins gpio/gpio_io_o]  [get_bd_pins dpd/ctrl]
connect_bd_net [get_bd_pins dpd/estado]      [get_bd_pins gpio/gpio2_io_i]

# --- Buses AXI, relojes y resets: que los arme Vivado ----------------------
set primero 1
foreach esclavo {dpd/S_AXI bram_ctrl_in/S_AXI bram_ctrl_out/S_AXI gpio/S_AXI} {
    set intc [expr {$primero ? "New AXI Interconnect" : "Auto"}]
    apply_bd_automation -rule xilinx.com:bd_rule:axi4 -config [list \
        Clk_master {Auto} Clk_slave {Auto} Clk_xbar {Auto} \
        Master {/ps/M_AXI_GP0} Slave "/$esclavo" ddr_seg {Auto} \
        intc_ip $intc master_apm {0}] [get_bd_intf_pins $esclavo]
    set primero 0
}

# Por si la automatizacion no engancho el reset del envoltorio.
if {[llength [get_bd_nets -quiet -of_objects [get_bd_pins dpd/aresetn]]] == 0} {
    connect_bd_net [get_bd_pins dpd/aresetn] \
        [get_bd_pins -of_objects [get_bd_cells -filter {VLNV =~ *proc_sys_reset*}] -filter {NAME == peripheral_aresetn}]
}

# 32 KB por BRAM: 8192 muestras de 32 bits.
assign_bd_address
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    if {[string match "*bram_ctrl*" $seg]} { set_property range 32K $seg }
}

validate_bd_design
save_bd_design

set bd [get_files sistema.bd]
make_wrapper -files $bd -top
add_files -norecurse [file join $PROY placa.gen sources_1 bd sistema hdl sistema_wrapper.v]
set_property top sistema_wrapper [current_fileset]
update_compile_order -fileset sources_1

# --- Mapa de direcciones para el programa del ARM --------------------------
set fh [open [file join $SALIDA placa_mapa.h] w]
puts $fh "/* Generado por rtl/synth/placa.tcl. No editar a mano. */"
puts $fh "#ifndef PLACA_MAPA_H"
puts $fh "#define PLACA_MAPA_H"
foreach seg [get_bd_addr_segs -of_objects [get_bd_addr_spaces ps/Data]] {
    set nombre [string toupper [regsub -all {[^A-Za-z0-9]} [lindex [split $seg /] end] _]]
    puts $fh [format "#define %-28s 0x%08XU" "${nombre}_BASE" [get_property offset $seg]]
    puts $fh [format "#define %-28s 0x%08XU" "${nombre}_RANGO" [get_property range $seg]]
}
puts $fh "#endif"
close $fh

# --- Implementacion -------------------------------------------------------
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1
if {[get_property PROGRESS [get_runs impl_1]] != "100%"} {
    error "la implementacion no termino, ver $PROY/placa.runs/impl_1"
}

open_run impl_1
report_timing_summary -max_paths 10 -file [file join $SALIDA placa_tiempos.rpt]
report_utilization                   -file [file join $SALIDA placa_utilizacion.rpt]
write_hw_platform -fixed -include_bit -force [file join $SALIDA placa.xsa]

puts "\n=== resumen ==="
set util [report_utilization -return_string]
foreach {etiqueta fila} {DSP48 {DSPs} LUT {Slice LUTs} FF {Slice Registers} BRAM {Block RAM Tile}} {
    set n "?"
    regexp -line "^\\|\\s*${fila}\\*?\\s*\\|\\s*(\[0-9.\]+)" $util -> n
    puts [format "  %-6s %s" $etiqueta $n]
}
set wns [get_property SLACK [get_timing_paths -delay_type max]]
set whs [get_property SLACK [get_timing_paths -delay_type min]]
puts "  WNS $wns ns   WHS $whs ns   (100 MHz)"
puts "  mapa en [file join $SALIDA placa_mapa.h]"
