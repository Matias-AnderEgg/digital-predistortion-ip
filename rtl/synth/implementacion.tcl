# Implementacion fuera de contexto del IP de aplicacion del DPD.
#
# Uso, despues de sintesis.tcl:
#   vivado -mode batch -source rtl/synth/implementacion.tcl
#
# Parte del diseno que guarda sintesis.tcl, con el mismo dispositivo y el
# mismo reloj. La sintesis estima el ruteo; los tiempos de verdad salen de
# aca, con el diseno ubicado y ruteado.
#
# No corre phys_opt_design a proposito: lo que interesa es si el RTL cierra
# tiempos por si mismo, no si la herramienta lo puede rescatar.

set AQUI   [file dirname [file normalize [info script]]]
set SALIDA [file join $AQUI reportes]

open_checkpoint [file join $SALIDA sintetizado.dcp]

opt_design
place_design
route_design

report_utilization    -file [file join $SALIDA impl_utilizacion.rpt]
report_timing_summary -max_paths 10 \
                      -file [file join $SALIDA impl_tiempos.rpt]
report_timing -max_paths 20 -nworst 1 -unique_pins -sort_by slack \
                      -file [file join $SALIDA impl_caminos.rpt]
write_checkpoint -force [file join $SALIDA implementado.dcp]

puts "\n=== implementacion ==="
puts "  setup (WNS): [get_property SLACK [get_timing_paths -delay_type max]] ns"
puts "  hold  (WHS): [get_property SLACK [get_timing_paths -delay_type min]] ns"
puts "\nreportes en $SALIDA"
