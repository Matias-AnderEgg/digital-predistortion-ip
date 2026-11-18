`timescale 1ns/1ps

// Definiciones compartidas del Memory Polynomial. La latencia del datapath
// esta escrita aca una sola vez porque dpd_apply_top la usa para alinear el
// bypass, y una copia desactualizada lo dejaria corrido sin aviso.
package mp_pkg;

    // Registros por cada etapa que multiplica: MUL_PIPE-1 entre la
    // multiplicacion y el redondeo, y uno despues del redondeo. Con 2 no
    // alcanza, porque las multiplicaciones de 37 por 37 terminan en una suma
    // en logica comun que Vivado deja en el ciclo del redondeo. Con 3 cierra
    // 100 MHz en la Artix-7 100T y en la Zynq-7020.
    localparam int MUL_PIPE = 3;

    // Ciclos entre una muestra valida en la entrada y su resultado en la
    // salida de dpd_apply_core. Las etapas son las que documenta ese modulo.
    function automatic int mp_latency(int M, int NORD, int ACC_PER_STAGE);
        int nterm  = M * NORD;
        int npow   = NORD - 1;                                   // a2, a4, a6
        int ngroup = (nterm + ACC_PER_STAGE - 1) / ACC_PER_STAGE;
        return 1                // linea de retardo
             + npow * MUL_PIPE  // magnitud y potencias
             + MUL_PIPE         // base
             + MUL_PIPE         // producto por el coeficiente
             + ngroup           // acumulacion
             + 1;               // reduccion de salida
    endfunction

endpackage
