`timescale 1ns/1ps

// Variante sin registros intermedios (ver README.md de este directorio):
// la latencia con un registro por etapa que multiplica.

package mp_pkg;

    function automatic int mp_latency(int M, int NORD, int ACC_PER_STAGE);
        int nterm  = M * NORD;
        int npow   = NORD - 1;
        int ngroup = (nterm + ACC_PER_STAGE - 1) / ACC_PER_STAGE;
        return 1
             + npow
             + 1
             + 1
             + ngroup
             + 1;
    endfunction

endpackage
