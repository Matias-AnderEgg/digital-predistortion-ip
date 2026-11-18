`timescale 1ns/1ps

// Wrapper de simulacion del bloque de redondeo y saturacion.
//
// Los parametros de un modulo se fijan al elaborar, asi que para probar
// varias combinaciones de anchos hay que instanciar el bloque varias veces.
// Cada instancia cubre un uso concreto del datapath del DPD.
module tb_fxp_round_sat;

    // A: salida de un multiplicador Q0.15. El producto de dos Q0.15 vive en
    //    Q0.30 y vuelve a Q0.15.
    logic signed [31:0] din_a;
    logic signed [15:0] dout_a;
    fxp_round_sat #(.IN_W(32), .IN_FRAC(30), .OUT_W(16), .OUT_FRAC(15))
        u_a (.din(din_a), .dout(dout_a));

    // B: salida del acumulador. El modelo de Python usa un acumulador
    //    s37/28 y la salida del bloque vuelve a Q0.15.
    logic signed [36:0] din_b;
    logic signed [15:0] dout_b;
    fxp_round_sat #(.IN_W(37), .IN_FRAC(28), .OUT_W(16), .OUT_FRAC(15))
        u_b (.din(din_b), .dout(dout_b));

    // C: mismo numero de bits fraccionarios en la entrada y en la salida.
    //    No hay nada que redondear, el bloque queda como pura saturacion.
    logic signed [19:0] din_c;
    logic signed [15:0] dout_c;
    fxp_round_sat #(.IN_W(20), .IN_FRAC(15), .OUT_W(16), .OUT_FRAC(15))
        u_c (.din(din_c), .dout(dout_c));

    // D: de s24/22 a s16/14, con bits enteros en los dos formatos.
    logic signed [23:0] din_d;
    logic signed [15:0] dout_d;
    fxp_round_sat #(.IN_W(24), .IN_FRAC(22), .OUT_W(16), .OUT_FRAC(14))
        u_d (.din(din_d), .dout(dout_d));

    // E: se descarta un solo bit fraccionario. Caso de borde: no hay bits
    //    de sticky, todo el redondeo lo decide el empate al par.
    logic signed [16:0] din_e;
    logic signed [15:0] dout_e;
    fxp_round_sat #(.IN_W(17), .IN_FRAC(16), .OUT_W(16), .OUT_FRAC(15))
        u_e (.din(din_e), .dout(dout_e));

endmodule
