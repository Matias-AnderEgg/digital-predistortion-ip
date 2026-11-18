`timescale 1ns/1ps

// Un paso de acumulacion del Memory Polynomial: corrimiento por orden mas
// suma con saturacion.
//
//     acc_out = sat(acc_in + (term << shift))
//
// El corrimiento es el G^(p-1) de la convencion normalizada. Con G=8=2^3 son
// 0, 6, 12 y 18 bits para los ordenes 1, 3, 5 y 7, sin multiplicador.
//
// El termino se corre a un ancho mayor antes de sumar. Correrlo dentro del
// ancho del acumulador perderia los bits altos, y un termino que tiene que
// saturar daria un valor equivocado.
module mp_acc_step #(
    parameter int A_W      = 37,  // formato del acumulador
    parameter int A_FRAC   = 28,
    parameter int MAX_SHIFT = 18, // 3*(Pmax-1) para G=8 y Pmax=7
    // Ancho del puerto de corrimiento. El caso MAX_SHIFT=0 hay que tratarlo
    // aparte: $clog2(1) vale 0 y el rango quedaria [-1:0], que es invalido.
    localparam int SH_W = (MAX_SHIFT == 0) ? 1 : $clog2(MAX_SHIFT + 1)
) (
    input  logic signed [A_W-1:0] acc_in_re,
    input  logic signed [A_W-1:0] acc_in_im,
    input  logic signed [A_W-1:0] term_re,
    input  logic signed [A_W-1:0] term_im,
    input  logic [SH_W-1:0] shift,
    output logic signed [A_W-1:0] acc_out_re,
    output logic signed [A_W-1:0] acc_out_im
);
    localparam int W_SH = A_W + MAX_SHIFT;  // termino corrido, sin perdida
    localparam int W_S  = W_SH + 1;         // + el acumulador

    logic signed [W_SH-1:0] sh_re, sh_im;
    logic signed [W_S-1:0]  s_re, s_im;

    assign sh_re = W_SH'(term_re) <<< shift;
    assign sh_im = W_SH'(term_im) <<< shift;

    assign s_re = W_S'(sh_re) + W_S'(acc_in_re);
    assign s_im = W_S'(sh_im) + W_S'(acc_in_im);

    // Mismos bits fraccionarios a la entrada y a la salida, asi que el
    // bloque de reduccion queda como pura saturacion, sin redondeo.
    fxp_round_sat #(
        .IN_W (W_S), .IN_FRAC (A_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
    ) u_re (.din (s_re), .dout (acc_out_re));

    fxp_round_sat #(
        .IN_W (W_S), .IN_FRAC (A_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
    ) u_im (.din (s_im), .dout (acc_out_im));
endmodule
