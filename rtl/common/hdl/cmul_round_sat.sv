`timescale 1ns/1ps

// Multiplicador complejo con reduccion de formato a la salida. Es el producto
// w[q,p] * base del Memory Polynomial, uno por termino.
//
// (a_re + j*a_im) * (b_re + j*b_im)
//     parte real       = a_re*b_re - a_im*b_im
//     parte imaginaria = a_re*b_im + a_im*b_re
//
// Los productos parciales y las sumas van al ancho completo y se redondea una
// sola vez al final, igual que el modelo de Python. Con un coeficiente de 16
// bits y una base de 37 la suma ocupa 53 bits, que entran en la mantisa de un
// double, asi que el modelo es exacto y el RTL coincide bit a bit.
//
// PIPE registros entre la suma y el redondeo, 0 es combinacional. En el
// datapath va con mp_pkg::MUL_PIPE-1.
module cmul_round_sat #(
    parameter int A_W    = 16,   // coeficiente
    parameter int A_FRAC = 15,
    parameter int B_W    = 37,   // base (formato del acumulador)
    parameter int B_FRAC = 28,
    parameter int O_W    = 37,   // salida
    parameter int O_FRAC = 28,
    parameter int PIPE   = 0
) (
    input  logic clk,            // solo se usan con PIPE>0
    input  logic rst,
    input  logic en,

    input  logic signed [A_W-1:0] a_re,
    input  logic signed [A_W-1:0] a_im,
    input  logic signed [B_W-1:0] b_re,
    input  logic signed [B_W-1:0] b_im,
    output logic signed [O_W-1:0] o_re,
    output logic signed [O_W-1:0] o_im
);
    localparam int P_W    = A_W + B_W;          // producto parcial, sin perdida
    localparam int P_FRAC = A_FRAC + B_FRAC;    // sus bits fraccionarios
    localparam int S_W    = P_W + 1;            // +1: suma/resta de dos parciales

    logic signed [P_W-1:0] p_rr, p_ii, p_ri, p_ir;
    logic signed [S_W-1:0] s_re, s_im, s_re_r, s_im_r;

    // Los operandos se extienden con signo a P_W antes de multiplicar,
    // porque el destino tiene ese ancho. El producto sale completo.
    assign p_rr = a_re * b_re;
    assign p_ii = a_im * b_im;
    assign p_ri = a_re * b_im;
    assign p_ir = a_im * b_re;

    assign s_re = S_W'(p_rr) - S_W'(p_ii);
    assign s_im = S_W'(p_ri) + S_W'(p_ir);

    delay_reg #(.W (S_W), .N (PIPE)) u_rre (
        .clk (clk), .rst (rst), .en (en), .d (s_re), .q (s_re_r));
    delay_reg #(.W (S_W), .N (PIPE)) u_rim (
        .clk (clk), .rst (rst), .en (en), .d (s_im), .q (s_im_r));

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (P_FRAC), .OUT_W (O_W), .OUT_FRAC (O_FRAC)
    ) u_re (.din (s_re_r), .dout (o_re));

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (P_FRAC), .OUT_W (O_W), .OUT_FRAC (O_FRAC)
    ) u_im (.din (s_im_r), .dout (o_im));
endmodule
