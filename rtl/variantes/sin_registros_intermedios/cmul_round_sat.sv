`timescale 1ns/1ps

// Variante sin registros intermedios (ver README.md de este directorio):
// cmul_round_sat sin el parametro PIPE, la multiplicacion y el redondeo en
// el mismo ciclo.

module cmul_round_sat #(
    parameter int A_W    = 16,
    parameter int A_FRAC = 15,
    parameter int B_W    = 37,
    parameter int B_FRAC = 28,
    parameter int O_W    = 37,
    parameter int O_FRAC = 28
) (
    input  logic signed [A_W-1:0] a_re,
    input  logic signed [A_W-1:0] a_im,
    input  logic signed [B_W-1:0] b_re,
    input  logic signed [B_W-1:0] b_im,
    output logic signed [O_W-1:0] o_re,
    output logic signed [O_W-1:0] o_im
);
    localparam int P_W    = A_W + B_W;
    localparam int P_FRAC = A_FRAC + B_FRAC;
    localparam int S_W    = P_W + 1;

    logic signed [P_W-1:0] p_rr, p_ii, p_ri, p_ir;
    logic signed [S_W-1:0] s_re, s_im;

    assign p_rr = a_re * b_re;
    assign p_ii = a_im * b_im;
    assign p_ri = a_re * b_im;
    assign p_ir = a_im * b_re;

    assign s_re = S_W'(p_rr) - S_W'(p_ii);
    assign s_im = S_W'(p_ri) + S_W'(p_ir);

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (P_FRAC), .OUT_W (O_W), .OUT_FRAC (O_FRAC)
    ) u_re (.din (s_re), .dout (o_re));

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (P_FRAC), .OUT_W (O_W), .OUT_FRAC (O_FRAC)
    ) u_im (.din (s_im), .dout (o_im));
endmodule
