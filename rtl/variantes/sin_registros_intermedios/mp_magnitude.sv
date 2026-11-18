`timescale 1ns/1ps

// Variante sin registros intermedios (ver README.md de este directorio):
// mp_magnitude con un solo registro por potencia, despues del redondeo.

module mp_magnitude #(
    parameter int X_W    = 16,
    parameter int X_FRAC = 15,
    parameter int A_W    = 37,
    parameter int A_FRAC = 28,
    parameter int NPOW   = 3,
    parameter int PIPE   = 0
) (
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic signed [X_W-1:0]      x_re,
    input  logic signed [X_W-1:0]      x_im,

    output logic signed [NPOW*A_W-1:0] pot
);
    localparam int S_W    = 2*X_W + 1;
    localparam int S_FRAC = 2*X_FRAC;
    localparam int P_W    = 2*A_W;
    localparam int P_FRAC = 2*A_FRAC;

    logic signed [S_W-1:0] sq_re, sq_im, suma;
    logic signed [A_W-1:0] a2_c, a2, a2_sal;

    assign sq_re = x_re * x_re;
    assign sq_im = x_im * x_im;
    assign suma  = sq_re + sq_im;

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (S_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
    ) u_a2 (.din (suma), .dout (a2_c));

    delay_reg #(.W (A_W), .N (PIPE)) u_r_a2 (
        .clk (clk), .rst (rst), .en (en), .d (a2_c), .q (a2)
    );

    for (genvar k = 1; k < NPOW; k++) begin : g_pot
        logic signed [A_W-1:0] a2_al;
        delay_reg #(.W (A_W), .N (PIPE*(k-1))) u_al (
            .clk (clk), .rst (rst), .en (en), .d (a2), .q (a2_al)
        );

        logic signed [P_W-1:0] prod;
        if (k == 1) begin : g_desde_a2
            assign prod = a2_al * a2_al;
        end else begin : g_desde_anterior
            assign prod = g_pot[k-1].a_out * a2_al;
        end

        logic signed [A_W-1:0] a_c, a_out;
        fxp_round_sat #(
            .IN_W (P_W), .IN_FRAC (P_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
        ) u (.din (prod), .dout (a_c));

        delay_reg #(.W (A_W), .N (PIPE)) u_r (
            .clk (clk), .rst (rst), .en (en), .d (a_c), .q (a_out)
        );
    end

    delay_reg #(.W (A_W), .N (PIPE*(NPOW-1))) u_r_sal (
        .clk (clk), .rst (rst), .en (en), .d (a2), .q (a2_sal)
    );
    assign pot[A_W-1:0] = a2_sal;

    for (genvar k = 1; k < NPOW; k++) begin : g_sal
        logic signed [A_W-1:0] alineada;
        delay_reg #(.W (A_W), .N (PIPE*(NPOW-1-k))) u_r (
            .clk (clk), .rst (rst), .en (en),
            .d (g_pot[k].a_out), .q (alineada)
        );
        assign pot[(k+1)*A_W-1 -: A_W] = alineada;
    end
endmodule
