`timescale 1ns/1ps

// Variante sin registros intermedios (ver README.md de este directorio):
// dpd_apply_core con un registro por etapa que multiplica, despues del
// redondeo.

module dpd_apply_core #(
    parameter int X_W       = 16,
    parameter int X_FRAC    = 15,
    parameter int C_W       = 16,
    parameter int C_FRAC    = 15,
    parameter int A_W       = 37,
    parameter int A_FRAC    = 28,
    parameter int M         = 3,
    parameter int NORD      = 4,
    parameter int LOG2_G    = 3,
    parameter int LOG2_GOUT = 7,
    parameter int ACC_PER_STAGE = 2
) (
    input  logic clk,
    input  logic rst,
    input  logic en,

    input  logic signed [X_W-1:0] x_re,
    input  logic signed [X_W-1:0] x_im,

    input  logic [M-1:0][NORD-1:0][C_W-1:0] w_re,
    input  logic [M-1:0][NORD-1:0][C_W-1:0] w_im,

    output logic signed [X_W-1:0] u_re,
    output logic signed [X_W-1:0] u_im,
    output logic                  valid
);
    localparam int NTERM     = M * NORD;
    localparam int NPOW      = NORD - 1;
    localparam int MAX_SHIFT = LOG2_G * 2 * NPOW;
    localparam int SH_W      = (MAX_SHIFT == 0) ? 1 : $clog2(MAX_SHIFT + 1);
    localparam int B_W       = X_W + A_W;
    localparam int B_FRAC    = X_FRAC + A_FRAC;
    localparam int LATENCY   = mp_pkg::mp_latency(M, NORD, ACC_PER_STAGE);

    logic [M*X_W-1:0] tap_re, tap_im;

    mp_delay_line #(.W (X_W), .M (M)) u_delay (
        .clk (clk), .rst (rst), .en (en),
        .x_re (x_re), .x_im (x_im),
        .tap_re (tap_re), .tap_im (tap_im)
    );

    for (genvar q = 0; q < M; q++) begin : g_tap
        logic signed [X_W-1:0] xq_re, xq_im;
        assign xq_re = $signed(tap_re[(q+1)*X_W-1 -: X_W]);
        assign xq_im = $signed(tap_im[(q+1)*X_W-1 -: X_W]);

        logic signed [NPOW*A_W-1:0] pot;
        mp_magnitude #(
            .X_W (X_W), .X_FRAC (X_FRAC), .A_W (A_W), .A_FRAC (A_FRAC),
            .NPOW (NPOW), .PIPE (1)
        ) u_mag (
            .clk (clk), .rst (rst), .en (en),
            .x_re (xq_re), .x_im (xq_im), .pot (pot)
        );

        logic [X_W-1:0] xal_re, xal_im;
        delay_reg #(.W (X_W), .N (NPOW)) u_dxr (
            .clk (clk), .rst (rst), .en (en), .d (xq_re), .q (xal_re));
        delay_reg #(.W (X_W), .N (NPOW)) u_dxi (
            .clk (clk), .rst (rst), .en (en), .d (xq_im), .q (xal_im));

        for (genvar j = 0; j < NORD; j++) begin : g_ord
            logic signed [A_W-1:0] base_c_re, base_c_im;

            if (j == 0) begin : g_lineal
                assign base_c_re = A_W'($signed(xal_re)) <<< (A_FRAC - X_FRAC);
                assign base_c_im = A_W'($signed(xal_im)) <<< (A_FRAC - X_FRAC);
            end else begin : g_no_lineal
                logic signed [A_W-1:0] bm;
                assign bm = $signed(pot[j*A_W-1 -: A_W]);

                logic signed [B_W-1:0] bx_re, bx_im;
                assign bx_re = $signed(xal_re) * bm;
                assign bx_im = $signed(xal_im) * bm;

                fxp_round_sat #(
                    .IN_W (B_W), .IN_FRAC (B_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
                ) u_br (.din (bx_re), .dout (base_c_re));

                fxp_round_sat #(
                    .IN_W (B_W), .IN_FRAC (B_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
                ) u_bi (.din (bx_im), .dout (base_c_im));
            end

            logic signed [A_W-1:0] base_re, base_im;
            delay_reg #(.W (A_W), .N (1)) u_rbr (
                .clk (clk), .rst (rst), .en (en), .d (base_c_re), .q (base_re));
            delay_reg #(.W (A_W), .N (1)) u_rbi (
                .clk (clk), .rst (rst), .en (en), .d (base_c_im), .q (base_im));

            logic signed [C_W-1:0] wc_re, wc_im;
            assign wc_re = $signed(w_re[q][j]);
            assign wc_im = $signed(w_im[q][j]);

            logic signed [A_W-1:0] prod_c_re, prod_c_im;
            cmul_round_sat #(
                .A_W (C_W), .A_FRAC (C_FRAC), .B_W (A_W), .B_FRAC (A_FRAC),
                .O_W (A_W), .O_FRAC (A_FRAC)
            ) u_cmul (
                .a_re (wc_re), .a_im (wc_im),
                .b_re (base_re), .b_im (base_im),
                .o_re (prod_c_re), .o_im (prod_c_im)
            );

            logic signed [A_W-1:0] prod_re, prod_im;
            delay_reg #(.W (A_W), .N (1)) u_rpr (
                .clk (clk), .rst (rst), .en (en), .d (prod_c_re), .q (prod_re));
            delay_reg #(.W (A_W), .N (1)) u_rpi (
                .clk (clk), .rst (rst), .en (en), .d (prod_c_im), .q (prod_im));
        end
    end

    for (genvar c = 0; c < NTERM; c++) begin : g_acc
        localparam int Q_IDX = c / NORD;
        localparam int J_IDX = c % NORD;
        localparam int SHIFT = LOG2_G * 2 * J_IDX;
        localparam int GRP   = c / ACC_PER_STAGE;
        localparam int CIERRA = ((c % ACC_PER_STAGE) == (ACC_PER_STAGE-1)
                                 || c == NTERM-1) ? 1 : 0;

        logic [A_W-1:0] term_re, term_im;
        delay_reg #(.W (A_W), .N (GRP)) u_dtr (
            .clk (clk), .rst (rst), .en (en),
            .d (g_tap[Q_IDX].g_ord[J_IDX].prod_re), .q (term_re));
        delay_reg #(.W (A_W), .N (GRP)) u_dti (
            .clk (clk), .rst (rst), .en (en),
            .d (g_tap[Q_IDX].g_ord[J_IDX].prod_im), .q (term_im));

        logic signed [A_W-1:0] prev_re, prev_im;
        if (c == 0) begin : g_primero
            assign prev_re = '0;
            assign prev_im = '0;
        end else begin : g_resto
            assign prev_re = g_acc[c-1].acc_re;
            assign prev_im = g_acc[c-1].acc_im;
        end

        logic signed [A_W-1:0] acc_c_re, acc_c_im;
        mp_acc_step #(
            .A_W (A_W), .A_FRAC (A_FRAC), .MAX_SHIFT (MAX_SHIFT)
        ) u_acc (
            .acc_in_re (prev_re), .acc_in_im (prev_im),
            .term_re ($signed(term_re)), .term_im ($signed(term_im)),
            .shift (SH_W'(SHIFT)),
            .acc_out_re (acc_c_re), .acc_out_im (acc_c_im)
        );

        logic [A_W-1:0] acc_re, acc_im;
        delay_reg #(.W (A_W), .N (CIERRA)) u_rar (
            .clk (clk), .rst (rst), .en (en), .d (acc_c_re), .q (acc_re));
        delay_reg #(.W (A_W), .N (CIERRA)) u_rai (
            .clk (clk), .rst (rst), .en (en), .d (acc_c_im), .q (acc_im));
    end

    localparam int OUT_IN_FRAC = A_FRAC + (LOG2_GOUT - LOG2_G);

    logic signed [X_W-1:0] u_c_re, u_c_im;

    fxp_round_sat #(
        .IN_W (A_W), .IN_FRAC (OUT_IN_FRAC), .OUT_W (X_W), .OUT_FRAC (X_FRAC)
    ) u_out_re (.din ($signed(g_acc[NTERM-1].acc_re)), .dout (u_c_re));

    fxp_round_sat #(
        .IN_W (A_W), .IN_FRAC (OUT_IN_FRAC), .OUT_W (X_W), .OUT_FRAC (X_FRAC)
    ) u_out_im (.din ($signed(g_acc[NTERM-1].acc_im)), .dout (u_c_im));

    delay_reg #(.W (X_W), .N (1)) u_rout_re (
        .clk (clk), .rst (rst), .en (en), .d (u_c_re), .q (u_re));
    delay_reg #(.W (X_W), .N (1)) u_rout_im (
        .clk (clk), .rst (rst), .en (en), .d (u_c_im), .q (u_im));

    logic [LATENCY-1:0] valid_sr;
    logic               lleno, en_d1;

    always_ff @(posedge clk) begin
        if (rst)      valid_sr <= '0;
        else if (en)  valid_sr <= {valid_sr[LATENCY-2:0], 1'b1};
    end

    always_ff @(posedge clk) begin
        if (rst) en_d1 <= 1'b0;
        else     en_d1 <= en;
    end

    assign lleno = valid_sr[LATENCY-1];
    assign valid = lleno & en_d1;
endmodule
