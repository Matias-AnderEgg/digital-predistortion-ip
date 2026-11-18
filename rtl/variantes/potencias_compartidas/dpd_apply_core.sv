`timescale 1ns/1ps

// Bloque de aplicacion del DPD: Memory Polynomial. Variante con las
// potencias de la magnitud calculadas una sola vez (ver README.md de este
// directorio).
//
//   u[n] = sum_q sum_p  w[q,p] * x[n-q] * |x[n-q]|^(p-1) * G^(p-1)
//
// con q en 0..M-1 y p en {1, 3, 5, 7}. Doce terminos complejos. Lo verifican
// los golden vectors de fixed_point/golden_vectors/out, con los mismos
// formatos, el mismo orden de coeficientes y la misma convencion de escalas.
//
// Etapas:
//   S0   linea de retardo             x[n-q]
//   S1   a2 = I^2 + Q^2
//   S2   a4 = a2*a2
//   S3   a6 = a4*a2
//   S4   base = x * |x|^(p-1)
//   S5   prod = w * base
//   S6+  acumulacion, ACC_PER_STAGE terminos por etapa
//   ult. cambio de escala y vuelta a Q0.15
//
// S1 a S5 multiplican, y cada una ocupa mp_pkg::MUL_PIPE ciclos. La
// acumulacion es una cadena que suma en el mismo orden que el modelo.
//
// La entrada esta en la escala G y la salida en G_out, porque el DPD expande
// los picos. El cambio de escala solo mueve la coma binaria del acumulador.
//
// LATENCY ciclos entre una muestra valida en la entrada y su resultado en la
// salida, 29 con los valores por defecto.
module dpd_apply_core #(
    parameter int X_W       = 16,  // muestra de entrada y de salida
    parameter int X_FRAC    = 15,
    parameter int C_W       = 16,  // coeficiente
    parameter int C_FRAC    = 15,
    parameter int A_W       = 37,  // acumulador y datapath interno
    parameter int A_FRAC    = 28,
    parameter int M         = 3,   // taps de memoria
    parameter int NORD      = 4,   // ordenes impares: 1, 3, 5, 7
    parameter int LOG2_G    = 3,   // G = 2^3 = 8   (escala de entrada)
    parameter int LOG2_GOUT = 7,   // G_out = 2^7 = 128 (escala de salida)
    // Terminos que se acumulan entre registro y registro. Con 2 no cierra
    // 100 MHz en la Artix-7 100T.
    parameter int ACC_PER_STAGE = 1
) (
    input  logic clk,
    input  logic rst,
    input  logic en,                       // hay muestra valida en x

    input  logic signed [X_W-1:0] x_re,
    input  logic signed [X_W-1:0] x_im,

    // Coeficientes indexados [tap][orden]. Como el arreglo es empaquetado, el
    // puerto es un vector plano de M*NORD*C_W bits con w[q][j] en los bits
    // (q*NORD+j)*C_W, el mismo orden que coeffs_w_re.hex y coeffs_w_im.hex.
    // No se declara signed porque el signo no pasa a cada elemento, asi que
    // se convierte en el punto de uso.
    input  logic [M-1:0][NORD-1:0][C_W-1:0] w_re,
    input  logic [M-1:0][NORD-1:0][C_W-1:0] w_im,

    output logic signed [X_W-1:0] u_re,
    output logic signed [X_W-1:0] u_im,
    output logic                  valid
);
    // `en` funciona como habilitacion de reloj del pipeline: si baja,
    // se congela entero y no se pierde ni se duplica ninguna muestra.
    localparam int NTERM     = M * NORD;
    localparam int NPOW      = NORD - 1;          // a2, a4, a6 (el orden 1 no usa)
    localparam int MAX_SHIFT = LOG2_G * 2 * NPOW; // 3*(Pmax-1) = 18
    localparam int SH_W      = (MAX_SHIFT == 0) ? 1 : $clog2(MAX_SHIFT + 1);
    // Producto muestra por potencia de la magnitud, sin perdida.
    localparam int B_W       = X_W + A_W;
    localparam int B_FRAC    = X_FRAC + A_FRAC;
    // Ciclos entre una muestra en la entrada y su resultado en la salida.
    localparam int LATENCY   = mp_pkg::mp_latency(M, NORD, ACC_PER_STAGE);

    // S0. Linea de retardo: x[n], x[n-1], ... x[n-M+1]
    logic [M*X_W-1:0] tap_re, tap_im;

    mp_delay_line #(.W (X_W), .M (M)) u_delay (
        .clk (clk), .rst (rst), .en (en),
        .x_re (x_re), .x_im (x_im),
        .tap_re (tap_re), .tap_im (tap_im)
    );

    // S1..S3. Magnitud y potencias una sola vez, sobre el tap 0.
    // |x[n-q]|^2 es |x[n]|^2 retrasada q muestras, asi que las potencias de
    // los otros taps bajan por su propia cadena de registros.
    logic signed [NPOW*A_W-1:0] pot0;
    mp_magnitude #(
        .X_W (X_W), .X_FRAC (X_FRAC), .A_W (A_W), .A_FRAC (A_FRAC),
        .NPOW (NPOW), .PIPE (mp_pkg::MUL_PIPE)
    ) u_mag (
        .clk (clk), .rst (rst), .en (en),
        .x_re ($signed(tap_re[X_W-1:0])), .x_im ($signed(tap_im[X_W-1:0])),
        .pot (pot0)
    );

    // Por cada tap: base (S4) y producto por el coeficiente (S5).
    for (genvar q = 0; q < M; q++) begin : g_tap
        logic signed [X_W-1:0] xq_re, xq_im;
        assign xq_re = $signed(tap_re[(q+1)*X_W-1 -: X_W]);
        assign xq_im = $signed(tap_im[(q+1)*X_W-1 -: X_W]);

        // Las potencias de este tap son las del anterior una muestra despues.
        logic [NPOW*A_W-1:0] pot;
        if (q == 0) begin : g_pot_directo
            assign pot = pot0;
        end else begin : g_pot_retrasado
            delay_reg #(.W (NPOW*A_W), .N (1)) u_dpot (
                .clk (clk), .rst (rst), .en (en),
                .d (g_tap[q-1].pot), .q (pot));
        end

        // La muestra tiene que llegar a la etapa de la base junto con sus
        // potencias.
        localparam int POT_LAT = NPOW * mp_pkg::MUL_PIPE;
        logic [X_W-1:0] xal_re, xal_im;
        delay_reg #(.W (X_W), .N (POT_LAT)) u_dxr (
            .clk (clk), .rst (rst), .en (en), .d (xq_re), .q (xal_re));
        delay_reg #(.W (X_W), .N (POT_LAT)) u_dxi (
            .clk (clk), .rst (rst), .en (en), .d (xq_im), .q (xal_im));

        for (genvar j = 0; j < NORD; j++) begin : g_ord
            logic signed [A_W-1:0] base_c_re, base_c_im;

            if (j == 0) begin : g_lineal
                // Orden 1: la base es la muestra. Pasarla al formato del
                // acumulador es un corrimiento exacto, y se retrasa lo mismo
                // que el producto de los otros ordenes.
                logic signed [A_W-1:0] lin_re, lin_im;
                assign lin_re = A_W'($signed(xal_re)) <<< (A_FRAC - X_FRAC);
                assign lin_im = A_W'($signed(xal_im)) <<< (A_FRAC - X_FRAC);
                delay_reg #(.W (A_W), .N (mp_pkg::MUL_PIPE-1)) u_rlr (
                    .clk (clk), .rst (rst), .en (en), .d (lin_re), .q (base_c_re));
                delay_reg #(.W (A_W), .N (mp_pkg::MUL_PIPE-1)) u_rli (
                    .clk (clk), .rst (rst), .en (en), .d (lin_im), .q (base_c_im));
            end else begin : g_no_lineal
                // Ordenes 3, 5, 7: la muestra por |x|^(p-1), que es la
                // potencia numero j-1. Complejo por real: dos multiplicaciones.
                logic signed [A_W-1:0] bm;
                assign bm = $signed(pot[j*A_W-1 -: A_W]);

                logic signed [B_W-1:0] bx_re, bx_im;
                assign bx_re = $signed(xal_re) * bm;
                assign bx_im = $signed(xal_im) * bm;

                logic signed [B_W-1:0] bx_re_r, bx_im_r;
                delay_reg #(.W (B_W), .N (mp_pkg::MUL_PIPE-1)) u_rmr (
                    .clk (clk), .rst (rst), .en (en), .d (bx_re), .q (bx_re_r));
                delay_reg #(.W (B_W), .N (mp_pkg::MUL_PIPE-1)) u_rmi (
                    .clk (clk), .rst (rst), .en (en), .d (bx_im), .q (bx_im_r));

                fxp_round_sat #(
                    .IN_W (B_W), .IN_FRAC (B_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
                ) u_br (.din (bx_re_r), .dout (base_c_re));

                fxp_round_sat #(
                    .IN_W (B_W), .IN_FRAC (B_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
                ) u_bi (.din (bx_im_r), .dout (base_c_im));
            end

            // S4. Registro de la base.
            logic signed [A_W-1:0] base_re, base_im;
            delay_reg #(.W (A_W), .N (1)) u_rbr (
                .clk (clk), .rst (rst), .en (en), .d (base_c_re), .q (base_re));
            delay_reg #(.W (A_W), .N (1)) u_rbi (
                .clk (clk), .rst (rst), .en (en), .d (base_c_im), .q (base_im));

            // S5. prod = w[q,p] * base, y su registro.
            logic signed [C_W-1:0] wc_re, wc_im;
            assign wc_re = $signed(w_re[q][j]);
            assign wc_im = $signed(w_im[q][j]);

            logic signed [A_W-1:0] prod_c_re, prod_c_im;
            cmul_round_sat #(
                .A_W (C_W), .A_FRAC (C_FRAC), .B_W (A_W), .B_FRAC (A_FRAC),
                .O_W (A_W), .O_FRAC (A_FRAC), .PIPE (mp_pkg::MUL_PIPE-1)
            ) u_cmul (
                .clk (clk), .rst (rst), .en (en),
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

    // S6+. Acumulacion de los 12 terminos en cadena, en el mismo orden que
    // el modelo (col = q*NORD + j), con un registro cada ACC_PER_STAGE
    // pasos. Los terminos salen juntos de S5 y cada uno se retrasa hasta que
    // le toca a su grupo.
    for (genvar c = 0; c < NTERM; c++) begin : g_acc
        localparam int Q_IDX = c / NORD;
        localparam int J_IDX = c % NORD;
        // Corrimiento por orden: G^(p-1) con p = 2j+1.
        localparam int SHIFT = LOG2_G * 2 * J_IDX;
        localparam int GRP   = c / ACC_PER_STAGE;
        // Se registra al cerrar cada grupo, y siempre al final.
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

    // Salida: cambio de escala y vuelta al formato de la senal. El
    // acumulador esta en la escala de entrada, y pasar a la de salida es
    // multiplicar por G/G_out = 2^-(LOG2_GOUT-LOG2_G). No se implementa: el
    // bloque de reduccion lee el acumulador con esos bits fraccionarios de
    // mas.
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

    // valid marca una muestra nueva en la salida y dura un ciclo. `lleno`
    // dice que el pipeline ya se cargo y se cuenta en pulsos de `en`, que es
    // lo que hace avanzar el datapath. `en_d1` dice que la salida se acaba de
    // actualizar, para que `valid` no quede alto en los ciclos en que `en`
    // esta bajo y la salida repite el valor anterior.
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
