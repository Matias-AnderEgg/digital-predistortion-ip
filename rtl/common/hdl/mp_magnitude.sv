`timescale 1ns/1ps

// Magnitud al cuadrado de una muestra compleja y sus potencias pares:
// a2 = I^2 + Q^2, a4 = a2*a2 y a6 = a4*a2, que son los |x|^(p-1) de los
// ordenes 3, 5 y 7.
//
// I^2 + Q^2 se calcula sin perdida, y la primera aparece al reducir al
// formato del acumulador. Con la entrada en Q0.15, |x| <= sqrt(2) y a6 <= 8,
// y el acumulador llega a +-256, asi que estas etapas nunca saturan.
//
// PIPE registros por potencia, uno despues del redondeo y los demas entre la
// multiplicacion y el redondeo (ver mp_pkg::MUL_PIPE). La latencia es
// NPOW*PIPE y todas las potencias salen alineadas. Con PIPE=0 es
// combinacional, y todos los modos dan los mismos valores.
module mp_magnitude #(
    parameter int X_W    = 16,   // muestra de entrada
    parameter int X_FRAC = 15,
    parameter int A_W    = 37,   // formato del acumulador
    parameter int A_FRAC = 28,
    parameter int NPOW   = 3,    // cuantas potencias pares: a2, a4, a6
    parameter int PIPE   = 0     // registros por potencia, 0 = combinacional
) (
    input  logic clk,            // solo se usan con PIPE>0
    input  logic rst,
    input  logic en,

    input  logic signed [X_W-1:0]      x_re,
    input  logic signed [X_W-1:0]      x_im,

    // Empaquetadas de menor a mayor potencia: a2 en los bits bajos, despues
    // a4, despues a6. Todas alineadas en el tiempo.
    output logic signed [NPOW*A_W-1:0] pot
);
    localparam int S_W    = 2*X_W + 1;   // I^2 + Q^2, sin perdida
    localparam int S_FRAC = 2*X_FRAC;
    localparam int P_W    = 2*A_W;       // a_k * a2, sin perdida
    localparam int P_FRAC = 2*A_FRAC;
    localparam int R_MUL  = (PIPE > 1) ? PIPE-1 : 0;   // antes del redondeo
    localparam int R_RED  = PIPE - R_MUL;              // despues

    logic signed [S_W-1:0] sq_re, sq_im, suma, suma_r;
    logic signed [A_W-1:0] a2_c, a2, a2_sal;

    assign sq_re = x_re * x_re;
    assign sq_im = x_im * x_im;
    assign suma  = sq_re + sq_im;

    delay_reg #(.W (S_W), .N (R_MUL)) u_rm_a2 (
        .clk (clk), .rst (rst), .en (en), .d (suma), .q (suma_r)
    );

    fxp_round_sat #(
        .IN_W (S_W), .IN_FRAC (S_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
    ) u_a2 (.din (suma_r), .dout (a2_c));

    delay_reg #(.W (A_W), .N (R_RED)) u_r_a2 (
        .clk (clk), .rst (rst), .en (en), .d (a2_c), .q (a2)
    );

    // Etapa k: |x|^(2k+2) = |x|^(2k) * |x|^2.
    for (genvar k = 1; k < NPOW; k++) begin : g_pot
        // La entrada de esta etapa sale (k-1)*PIPE ciclos despues que a2,
        // asi que hay que retrasar a2 lo mismo para multiplicarlos.
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

        logic signed [P_W-1:0] prod_r;
        delay_reg #(.W (P_W), .N (R_MUL)) u_rm (
            .clk (clk), .rst (rst), .en (en), .d (prod), .q (prod_r)
        );

        logic signed [A_W-1:0] a_c, a_out;
        fxp_round_sat #(
            .IN_W (P_W), .IN_FRAC (P_FRAC), .OUT_W (A_W), .OUT_FRAC (A_FRAC)
        ) u (.din (prod_r), .dout (a_c));

        delay_reg #(.W (A_W), .N (R_RED)) u_r (
            .clk (clk), .rst (rst), .en (en), .d (a_c), .q (a_out)
        );
    end

    // Alineacion de la salida: la potencia k esta lista k*PIPE ciclos
    // despues de a2, asi que se la retrasa (NPOW-1-k)*PIPE para que todas
    // salgan juntas.
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
