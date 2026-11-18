`timescale 1ns/1ps

// Wrapper de simulacion del multiplicador complejo. Igual que con el bloque
// de redondeo, los parametros se fijan al elaborar, asi que para probar
// varias combinaciones de anchos hay que instanciar varias veces.
module tb_cmul_round_sat;

    logic clk;
    logic rst;
    logic en;

    // A: el uso real en el DPD. Coeficiente Q0.15 por base en formato del
    //    acumulador, resultado al acumulador.
    logic signed [15:0] a_re_a, a_im_a;
    logic signed [36:0] b_re_a, b_im_a;
    logic signed [36:0] o_re_a, o_im_a;
    cmul_round_sat #(.A_W(16), .A_FRAC(15), .B_W(37), .B_FRAC(28),
                     .O_W(37), .O_FRAC(28))
        u_a (.clk(clk), .rst(rst), .en(en),
             .a_re(a_re_a), .a_im(a_im_a), .b_re(b_re_a), .b_im(b_im_a),
             .o_re(o_re_a), .o_im(o_im_a));

    // B: todo en Q0.15. Caso chico, el que mas satura: el producto de dos
    //    numeros de modulo cercano a 1 puede pasarse de fondo de escala.
    logic signed [15:0] a_re_b, a_im_b, b_re_b, b_im_b, o_re_b, o_im_b;
    cmul_round_sat #(.A_W(16), .A_FRAC(15), .B_W(16), .B_FRAC(15),
                     .O_W(16), .O_FRAC(15))
        u_b (.clk(clk), .rst(rst), .en(en),
             .a_re(a_re_b), .a_im(a_im_b), .b_re(b_re_b), .b_im(b_im_b),
             .o_re(o_re_b), .o_im(o_im_b));

    // C: anchos y comas distintos en los tres lados, para que el barrido no
    //    pruebe solo casos simetricos.
    logic signed [15:0] a_re_c, a_im_c;
    logic signed [15:0] b_re_c, b_im_c;
    logic signed [23:0] o_re_c, o_im_c;
    cmul_round_sat #(.A_W(16), .A_FRAC(14), .B_W(16), .B_FRAC(15),
                     .O_W(24), .O_FRAC(22))
        u_c (.clk(clk), .rst(rst), .en(en),
             .a_re(a_re_c), .a_im(a_im_c), .b_re(b_re_c), .b_im(b_im_c),
             .o_re(o_re_c), .o_im(o_im_c));

    // D: salida mas angosta que la entrada en bits fraccionarios y en
    //    rango, para ejercitar redondeo y saturacion a la vez.
    logic signed [15:0] a_re_d, a_im_d;
    logic signed [36:0] b_re_d, b_im_d;
    logic signed [15:0] o_re_d, o_im_d;
    cmul_round_sat #(.A_W(16), .A_FRAC(15), .B_W(37), .B_FRAC(28),
                     .O_W(16), .O_FRAC(15))
        u_d (.clk(clk), .rst(rst), .en(en),
             .a_re(a_re_d), .a_im(a_im_d), .b_re(b_re_d), .b_im(b_im_d),
             .o_re(o_re_d), .o_im(o_im_d));

    // E: la misma configuracion que A pero con dos registros antes del
    //    redondeo, como va en el datapath. Tiene que dar lo mismo que A dos
    //    ciclos despues.
    logic signed [15:0] a_re_e, a_im_e;
    logic signed [36:0] b_re_e, b_im_e;
    logic signed [36:0] o_re_e, o_im_e;
    cmul_round_sat #(.A_W(16), .A_FRAC(15), .B_W(37), .B_FRAC(28),
                     .O_W(37), .O_FRAC(28), .PIPE(2))
        u_e (.clk(clk), .rst(rst), .en(en),
             .a_re(a_re_e), .a_im(a_im_e), .b_re(b_re_e), .b_im(b_im_e),
             .o_re(o_re_e), .o_im(o_im_e));

endmodule
