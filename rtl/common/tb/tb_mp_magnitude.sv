`timescale 1ns/1ps

// Wrapper de simulacion del bloque de magnitud.
module tb_mp_magnitude;

    logic clk;
    logic rst;
    logic en;

    // A: el uso real en el DPD. Muestra Q0.15, potencias al acumulador
    //    s37/28, tres potencias (a2, a4, a6) para Pmax=7.
    logic signed [15:0]    x_re_a, x_im_a;
    logic signed [3*37-1:0] pot_a;
    mp_magnitude #(.X_W(16), .X_FRAC(15), .A_W(37), .A_FRAC(28), .NPOW(3))
        u_a (.clk(clk), .rst(rst), .en(en), .x_re(x_re_a), .x_im(x_im_a), .pot(pot_a));

    // B: una sola potencia. Es el caso Pmax=3, y de paso verifica que la
    //    cadena de multiplicaciones sucesivas no se rompe cuando no hay.
    logic signed [15:0]  x_re_b, x_im_b;
    logic signed [36:0]  pot_b;
    mp_magnitude #(.X_W(16), .X_FRAC(15), .A_W(37), .A_FRAC(28), .NPOW(1))
        u_b (.clk(clk), .rst(rst), .en(en), .x_re(x_re_b), .x_im(x_im_b), .pot(pot_b));

    // C: acumulador angosto a proposito, para que la saturacion de las
    //    potencias sea alcanzable y quede ejercitada.
    logic signed [15:0]     x_re_c, x_im_c;
    logic signed [3*20-1:0] pot_c;
    mp_magnitude #(.X_W(16), .X_FRAC(15), .A_W(20), .A_FRAC(18), .NPOW(3))
        u_c (.clk(clk), .rst(rst), .en(en), .x_re(x_re_c), .x_im(x_im_c), .pot(pot_c));

    // D: entrada mas angosta, para que el barrido no pruebe un solo ancho.
    //    A_FRAC tiene que ser <= 2*X_FRAC: el bloque de redondeo saca bits
    //    fraccionarios, no los agrega.
    logic signed [11:0]     x_re_d, x_im_d;
    logic signed [3*32-1:0] pot_d;
    mp_magnitude #(.X_W(12), .X_FRAC(11), .A_W(32), .A_FRAC(20), .NPOW(3))
        u_d (.clk(clk), .rst(rst), .en(en), .x_re(x_re_d), .x_im(x_im_d), .pot(pot_d));

    // E: la misma configuracion que A pero con pipeline. Los dos modos
    //    tienen que dar exactamente los mismos valores. Si difieren, la
    //    alineacion entre potencias esta mal.
    logic signed [15:0]     x_re_e, x_im_e;
    logic signed [3*37-1:0] pot_e;
    mp_magnitude #(.X_W(16), .X_FRAC(15), .A_W(37), .A_FRAC(28), .NPOW(3), .PIPE(1))
        u_e (.clk(clk), .rst(rst), .en(en), .x_re(x_re_e), .x_im(x_im_e), .pot(pot_e));

    // F: con PIPE=3, que es como va en el datapath: dos registros entre
    //    cada multiplicacion y su redondeo, ademas del de despues.
    logic signed [15:0]     x_re_f, x_im_f;
    logic signed [3*37-1:0] pot_f;
    mp_magnitude #(.X_W(16), .X_FRAC(15), .A_W(37), .A_FRAC(28), .NPOW(3), .PIPE(3))
        u_f (.clk(clk), .rst(rst), .en(en), .x_re(x_re_f), .x_im(x_im_f), .pot(pot_f));

endmodule
