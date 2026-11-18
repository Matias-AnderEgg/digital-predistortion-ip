`timescale 1ns/1ps

// Wrapper de simulacion de la linea de retardo. Las tres instancias
// comparten reloj, reset y enable.
module tb_mp_delay_line;

    logic clk;
    logic rst;
    logic en;

    // A: el uso real. Muestras Q0.15, M=3 taps.
    logic signed [15:0] x_re_a, x_im_a;
    logic [3*16-1:0]    tap_re_a, tap_im_a;
    mp_delay_line #(.W(16), .M(3))
        u_a (.clk(clk), .rst(rst), .en(en), .x_re(x_re_a), .x_im(x_im_a),
             .tap_re(tap_re_a), .tap_im(tap_im_a));

    // B: un solo tap. Caso de borde: sin memoria, el lazo interno no corre.
    logic signed [15:0] x_re_b, x_im_b;
    logic [15:0]        tap_re_b, tap_im_b;
    mp_delay_line #(.W(16), .M(1))
        u_b (.clk(clk), .rst(rst), .en(en), .x_re(x_re_b), .x_im(x_im_b),
             .tap_re(tap_re_b), .tap_im(tap_im_b));

    // C: mas taps y otro ancho, para que el barrido no pruebe un solo caso.
    logic signed [7:0] x_re_c, x_im_c;
    logic [5*8-1:0]    tap_re_c, tap_im_c;
    mp_delay_line #(.W(8), .M(5))
        u_c (.clk(clk), .rst(rst), .en(en), .x_re(x_re_c), .x_im(x_im_c),
             .tap_re(tap_re_c), .tap_im(tap_im_c));

endmodule
