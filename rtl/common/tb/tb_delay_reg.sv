`timescale 1ns/1ps

// Wrapper de simulacion del retardo de N ciclos.
module tb_delay_reg;

    logic clk, rst, en;

    // N=0 tiene que ser un cable. Es el caso que mas facil se rompe, porque
    // la profundidad de alineacion sale de una cuenta con parametros y a
    // veces da cero.
    logic [15:0] d_a, q_a;
    delay_reg #(.W(16), .N(0)) u_a (.clk(clk), .rst(rst), .en(en), .d(d_a), .q(q_a));

    logic [15:0] d_b, q_b;
    delay_reg #(.W(16), .N(1)) u_b (.clk(clk), .rst(rst), .en(en), .d(d_b), .q(q_b));

    logic [15:0] d_c, q_c;
    delay_reg #(.W(16), .N(5)) u_c (.clk(clk), .rst(rst), .en(en), .d(d_c), .q(q_c));

    logic [73:0] d_d, q_d;
    delay_reg #(.W(74), .N(3)) u_d (.clk(clk), .rst(rst), .en(en), .d(d_d), .q(q_d));

endmodule
