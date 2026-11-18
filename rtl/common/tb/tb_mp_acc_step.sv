`timescale 1ns/1ps

// Wrapper de simulacion del paso de acumulacion.
module tb_mp_acc_step;

    // A: el uso real. Acumulador s37/28, corrimiento de hasta 18 bits
    //    (3*(Pmax-1) con G=8 y Pmax=7).
    logic signed [36:0] acc_in_re_a, acc_in_im_a, term_re_a, term_im_a;
    logic signed [36:0] acc_out_re_a, acc_out_im_a;
    logic [4:0]         shift_a;
    mp_acc_step #(.A_W(37), .A_FRAC(28), .MAX_SHIFT(18))
        u_a (.acc_in_re(acc_in_re_a), .acc_in_im(acc_in_im_a),
             .term_re(term_re_a), .term_im(term_im_a), .shift(shift_a),
             .acc_out_re(acc_out_re_a), .acc_out_im(acc_out_im_a));

    // B: acumulador angosto a proposito, para que la saturacion se alcance
    //    seguido en vez de ser un caso raro.
    logic signed [19:0] acc_in_re_b, acc_in_im_b, term_re_b, term_im_b;
    logic signed [19:0] acc_out_re_b, acc_out_im_b;
    logic [2:0]         shift_b;
    mp_acc_step #(.A_W(20), .A_FRAC(15), .MAX_SHIFT(6))
        u_b (.acc_in_re(acc_in_re_b), .acc_in_im(acc_in_im_b),
             .term_re(term_re_b), .term_im(term_im_b), .shift(shift_b),
             .acc_out_re(acc_out_re_b), .acc_out_im(acc_out_im_b));

    // C: sin corrimiento. Es el caso G=1 (convencion no normalizada) y de
    //    paso verifica que el bloque no se rompe cuando MAX_SHIFT es 0.
    logic signed [36:0] acc_in_re_c, acc_in_im_c, term_re_c, term_im_c;
    logic signed [36:0] acc_out_re_c, acc_out_im_c;
    logic [0:0]         shift_c;
    mp_acc_step #(.A_W(37), .A_FRAC(28), .MAX_SHIFT(0))
        u_c (.acc_in_re(acc_in_re_c), .acc_in_im(acc_in_im_c),
             .term_re(term_re_c), .term_im(term_im_c), .shift(shift_c),
             .acc_out_re(acc_out_re_c), .acc_out_im(acc_out_im_c));

endmodule
