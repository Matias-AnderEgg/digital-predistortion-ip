`timescale 1ns/1ps

// Linea de retardo de M taps para muestras complejas. Es la memoria del
// Memory Polynomial: el termino de tap q usa x[n-q].
//
// El orden de los taps lo verifican los casos `imp_*` de los golden vectors:
// una muestra sola seguida de M ceros recorre los taps uno por uno, y
// despues la salida tiene que dar cero.
module mp_delay_line #(
    parameter int W = 16,   // ancho de cada componente
    parameter int M = 3     // cantidad de taps
) (
    input  logic clk,
    input  logic rst,
    input  logic en,        // avanza la linea una muestra

    input  logic signed [W-1:0] x_re,
    input  logic signed [W-1:0] x_im,

    // Empaquetados de menor a mayor retardo: x[n] en los bits bajos,
    // despues x[n-1], despues x[n-2].
    output logic [M*W-1:0] tap_re,
    output logic [M*W-1:0] tap_im
);
    logic [M*W-1:0] r_re, r_im;

    always_ff @(posedge clk) begin
        if (rst) begin
            r_re <= '0;
            r_im <= '0;
        end else if (en) begin
            r_re[W-1:0] <= x_re;
            r_im[W-1:0] <= x_im;
            for (int k = 1; k < M; k++) begin
                r_re[(k+1)*W-1 -: W] <= r_re[k*W-1 -: W];
                r_im[(k+1)*W-1 -: W] <= r_im[k*W-1 -: W];
            end
        end
    end

    assign tap_re = r_re;
    assign tap_im = r_im;
endmodule
