`timescale 1ns/1ps

// Retardo de N ciclos para un vector, para alinear caminos de distinta
// profundidad. Con N=0 es un cable, porque la profundidad sale de una cuenta
// con parametros y a veces da cero.
module delay_reg #(
    parameter int W = 16,
    parameter int N = 1
) (
    input  logic clk,
    input  logic rst,
    input  logic en,
    input  logic [W-1:0] d,
    output logic [W-1:0] q
);
    if (N == 0) begin : g_cable
        assign q = d;
        // Marca clk, rst y en como usados, para que el lint no avise en
        // cada instancia con N=0.
        logic unused_ok;
        assign unused_ok = &{1'b0, clk, rst, en};
    end else begin : g_regs
        logic [N*W-1:0] r;
        always_ff @(posedge clk) begin
            if (rst) begin
                r <= '0;
            end else if (en) begin
                r[W-1:0] <= d;
                for (int k = 1; k < N; k++) begin
                    r[(k+1)*W-1 -: W] <= r[k*W-1 -: W];
                end
            end
        end
        assign q = r[N*W-1 -: W];
    end
endmodule
