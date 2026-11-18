`timescale 1ns/1ps

// Reproductor y grabador para la prueba en hardware. Lee N muestras de una
// BRAM, se las pasa al IP a una por ciclo y guarda cada salida en otra BRAM,
// en la misma posicion que su entrada. Cada muestra ocupa una palabra de 32
// bits, {im, re}.
//
// Protocolo con el ARM (por un AXI GPIO):
//   1. escribir n_muestras y start = 1
//   2. esperar done = 1
//   3. escribir start = 0, que baja done y deja todo listo para otra corrida
//
// No se mira `valid` porque el IP conserva su estado entre corridas, y en la
// segunda las primeras salidas serian restos de la anterior. Resetearlo
// tampoco sirve, porque vuelve los coeficientes al paso directo. Se cuentan
// los pulsos de `en`: la salida de la muestra s aparece despues del pulso
// s + LATENCY - 1, asi que se dan N + LATENCY - 1 pulsos, con ceros despues
// de las N muestras. Esos ceros dejan la linea de retardo en cero, como
// arrancan los golden vectors.
module dpd_test_player #(
    parameter int M             = 3,
    parameter int NORD          = 4,
    parameter int ACC_PER_STAGE = 1,
    parameter int N_W           = 16,     // ancho de la cuenta de muestras
    localparam int LATENCY      = mp_pkg::mp_latency(M, NORD, ACC_PER_STAGE)
) (
    input  logic        clk,
    input  logic        rst,

    // --- Control, desde el ARM ----------------------------------------
    input  logic        start,
    input  logic [N_W-1:0] n_muestras,
    output logic        busy,
    output logic        done,

    // --- BRAM de entrada, puerto B, latencia de lectura de 1 ciclo -----
    output logic [31:0] in_addr,          // en bytes, como el BRAM controller
    output logic        in_en,
    input  logic [31:0] in_dout,

    // --- BRAM de salida, puerto B -------------------------------------
    output logic [31:0] out_addr,
    output logic        out_en,
    output logic [3:0]  out_we,
    output logic [31:0] out_din,

    // --- Hacia el IP --------------------------------------------------
    output logic        ip_en,
    output logic [15:0] ip_x_re,
    output logic [15:0] ip_x_im,
    input  logic [15:0] ip_u_re,
    input  logic [15:0] ip_u_im
);
    // Un bit mas que N_W, para contar pulsos hasta N + LATENCY.
    localparam int P_W = N_W + 1;

    logic           start_d1, arranque;
    logic           corriendo;
    logic [N_W-1:0] n;
    logic [P_W-1:0] pulsos_total;          // N + LATENCY - 1
    logic [P_W-1:0] leido;                 // pulsos pedidos a la BRAM
    logic           pedido_d1, en_rango_d1;
    logic [P_W-1:0] dado;                  // pulsos de `en` ya dados
    logic           en_d1;
    logic [P_W-1:0] dado_d1;

    always_ff @(posedge clk) begin
        if (rst) start_d1 <= 1'b0;
        else     start_d1 <= start;
    end
    assign arranque = start & ~start_d1;

    // Lectura, un pedido por ciclo. El dato llega un ciclo despues, y en ese
    // ciclo se le da `en` al IP.
    logic pedir;
    assign pedir = corriendo && (leido < pulsos_total);

    assign in_en   = pedir && (leido < P_W'(n));
    assign in_addr = 32'(leido) << 2;

    always_ff @(posedge clk) begin
        if (rst) begin
            pedido_d1   <= 1'b0;
            en_rango_d1 <= 1'b0;
        end else begin
            pedido_d1   <= pedir;
            en_rango_d1 <= in_en;
        end
    end

    assign ip_en   = pedido_d1;
    assign ip_x_re = en_rango_d1 ? in_dout[15:0]  : 16'd0;
    assign ip_x_im = en_rango_d1 ? in_dout[31:16] : 16'd0;

    // Escritura. La salida del pulso p es la de la muestra p - (LATENCY-1).
    always_ff @(posedge clk) begin
        if (rst) begin
            en_d1   <= 1'b0;
            dado_d1 <= '0;
        end else begin
            en_d1   <= ip_en;
            dado_d1 <= dado;
        end
    end

    logic           escribir;
    logic [P_W-1:0] muestra;
    assign muestra  = dado_d1 - P_W'(LATENCY - 1);
    assign escribir = corriendo && en_d1 && (dado_d1 >= P_W'(LATENCY - 1))
                      && (muestra < P_W'(n));

    assign out_en   = escribir;
    assign out_we   = {4{escribir}};
    assign out_addr = 32'(muestra) << 2;
    assign out_din  = {ip_u_im, ip_u_re};

    // Estado
    always_ff @(posedge clk) begin
        if (rst) begin
            corriendo    <= 1'b0;
            done         <= 1'b0;
            n            <= '0;
            pulsos_total <= '0;
            leido        <= '0;
            dado         <= '0;
        end else begin
            if (arranque && !corriendo) begin
                corriendo    <= (n_muestras != 0);
                done         <= (n_muestras == 0);
                n            <= n_muestras;
                pulsos_total <= P_W'(n_muestras) + P_W'(LATENCY - 1);
                leido        <= '0;
                dado         <= '0;
            end else if (corriendo) begin
                if (pedir)  leido <= leido + 1'b1;
                if (ip_en)  dado  <= dado + 1'b1;
                // La ultima muestra se escribe en este ciclo.
                if (escribir && muestra == P_W'(n) - 1'b1) begin
                    corriendo <= 1'b0;
                    done      <= 1'b1;
                end
            end
            if (!start) done <= 1'b0;
        end
    end

    assign busy = corriendo;
endmodule
