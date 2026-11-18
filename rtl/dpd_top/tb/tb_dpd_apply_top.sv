`timescale 1ns/1ps

// Wrapper de simulacion del IP completo.
//
// Igual que en dpd_regs, existe para aplanar lo que cocotb no puede tocar. Aca
// el DUT ya tiene los puertos planos, asi que el wrapper solo agrega la
// visibilidad del banco activo, que es interno y es justo lo que hay que mirar
// para verificar el doble buffer.
module tb_dpd_apply_top;

    localparam int X_W   = 16;
    localparam int C_W   = 16;
    localparam int M     = 3;
    localparam int NORD  = 4;
    localparam int NTERM = M * NORD;

    logic clk;
    logic rst;

    logic        s_axil_awready, s_axil_awvalid;
    logic [6:0]  s_axil_awaddr;
    logic [2:0]  s_axil_awprot;
    logic        s_axil_wready, s_axil_wvalid;
    logic [31:0] s_axil_wdata;
    logic [3:0]  s_axil_wstrb;
    logic        s_axil_bready, s_axil_bvalid;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_arready, s_axil_arvalid;
    logic [6:0]  s_axil_araddr;
    logic [2:0]  s_axil_arprot;
    logic        s_axil_rready, s_axil_rvalid;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;

    logic                  en;
    logic signed [X_W-1:0] x_re, x_im;
    logic signed [X_W-1:0] u_re, u_im;
    logic                  valid;

    logic                  lazo_we;
    logic [3:0]            lazo_idx;
    logic signed [C_W-1:0] lazo_re, lazo_im;

    dpd_apply_top #(
        .M (M), .NORD (NORD), .ACC_PER_STAGE (1)
    ) u_dut (
        .clk (clk), .rst (rst),
        .s_axil_awready (s_axil_awready), .s_axil_awvalid (s_axil_awvalid),
        .s_axil_awaddr  (s_axil_awaddr),  .s_axil_awprot  (s_axil_awprot),
        .s_axil_wready  (s_axil_wready),  .s_axil_wvalid  (s_axil_wvalid),
        .s_axil_wdata   (s_axil_wdata),   .s_axil_wstrb   (s_axil_wstrb),
        .s_axil_bready  (s_axil_bready),  .s_axil_bvalid  (s_axil_bvalid),
        .s_axil_bresp   (s_axil_bresp),
        .s_axil_arready (s_axil_arready), .s_axil_arvalid (s_axil_arvalid),
        .s_axil_araddr  (s_axil_araddr),  .s_axil_arprot  (s_axil_arprot),
        .s_axil_rready  (s_axil_rready),  .s_axil_rvalid  (s_axil_rvalid),
        .s_axil_rdata   (s_axil_rdata),   .s_axil_rresp   (s_axil_rresp),
        .en (en), .x_re (x_re), .x_im (x_im),
        .u_re (u_re), .u_im (u_im), .valid (valid),
        .lazo_we (lazo_we), .lazo_idx (lazo_idx),
        .lazo_re (lazo_re), .lazo_im (lazo_im)
    );

    // El banco activo aplanado. Es interno del DUT, y es lo que separa al
    // sombra de lo que el datapath realmente usa.
    logic [NTERM*C_W-1:0] act_re, act_im;
    for (genvar i = 0; i < NTERM; i++) begin : g_activo
        assign act_re[i*C_W +: C_W] = u_dut.g_activo[i].act_re;
        assign act_im[i*C_W +: C_W] = u_dut.g_activo[i].act_im;
    end

    // Senales de control que los tests miran directo.
    logic armado, pedido, aplicar;
    assign armado  = u_dut.armado;
    assign pedido  = u_dut.pedido;
    assign aplicar = u_dut.aplicar;

endmodule
