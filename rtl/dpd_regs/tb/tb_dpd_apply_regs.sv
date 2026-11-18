`timescale 1ns/1ps

// Wrapper de simulacion del bloque de registros.
//
// Existe por una sola razon: `hwif_in` y `hwif_out` son estructuras no
// empaquetadas, y ni Verilator ni cocotb pueden manejar o leer un miembro de
// una estructura no empaquetada desde afuera. Aca se aplanan a vectores y
// senales sueltas, que si se ven.
module tb_dpd_apply_regs;

    import dpd_apply_regs_pkg::*;

    localparam int NTERM = M * NORD;   // 12 coeficientes complejos
    localparam int C_W   = 16;

    logic clk;
    logic rst;

    // --- AXI4-Lite, puertos planos --------------------------------------
    logic        s_axil_awready;
    logic        s_axil_awvalid;
    logic [6:0]  s_axil_awaddr;
    logic [2:0]  s_axil_awprot;
    logic        s_axil_wready;
    logic        s_axil_wvalid;
    logic [31:0] s_axil_wdata;
    logic [3:0]  s_axil_wstrb;
    logic        s_axil_bready;
    logic        s_axil_bvalid;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_arready;
    logic        s_axil_arvalid;
    logic [6:0]  s_axil_araddr;
    logic [2:0]  s_axil_arprot;
    logic        s_axil_rready;
    logic        s_axil_rvalid;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;

    // --- Lado hardware, aplanado ----------------------------------------
    // Entrada: lo que el wrapper le contaria al bloque de registros.
    logic commit_pending;

    // Salidas: lo que el bloque de registros le da al wrapper.
    logic                   bypass;
    logic                   coef_src;
    logic                   commit;
    logic [NTERM*C_W-1:0]   w_re;
    logic [NTERM*C_W-1:0]   w_im;
    logic [NTERM-1:0]       swmod_re;
    logic [NTERM-1:0]       swmod_im;

    dpd_apply_regs__in_t  hwif_in;
    dpd_apply_regs__out_t hwif_out;

    always_comb hwif_in.STATUS.COMMIT_PENDING.next = commit_pending;

    assign bypass   = hwif_out.CTRL.BYPASS.value;
    assign coef_src = hwif_out.CTRL.COEF_SRC.value;
    assign commit   = hwif_out.CTRL.COMMIT.value;

    for (genvar i = 0; i < NTERM; i++) begin : g_aplanar
        assign w_re[i*C_W +: C_W] = hwif_out.W_COEFF[i].RE.value;
        assign w_im[i*C_W +: C_W] = hwif_out.W_COEFF[i].IM.value;
        assign swmod_re[i]        = hwif_out.W_COEFF[i].RE.swmod;
        assign swmod_im[i]        = hwif_out.W_COEFF[i].IM.swmod;
    end

    dpd_apply_regs u_dut (
        .clk             (clk),
        .rst             (rst),
        .s_axil_awready  (s_axil_awready),
        .s_axil_awvalid  (s_axil_awvalid),
        .s_axil_awaddr   (s_axil_awaddr),
        .s_axil_awprot   (s_axil_awprot),
        .s_axil_wready   (s_axil_wready),
        .s_axil_wvalid   (s_axil_wvalid),
        .s_axil_wdata    (s_axil_wdata),
        .s_axil_wstrb    (s_axil_wstrb),
        .s_axil_bready   (s_axil_bready),
        .s_axil_bvalid   (s_axil_bvalid),
        .s_axil_bresp    (s_axil_bresp),
        .s_axil_arready  (s_axil_arready),
        .s_axil_arvalid  (s_axil_arvalid),
        .s_axil_araddr   (s_axil_araddr),
        .s_axil_arprot   (s_axil_arprot),
        .s_axil_rready   (s_axil_rready),
        .s_axil_rvalid   (s_axil_rvalid),
        .s_axil_rdata    (s_axil_rdata),
        .s_axil_rresp    (s_axil_rresp),
        .hwif_in         (hwif_in),
        .hwif_out        (hwif_out)
    );

endmodule
