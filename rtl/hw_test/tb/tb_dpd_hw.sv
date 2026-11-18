`timescale 1ns/1ps

// Banco de pruebas de lo que va a la placa: el envoltorio (IP mas
// reproductor) con dos BRAM de doble puerto alrededor.
//
// El puerto B de cada BRAM es el que maneja el reproductor. El puerto A lo
// maneja cocotb, y hace lo que en la placa hace el ARM a traves del AXI BRAM
// controller: cargar la entrada y leer la salida. El AXI-Lite y el GPIO de
// control quedan expuestos tal cual.
module tb_dpd_hw;

    localparam int PROF = 8192;          // palabras por BRAM, como en la placa

    logic clk;
    logic aresetn;

    // AXI-Lite, con los nombres que espera dpd_top/tb/axil.py
    logic        s_axil_awready, s_axil_awvalid;
    logic [11:0] s_axil_awaddr;
    logic [2:0]  s_axil_awprot;
    logic        s_axil_wready, s_axil_wvalid;
    logic [31:0] s_axil_wdata;
    logic [3:0]  s_axil_wstrb;
    logic        s_axil_bready, s_axil_bvalid;
    logic [1:0]  s_axil_bresp;
    logic        s_axil_arready, s_axil_arvalid;
    logic [11:0] s_axil_araddr;
    logic [2:0]  s_axil_arprot;
    logic        s_axil_rready, s_axil_rvalid;
    logic [31:0] s_axil_rdata;
    logic [1:0]  s_axil_rresp;

    logic [31:0] ctrl, estado;

    // Puerto A de cada BRAM, para cocotb
    logic        in_a_we;
    logic [12:0] in_a_addr;
    logic [31:0] in_a_din;
    logic [12:0] out_a_addr;
    logic [31:0] out_a_dout;

    // Puerto B, del envoltorio
    logic [31:0] in_b_addr, in_b_din, in_b_dout;
    logic        in_b_en, in_b_clk, in_b_rst;
    logic [3:0]  in_b_we;
    logic [31:0] out_b_addr, out_b_din, out_b_dout;
    logic        out_b_en, out_b_clk, out_b_rst;
    logic [3:0]  out_b_we;

    dpd_hw_wrap u_dut (
        .aclk (clk), .aresetn (aresetn),
        .s_axi_awaddr (s_axil_awaddr), .s_axi_awprot (s_axil_awprot),
        .s_axi_awvalid (s_axil_awvalid), .s_axi_awready (s_axil_awready),
        .s_axi_wdata (s_axil_wdata), .s_axi_wstrb (s_axil_wstrb),
        .s_axi_wvalid (s_axil_wvalid), .s_axi_wready (s_axil_wready),
        .s_axi_bresp (s_axil_bresp), .s_axi_bvalid (s_axil_bvalid),
        .s_axi_bready (s_axil_bready),
        .s_axi_araddr (s_axil_araddr), .s_axi_arprot (s_axil_arprot),
        .s_axi_arvalid (s_axil_arvalid), .s_axi_arready (s_axil_arready),
        .s_axi_rdata (s_axil_rdata), .s_axi_rresp (s_axil_rresp),
        .s_axi_rvalid (s_axil_rvalid), .s_axi_rready (s_axil_rready),
        .ctrl (ctrl), .estado (estado),
        .bram_in_addr (in_b_addr), .bram_in_clk (in_b_clk), .bram_in_din (in_b_din),
        .bram_in_dout (in_b_dout), .bram_in_en (in_b_en), .bram_in_rst (in_b_rst),
        .bram_in_we (in_b_we),
        .bram_out_addr (out_b_addr), .bram_out_clk (out_b_clk), .bram_out_din (out_b_din),
        .bram_out_dout (out_b_dout), .bram_out_en (out_b_en), .bram_out_rst (out_b_rst),
        .bram_out_we (out_b_we)
    );

    // BRAM de entrada: A escribe (cocotb), B lee (reproductor), un ciclo de
    // latencia de lectura como el block memory generator sin registro de
    // salida.
    logic [31:0] mem_in [PROF];
    always_ff @(posedge clk) begin
        if (in_a_we) mem_in[in_a_addr] <= in_a_din;
        if (in_b_en) in_b_dout <= mem_in[in_b_addr[14:2]];
    end

    // BRAM de salida: B escribe (reproductor), A lee (cocotb).
    logic [31:0] mem_out [PROF];
    always_ff @(posedge clk) begin
        if (out_b_en && (&out_b_we)) mem_out[out_b_addr[14:2]] <= out_b_din;
        out_a_dout <= mem_out[out_a_addr];
    end
    assign out_b_dout = 32'd0;
endmodule
