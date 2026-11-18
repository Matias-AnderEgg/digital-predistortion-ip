`timescale 1ns/1ps

// Envoltorio del IP y del reproductor para el diseno de bloques de Vivado.
//
// Existe por dos motivos. Vivado solo acepta como referencia de modulo en un
// diseno de bloques un top en Verilog o VHDL, no en SystemVerilog. Y las
// anotaciones X_INTERFACE_* le dicen que agrupe los puertos en un bus
// AXI4-Lite y dos puertos de BRAM, asi se conectan de a un cable en el
// diseno de bloques en vez de senal por senal.
//
// No tiene logica propia: invierte el reset (el de la Zynq es activo en bajo
// y el del IP en alto) y reparte el GPIO de control.
//
// ctrl[15:0]  cantidad de muestras         ctrl[31]  start
// estado[0]   done                         estado[1] busy
module dpd_hw_wrap (
    (* X_INTERFACE_INFO = "xilinx.com:signal:clock:1.0 aclk CLK" *)
    (* X_INTERFACE_PARAMETER = "ASSOCIATED_BUSIF S_AXI:BRAM_IN:BRAM_OUT, ASSOCIATED_RESET aresetn" *)
    input  wire        aclk,
    (* X_INTERFACE_INFO = "xilinx.com:signal:reset:1.0 aresetn RST" *)
    (* X_INTERFACE_PARAMETER = "POLARITY ACTIVE_LOW" *)
    input  wire        aresetn,

    // --- AXI4-Lite: banco de registros del IP --------------------------
    // 12 bits de direccion porque el editor de direcciones asigna de a 4 KB.
    // El IP usa los 7 de abajo.
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWADDR" *)
    (* X_INTERFACE_PARAMETER = "PROTOCOL AXI4LITE, DATA_WIDTH 32, ADDR_WIDTH 12" *)
    input  wire [11:0] s_axi_awaddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWPROT" *)
    input  wire [2:0]  s_axi_awprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWVALID" *)
    input  wire        s_axi_awvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI AWREADY" *)
    output wire        s_axi_awready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WDATA" *)
    input  wire [31:0] s_axi_wdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WSTRB" *)
    input  wire [3:0]  s_axi_wstrb,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WVALID" *)
    input  wire        s_axi_wvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI WREADY" *)
    output wire        s_axi_wready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BRESP" *)
    output wire [1:0]  s_axi_bresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BVALID" *)
    output wire        s_axi_bvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI BREADY" *)
    input  wire        s_axi_bready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARADDR" *)
    input  wire [11:0] s_axi_araddr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARPROT" *)
    input  wire [2:0]  s_axi_arprot,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARVALID" *)
    input  wire        s_axi_arvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI ARREADY" *)
    output wire        s_axi_arready,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RDATA" *)
    output wire [31:0] s_axi_rdata,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RRESP" *)
    output wire [1:0]  s_axi_rresp,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RVALID" *)
    output wire        s_axi_rvalid,
    (* X_INTERFACE_INFO = "xilinx.com:interface:aximm:1.0 S_AXI RREADY" *)
    input  wire        s_axi_rready,

    // --- Control y estado, por un AXI GPIO -----------------------------
    input  wire [31:0] ctrl,
    output wire [31:0] estado,

    // --- BRAM de entrada, puerto B -------------------------------------
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN ADDR" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_ECC NONE, MEM_WIDTH 32, MEM_SIZE 32768, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
    output wire [31:0] bram_in_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN CLK" *)
    output wire        bram_in_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN DIN" *)
    output wire [31:0] bram_in_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN DOUT" *)
    input  wire [31:0] bram_in_dout,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN EN" *)
    output wire        bram_in_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN RST" *)
    output wire        bram_in_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_IN WE" *)
    output wire [3:0]  bram_in_we,

    // --- BRAM de salida, puerto B --------------------------------------
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT ADDR" *)
    (* X_INTERFACE_PARAMETER = "MASTER_TYPE BRAM_CTRL, MEM_ECC NONE, MEM_WIDTH 32, MEM_SIZE 32768, READ_WRITE_MODE READ_WRITE, READ_LATENCY 1" *)
    output wire [31:0] bram_out_addr,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT CLK" *)
    output wire        bram_out_clk,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT DIN" *)
    output wire [31:0] bram_out_din,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT DOUT" *)
    input  wire [31:0] bram_out_dout,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT EN" *)
    output wire        bram_out_en,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT RST" *)
    output wire        bram_out_rst,
    (* X_INTERFACE_INFO = "xilinx.com:interface:bram:1.0 BRAM_OUT WE" *)
    output wire [3:0]  bram_out_we
);
    wire rst = ~aresetn;

    wire        ip_en;
    wire [15:0] ip_x_re, ip_x_im, ip_u_re, ip_u_im;
    wire        ip_valid;
    wire        busy, done;

    dpd_apply_top u_ip (
        .clk (aclk), .rst (rst),
        .s_axil_awready (s_axi_awready), .s_axil_awvalid (s_axi_awvalid),
        .s_axil_awaddr  (s_axi_awaddr[6:0]), .s_axil_awprot (s_axi_awprot),
        .s_axil_wready  (s_axi_wready),  .s_axil_wvalid  (s_axi_wvalid),
        .s_axil_wdata   (s_axi_wdata),   .s_axil_wstrb   (s_axi_wstrb),
        .s_axil_bready  (s_axi_bready),  .s_axil_bvalid  (s_axi_bvalid),
        .s_axil_bresp   (s_axi_bresp),
        .s_axil_arready (s_axi_arready), .s_axil_arvalid (s_axi_arvalid),
        .s_axil_araddr  (s_axi_araddr[6:0]), .s_axil_arprot (s_axi_arprot),
        .s_axil_rready  (s_axi_rready),  .s_axil_rvalid  (s_axi_rvalid),
        .s_axil_rdata   (s_axi_rdata),   .s_axil_rresp   (s_axi_rresp),
        .en (ip_en), .x_re (ip_x_re), .x_im (ip_x_im),
        .u_re (ip_u_re), .u_im (ip_u_im), .valid (ip_valid),
        // El lazo de entrenamiento en logica no existe: corre en el ARM.
        .lazo_we (1'b0), .lazo_idx (4'd0), .lazo_re (16'd0), .lazo_im (16'd0)
    );

    dpd_test_player u_player (
        .clk (aclk), .rst (rst),
        .start (ctrl[31]), .n_muestras (ctrl[15:0]),
        .busy (busy), .done (done),
        .in_addr (bram_in_addr), .in_en (bram_in_en), .in_dout (bram_in_dout),
        .out_addr (bram_out_addr), .out_en (bram_out_en), .out_we (bram_out_we),
        .out_din (bram_out_din),
        .ip_en (ip_en), .ip_x_re (ip_x_re), .ip_x_im (ip_x_im),
        .ip_u_re (ip_u_re), .ip_u_im (ip_u_im)
    );

    assign estado = {30'd0, busy, done};

    // Los puertos B solo leen (entrada) o solo escriben (salida).
    assign bram_in_clk  = aclk;
    assign bram_in_rst  = 1'b0;
    assign bram_in_din  = 32'd0;
    assign bram_in_we   = 4'd0;
    assign bram_out_clk = aclk;
    assign bram_out_rst = 1'b0;
endmodule
