`timescale 1ns/1ps

// IP de aplicacion del DPD: el datapath del Memory Polynomial mas su bloque
// de registros AXI-Lite. Arma el puerto de coeficientes del core desde el
// banco activo, y el banco activo se carga desde los registros.
//
// Hay dos bancos. El sombra esta en dpd_apply_regs, lo escribe el host por
// AXI-Lite de a un coeficiente y leerlo devuelve lo escrito. El activo esta
// aca, lo lee el datapath y se carga entero con COMMIT, para que el datapath
// nunca use un juego mezclado.
//
// La copia se hace con `en` en alto, entre dos muestras. En el datapath
// paralelo los doce productos leen el banco en el mismo ciclo, asi que esto
// solo importaria en una version serializada, donde los terminos de una
// muestra se calculan en ciclos distintos.
//
// El bypass no copia `x` a `u`, que subiria el nivel LOG2_GOUT-LOG2_G bits
// (24 dB con los valores por defecto). Lo pasa a la escala de salida y lo
// alinea en latencia con el camino normal, para que conmutarlo no cambie el
// nivel ni meta un salto en la senal.
module dpd_apply_top #(
    parameter int X_W       = 16,
    parameter int X_FRAC    = 15,
    parameter int C_W       = 16,
    parameter int C_FRAC    = 15,
    parameter int A_W       = 37,
    parameter int A_FRAC    = 28,
    parameter int M         = 3,
    parameter int NORD      = 4,
    parameter int LOG2_G    = 3,
    parameter int LOG2_GOUT = 7,
    parameter int ACC_PER_STAGE = 1,
    // Derivados. Van aca y no en el cuerpo porque el ancho del indice del
    // lazo se usa en la lista de puertos.
    localparam int NTERM      = M * NORD,
    localparam int LAZO_IDX_W = (NTERM <= 1) ? 1 : $clog2(NTERM)
) (
    input  logic clk,
    input  logic rst,

    // --- AXI4-Lite, configuracion -------------------------------------
    output logic        s_axil_awready,
    input  logic        s_axil_awvalid,
    input  logic [6:0]  s_axil_awaddr,
    input  logic [2:0]  s_axil_awprot,
    output logic        s_axil_wready,
    input  logic        s_axil_wvalid,
    input  logic [31:0] s_axil_wdata,
    input  logic [3:0]  s_axil_wstrb,
    input  logic        s_axil_bready,
    output logic        s_axil_bvalid,
    output logic [1:0]  s_axil_bresp,
    output logic        s_axil_arready,
    input  logic        s_axil_arvalid,
    input  logic [6:0]  s_axil_araddr,
    input  logic [2:0]  s_axil_arprot,
    input  logic        s_axil_rready,
    output logic        s_axil_rvalid,
    output logic [31:0] s_axil_rdata,
    output logic [1:0]  s_axil_rresp,

    // --- Streaming de datos -------------------------------------------
    input  logic                  en,
    input  logic signed [X_W-1:0] x_re,
    input  logic signed [X_W-1:0] x_im,
    output logic signed [X_W-1:0] u_re,
    output logic signed [X_W-1:0] u_im,
    output logic                  valid,

    // --- Escritura directa del banco activo ---------------------------
    // Prevista para un lazo de adaptacion en la logica, que no se construyo.
    // Manda solo con CTRL.COEF_SRC en 1. En la placa van en cero.
    input  logic                 lazo_we,
    input  logic [LAZO_IDX_W-1:0] lazo_idx,
    input  logic signed [C_W-1:0] lazo_re,
    input  logic signed [C_W-1:0] lazo_im
);
    localparam int LATENCY = mp_pkg::mp_latency(M, NORD, ACC_PER_STAGE);

    // El bloque de registros se genera con M, NORD y las dos escalas fijas,
    // porque el mapa tiene un registro por coeficiente y CONFIG informa las
    // escalas. Con otros valores el mapa no corresponde y la simulacion no
    // lo muestra, asi que se corta al elaborar.
    if (M         != dpd_apply_regs_pkg::M         ||
        NORD      != dpd_apply_regs_pkg::NORD      ||
        LOG2_G    != dpd_apply_regs_pkg::LOG2_G    ||
        LOG2_GOUT != dpd_apply_regs_pkg::LOG2_GOUT) begin : g_mapa_no_coincide
        $error("dpd_apply_top: instancia M=%0d NORD=%0d LOG2_G=%0d LOG2_GOUT=%0d contra el mapa generado M=%0d NORD=%0d LOG2_G=%0d LOG2_GOUT=%0d. Regenerar rtl/dpd_regs con los mismos valores.",
               M, NORD, LOG2_G, LOG2_GOUT,
               dpd_apply_regs_pkg::M, dpd_apply_regs_pkg::NORD,
               dpd_apply_regs_pkg::LOG2_G, dpd_apply_regs_pkg::LOG2_GOUT);
    end

    // Bloque de registros
    dpd_apply_regs_pkg::dpd_apply_regs__in_t  hwif_in;
    dpd_apply_regs_pkg::dpd_apply_regs__out_t hwif_out;

    dpd_apply_regs u_regs (
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
        .hwif_in (hwif_in), .hwif_out (hwif_out)
    );

    logic bypass, coef_src, commit;
    assign bypass   = hwif_out.CTRL.BYPASS.value;
    assign coef_src = hwif_out.CTRL.COEF_SRC.value;
    assign commit   = hwif_out.CTRL.COMMIT.value;

    // Armado del commit. El sombra arranca en cero y el activo en paso
    // directo, asi que un commit antes de la primera carga dejaria la salida
    // en cero. SystemRDL no deja darle otro valor de reset a un solo elemento
    // del arreglo, entonces el commit no hace nada hasta que el host escribe
    // algun coeficiente.
    logic algun_swmod;
    always_comb begin
        algun_swmod = 1'b0;
        for (int i = 0; i < NTERM; i++)
            algun_swmod |= hwif_out.W_COEFF[i].RE.swmod | hwif_out.W_COEFF[i].IM.swmod;
    end

    logic armado;
    always_ff @(posedge clk) begin
        if (rst)              armado <= 1'b0;
        else if (algun_swmod) armado <= 1'b1;
    end

    logic pedido, commit_valido, aplicar;
    assign commit_valido = commit & armado;
    assign aplicar       = (pedido | commit_valido) & en;

    always_ff @(posedge clk) begin
        if (rst)                  pedido <= 1'b0;
        else if (aplicar)         pedido <= 1'b0;
        else if (commit_valido)   pedido <= 1'b1;
    end

    assign hwif_in.STATUS.COMMIT_PENDING.next = pedido;

    // Banco activo. Arranca en paso directo, w[q=0, p=1] al maximo positivo
    // y el resto en cero, para que la cadena funcione sin predistorsion hasta
    // que el host cargue coeficientes.
    localparam logic signed [C_W-1:0] UNO = {1'b0, {(C_W-1){1'b1}}};

    logic [M-1:0][NORD-1:0][C_W-1:0] w_re, w_im;

    for (genvar i = 0; i < NTERM; i++) begin : g_activo
        logic signed [C_W-1:0] act_re, act_im;

        logic escribe_lazo;
        assign escribe_lazo = coef_src & lazo_we & (lazo_idx == LAZO_IDX_W'(i));

        always_ff @(posedge clk) begin
            if (rst) begin
                act_re <= (i == 0) ? UNO : '0;
                act_im <= '0;
            end else if (escribe_lazo) begin
                act_re <= lazo_re;
                act_im <= lazo_im;
            end else if (aplicar & ~coef_src) begin
                act_re <= $signed(hwif_out.W_COEFF[i].RE.value);
                act_im <= $signed(hwif_out.W_COEFF[i].IM.value);
            end
        end

        // El mapa numera los coeficientes plano y el datapath los indexa
        // [tap][orden], con la misma relacion que el modelo y los .hex:
        // col = q*NORD + j.
        assign w_re[i/NORD][i%NORD] = act_re;
        assign w_im[i/NORD][i%NORD] = act_im;
    end

    // Datapath
    logic signed [X_W-1:0] u_mp_re, u_mp_im;

    dpd_apply_core #(
        .X_W (X_W), .X_FRAC (X_FRAC), .C_W (C_W), .C_FRAC (C_FRAC),
        .A_W (A_W), .A_FRAC (A_FRAC), .M (M), .NORD (NORD),
        .LOG2_G (LOG2_G), .LOG2_GOUT (LOG2_GOUT),
        .ACC_PER_STAGE (ACC_PER_STAGE)
    ) u_core (
        .clk (clk), .rst (rst), .en (en),
        .x_re (x_re), .x_im (x_im),
        .w_re (w_re), .w_im (w_im),
        .u_re (u_mp_re), .u_im (u_mp_im), .valid (valid)
    );

    // Bypass. La muestra se lee con LOG2_GOUT-LOG2_G bits fraccionarios de
    // mas y se redondea al formato de la senal, igual que en la salida del
    // core, asi que los dos caminos quedan en la misma escala.
    logic signed [X_W-1:0] byp_c_re, byp_c_im, byp_re, byp_im;

    fxp_round_sat #(
        .IN_W (X_W), .IN_FRAC (X_FRAC + LOG2_GOUT - LOG2_G),
        .OUT_W (X_W), .OUT_FRAC (X_FRAC)
    ) u_byp_re (.din (x_re), .dout (byp_c_re));

    fxp_round_sat #(
        .IN_W (X_W), .IN_FRAC (X_FRAC + LOG2_GOUT - LOG2_G),
        .OUT_W (X_W), .OUT_FRAC (X_FRAC)
    ) u_byp_im (.din (x_im), .dout (byp_c_im));

    delay_reg #(.W (X_W), .N (LATENCY)) u_byp_dre (
        .clk (clk), .rst (rst), .en (en), .d (byp_c_re), .q (byp_re));
    delay_reg #(.W (X_W), .N (LATENCY)) u_byp_dim (
        .clk (clk), .rst (rst), .en (en), .d (byp_c_im), .q (byp_im));

    assign u_re = bypass ? byp_re : u_mp_re;
    assign u_im = bypass ? byp_im : u_mp_im;

endmodule
