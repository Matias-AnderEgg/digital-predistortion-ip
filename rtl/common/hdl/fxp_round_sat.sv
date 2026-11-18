`timescale 1ns/1ps

// Reduccion de formato de punto fijo: redondeo al mas cercano (empate al
// par) y saturacion en los extremos del formato de salida. Es el `quantize`
// del modelo de punto fijo (fxpmath con rounding='around',
// overflow='saturate') y se usa en cada punto del datapath donde un valor
// pasa a un formato mas angosto.
//
// `din` es un entero con signo de IN_W bits que vale din / 2^IN_FRAC, y
// `dout` lo mismo con OUT_W y OUT_FRAC. El bloque solo saca bits
// fraccionarios, asi que pide IN_FRAC >= OUT_FRAC.
module fxp_round_sat #(
    parameter int IN_W     = 32,
    parameter int IN_FRAC  = 30,
    parameter int OUT_W    = 16,
    parameter int OUT_FRAC = 15
) (
    input  logic signed [IN_W-1:0]  din,
    output logic signed [OUT_W-1:0] dout
);
    // Bits fraccionarios que se descartan. Con 0 el bloque solo satura.
    localparam int SHIFT = IN_FRAC - OUT_FRAC;

    // Con SHIFT negativo el corrimiento queda indefinido y la salida daria
    // cero sin aviso, asi que se corta al elaborar.
    if (SHIFT < 0) begin : g_parametros_invalidos
        $error("fxp_round_sat: IN_FRAC=%0d < OUT_FRAC=%0d. Este bloque saca bits fraccionarios, no los agrega.",
               IN_FRAC, OUT_FRAC);
    end

    // Ancho para el valor ya corrido: +1 bit porque el redondeo puede
    // generar un acarreo que empuje un bit mas arriba.
    localparam int ROUND_W = IN_W - SHIFT + 1;

    // Ancho de comparacion: tiene que poder representar MAX y MIN sin
    // recortarlos, incluso si ROUND_W es chico.
    localparam int CMP_W = (ROUND_W > (OUT_W + 1)) ? ROUND_W : (OUT_W + 1);

    localparam logic signed [OUT_W-1:0] MAX = {1'b0, {(OUT_W-1){1'b1}}};
    localparam logic signed [OUT_W-1:0] MIN = {1'b1, {(OUT_W-1){1'b0}}};

    logic signed [ROUND_W-1:0] piso;
    logic signed [CMP_W-1:0]   redondeado;

    logic guard;   // el mas significativo de los bits que se descartan
    logic sticky;  // OR de todos los demas descartados
    logic inc;

    // Corrimiento aritmetico: mantiene el signo, y para negativos redondea
    // hacia menos infinito (piso), no hacia cero.
    assign piso = ROUND_W'(din >>> SHIFT);

    if (SHIFT >= 1) begin : g_guard
        assign guard = din[SHIFT-1];
    end else begin : g_guard_vacio
        assign guard = 1'b0;
    end

    if (SHIFT >= 2) begin : g_sticky
        assign sticky = |din[SHIFT-2:0];
    end else begin : g_sticky_vacio
        assign sticky = 1'b0;
    end

    // Redondeo al mas cercano con empate al par:
    //   descartado < 1/2  -> guard=0             -> se queda en el piso
    //   descartado > 1/2  -> guard=1, sticky=1   -> sube uno
    //   descartado = 1/2  -> guard=1, sticky=0   -> sube solo si el piso es
    //                                               impar, para terminar par
    assign inc = guard & (sticky | piso[0]);

    assign redondeado = CMP_W'(piso) + (inc ? CMP_W'(1) : CMP_W'(0));

    // Saturacion: si el valor redondeado no entra en el formato de salida,
    // se entrega el extremo mas cercano en vez de dejarlo dar la vuelta.
    always_comb begin
        if      (redondeado > CMP_W'(MAX)) dout = MAX;
        else if (redondeado < CMP_W'(MIN)) dout = MIN;
        else                               dout = OUT_W'(redondeado);
    end
endmodule
