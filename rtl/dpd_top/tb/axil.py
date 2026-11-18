"""Maestro AXI4-Lite mínimo para cocotb.

No usa cocotbext-axi a propósito. AXI-Lite son dos canales de escritura, uno
de respuesta y dos de lectura, todos con el mismo apreton de manos, así que
escribirlo entero son cuarenta lineas y evita una dependencia mas.

Convención de muestreo, la misma que el resto de los testbench del repo: el
reloj es de 10 ns y las señales se miran 1 ns después del flanco, ya
asentadas. Como el diseno es sincrono, lo que se ve en t+1ns es lo que va a
haber en el flanco siguiente, así que ese muestreo predice el apreton de
manos sin tener que mirar hacia atrás.
"""
import cocotb
from cocotb.triggers import RisingEdge, Timer

# Respuestas de AXI.
OKAY   = 0
SLVERR = 2
DECERR = 3

MUESTREO_NS = 1


class ErrorAxi(Exception):
    """El esclavo contesto algo distinto de OKAY."""

    def __init__(self, op, addr, resp):
        self.op, self.addr, self.resp = op, addr, resp
        nombre = {OKAY: "OKAY", SLVERR: "SLVERR", DECERR: "DECERR"}.get(resp, str(resp))
        super().__init__(f"{op} en 0x{addr:03x} contesto {nombre}")


class AxiLite:
    def __init__(self, dut):
        self.dut = dut
        self.clk = dut.clk

    def reposo(self):
        """Deja el bus quieto. Se llama antes de soltar el reset."""
        d = self.dut
        d.s_axil_awvalid.value = 0
        d.s_axil_awaddr.value = 0
        d.s_axil_awprot.value = 0
        d.s_axil_wvalid.value = 0
        d.s_axil_wdata.value = 0
        d.s_axil_wstrb.value = 0
        d.s_axil_bready.value = 0
        d.s_axil_arvalid.value = 0
        d.s_axil_araddr.value = 0
        d.s_axil_arprot.value = 0
        d.s_axil_rready.value = 0

    async def _emitir(self, valid, ready, cargas):
        """Maneja un canal de salida hasta que valid y ready coinciden."""
        for sig, val in cargas:
            sig.value = val
        valid.value = 1
        while True:
            await Timer(MUESTREO_NS, unit="ns")
            aceptado = int(ready.value) == 1
            await RisingEdge(self.clk)
            if aceptado:
                break
        valid.value = 0

    async def _recibir(self, valid, ready, capturar):
        """Acepta un canal de entrada y devuelve lo que traia."""
        ready.value = 1
        while True:
            await Timer(MUESTREO_NS, unit="ns")
            if int(valid.value) == 1:
                traido = capturar()
                await RisingEdge(self.clk)
                break
            await RisingEdge(self.clk)
        ready.value = 0
        return traido

    async def escribir_cruda(self, addr, data, strb=0xF):
        """Escribe y devuelve la respuesta sin interpretarla."""
        d = self.dut
        await RisingEdge(self.clk)
        # Los dos canales van en paralelo: AXI no obliga a que el esclavo
        # acepte la dirección y el dato en el mismo ciclo, ni en un orden.
        aw = cocotb.start_soon(self._emitir(
            d.s_axil_awvalid, d.s_axil_awready,
            [(d.s_axil_awaddr, addr), (d.s_axil_awprot, 0)]))
        w = cocotb.start_soon(self._emitir(
            d.s_axil_wvalid, d.s_axil_wready,
            [(d.s_axil_wdata, data), (d.s_axil_wstrb, strb)]))
        await aw
        await w
        return await self._recibir(d.s_axil_bvalid, d.s_axil_bready,
                                   lambda: int(d.s_axil_bresp.value))

    async def leer_cruda(self, addr):
        """Lee y devuelve (dato, respuesta) sin interpretar la respuesta."""
        d = self.dut
        await RisingEdge(self.clk)
        await self._emitir(d.s_axil_arvalid, d.s_axil_arready,
                           [(d.s_axil_araddr, addr), (d.s_axil_arprot, 0)])
        return await self._recibir(
            d.s_axil_rvalid, d.s_axil_rready,
            lambda: (int(d.s_axil_rdata.value), int(d.s_axil_rresp.value)))

    async def escribir(self, addr, data, strb=0xF):
        """Escribe y falla si la respuesta no es OKAY."""
        resp = await self.escribir_cruda(addr, data, strb)
        if resp != OKAY:
            raise ErrorAxi("escritura", addr, resp)

    async def leer(self, addr):
        """Lee y falla si la respuesta no es OKAY."""
        data, resp = await self.leer_cruda(addr)
        if resp != OKAY:
            raise ErrorAxi("lectura", addr, resp)
        return data
