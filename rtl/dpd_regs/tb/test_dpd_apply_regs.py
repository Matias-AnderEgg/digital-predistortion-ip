"""Test cocotb del bloque de registros AXI-Lite.

Lo que se verifica no es el RTL generado, que es responsabilidad de PeakRDL,
sino que el MAPA sea el que queremos: que las direcciones caigan donde dice
el .rdl, que los campos tengan el ancho y la posición que dice, que las
señales que el wrapper va a necesitar salgan cuando tienen que salir, y que
un acceso invalido se vea en vez de pasar en silencio.
"""
import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer

from axil import AxiLite, OKAY, SLVERR, MUESTREO_NS

# Parametros de la instancia, los mismos que el .rdl.
M, NORD = 3, 4
NTERM = M * NORD
LOG2_G, LOG2_GOUT = 3, 7
C_W = 16
MASK_C = (1 << C_W) - 1

ADDR_ID     = 0x00
ADDR_CONFIG = 0x04
ADDR_CTRL   = 0x08
ADDR_STATUS = 0x0C
ADDR_COEF   = 0x40

ID_ESPERADO = 0x44504431   # 'DPD1'

BIT_BYPASS   = 1 << 0
BIT_COEF_SRC = 1 << 1
BIT_COMMIT   = 1 << 8

# Direcciones que NO están en el mapa. La primera cae en el hueco reservado
# entre STATUS y los coeficientes, la segunda justo después del ultimo
# coeficiente.
HUECOS = [0x20, 0x70]


def addr_coef(i: int) -> int:
    return ADDR_COEF + 4 * i


def valores_de_prueba():
    """Doce pares distintos entre si y distintos entre parte real e imaginaria.

    Que sean todos distintos es lo que hace que el test pueda fallar si se
    cruzan dos indices o si se cambian real por imaginaria.
    """
    return [(((0x1234 + i * 0x0111) & MASK_C),
             ((0xABCD - i * 0x0222) & MASK_C)) for i in range(NTERM)]


def coefs_de_hw(dut):
    """Lee el banco sombra por el lado de hardware, no por el bus."""
    plano_re = int(dut.w_re.value)
    plano_im = int(dut.w_im.value)
    return [((plano_re >> (i * C_W)) & MASK_C,
             (plano_im >> (i * C_W)) & MASK_C) for i in range(NTERM)]


async def resetear(dut):
    dut.rst.value = 1
    dut.commit_pending.value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.rst.value = 0
    await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")


async def arrancar(dut):
    cocotb.start_soon(Clock(dut.clk, 10, unit="ns").start())
    bus = AxiLite(dut)
    bus.reposo()
    await resetear(dut)
    return bus


class Vigia:
    """Junta lo que se ve en una señal ciclo a ciclo mientras corre algo.

    Los pulsos de swmod y de commit duran un ciclo, así que no alcanza con
    mirar después: hay que estar mirando mientras pasa.
    """

    def __init__(self, dut, *nombres):
        self.dut = dut
        self.nombres = nombres
        self.muestras = []
        self.tarea = None

    async def _correr(self):
        while True:
            await RisingEdge(self.dut.clk)
            await Timer(MUESTREO_NS, unit="ns")
            self.muestras.append(
                tuple(int(getattr(self.dut, n).value) for n in self.nombres))

    def arrancar(self):
        self.muestras = []
        self.tarea = cocotb.start_soon(self._correr())
        return self

    def parar(self):
        self.tarea.cancel()
        return self

    def ciclos_activos(self, i=0):
        """Cuantos ciclos la senal i estuvo en algo distinto de cero."""
        return sum(1 for m in self.muestras if m[i] != 0)

    def union(self, i=0):
        """OR de todo lo visto en la senal i."""
        vista = 0
        for m in self.muestras:
            vista |= m[i]
        return vista


@cocotb.test()
async def test_reset_y_constantes(dut):
    """Saliendo del reset: las constantes son las que son y el sombra esta limpio."""
    bus = await arrancar(dut)

    ident = await bus.leer(ADDR_ID)
    assert ident == ID_ESPERADO, f"ID = 0x{ident:08x}, esperaba 0x{ID_ESPERADO:08x}"

    cfg = await bus.leer(ADDR_CONFIG)
    leidos = {
        "M":         (cfg >> 0)  & 0xF,
        "NORD":      (cfg >> 4)  & 0xF,
        "LOG2_G":    (cfg >> 8)  & 0x1F,
        "LOG2_GOUT": (cfg >> 16) & 0x1F,
    }
    esperados = {"M": M, "NORD": NORD, "LOG2_G": LOG2_G, "LOG2_GOUT": LOG2_GOUT}
    assert leidos == esperados, f"CONFIG = {leidos}, esperaba {esperados}"

    assert await bus.leer(ADDR_CTRL) == 0, "CTRL tendria que arrancar en cero"
    assert await bus.leer(ADDR_STATUS) == 0, "STATUS tendria que arrancar en cero"

    for i in range(NTERM):
        v = await bus.leer(addr_coef(i))
        assert v == 0, f"W_COEFF[{i}] arranco en 0x{v:08x} y no en cero"

    assert coefs_de_hw(dut) == [(0, 0)] * NTERM, \
        "el lado de hardware del sombra no arranco en cero"
    assert int(dut.bypass.value) == 0
    assert int(dut.coef_src.value) == 0
    assert int(dut.commit.value) == 0


@cocotb.test()
async def test_coeficientes_ida_y_vuelta(dut):
    """Lo que se escribe en el sombra se lee igual, por el bus y por hardware."""
    bus = await arrancar(dut)
    esperados = valores_de_prueba()

    for i, (re, im) in enumerate(esperados):
        await bus.escribir(addr_coef(i), (im << 16) | re)

    for i, (re, im) in enumerate(esperados):
        v = await bus.leer(addr_coef(i))
        assert v == (im << 16) | re, \
            f"W_COEFF[{i}] leyo 0x{v:08x}, esperaba 0x{(im << 16) | re:08x}"

    assert coefs_de_hw(dut) == esperados, \
        "el sombra que ve el hardware no coincide con lo escrito por el bus"


@cocotb.test()
async def test_wstrb_separa_las_componentes(dut):
    """Una escritura de media palabra toca una sola componente.

    Importa porque el software puede escribir de a 16 bits, y si el strobe se
    ignorara cada escritura de la parte real pisaria la imaginaria con basura.
    """
    bus = await arrancar(dut)
    i = 5
    await bus.escribir(addr_coef(i), (0xBBBB << 16) | 0xAAAA)

    await bus.escribir(addr_coef(i), 0x00001111, strb=0x3)
    v = await bus.leer(addr_coef(i))
    assert v == (0xBBBB << 16) | 0x1111, \
        f"escribiendo solo RE quedo 0x{v:08x}, esperaba 0xBBBB1111"

    await bus.escribir(addr_coef(i), 0x22220000, strb=0xC)
    v = await bus.leer(addr_coef(i))
    assert v == (0x2222 << 16) | 0x1111, \
        f"escribiendo solo IM quedo 0x{v:08x}, esperaba 0x22221111"


@cocotb.test()
async def test_swmod_avisa_que_el_host_escribio(dut):
    """swmod pulsa en el coeficiente escrito y en ninguno mas.

    Es la señal con la que el wrapper arma el COMMIT, así que si pulsara de
    mas o de menos el armado quedaría mal.
    """
    bus = await arrancar(dut)

    for i in (0, 7, NTERM - 1):
        v = Vigia(dut, "swmod_re", "swmod_im").arrancar()
        await bus.escribir(addr_coef(i), 0x0BAD_F00D)
        await RisingEdge(dut.clk)
        v.parar()
        assert v.union(0) == (1 << i), \
            f"escribiendo W_COEFF[{i}] pulso swmod_re = 0b{v.union(0):012b}"
        assert v.union(1) == (1 << i), \
            f"escribiendo W_COEFF[{i}] pulso swmod_im = 0b{v.union(1):012b}"

    # Con strobe de media palabra tiene que pulsar una sola de las dos.
    v = Vigia(dut, "swmod_re", "swmod_im").arrancar()
    await bus.escribir(addr_coef(3), 0x0000_1234, strb=0x3)
    await RisingEdge(dut.clk)
    v.parar()
    assert v.union(0) == (1 << 3), "swmod_re no pulso escribiendo solo RE"
    assert v.union(1) == 0, "swmod_im pulso sin que se escribiera IM"


@cocotb.test()
async def test_commit_pulsa_un_ciclo_y_se_limpia(dut):
    """COMMIT sale como pulso de un ciclo y no queda pegado en el registro."""
    bus = await arrancar(dut)

    v = Vigia(dut, "commit").arrancar()
    await bus.escribir(ADDR_CTRL, BIT_COMMIT | BIT_BYPASS)
    for _ in range(4):
        await RisingEdge(dut.clk)
    v.parar()

    activos = v.ciclos_activos()
    assert activos == 1, f"commit estuvo alto {activos} ciclos, esperaba 1"

    ctrl = await bus.leer(ADDR_CTRL)
    assert ctrl & BIT_COMMIT == 0, \
        f"CTRL leyo 0x{ctrl:08x}: COMMIT quedo pegado en vez de autolimpiarse"
    assert ctrl & BIT_BYPASS != 0, \
        "el COMMIT se llevo puesto el BYPASS de la misma escritura"
    assert int(dut.commit.value) == 0, "commit quedo alto despues del pulso"


@cocotb.test()
async def test_bits_de_modo_quedan(dut):
    """BYPASS y COEF_SRC son estado, no pulso, y salen al hardware."""
    bus = await arrancar(dut)

    for valor, bypass, coef_src in [
        (BIT_BYPASS, 1, 0),
        (BIT_COEF_SRC, 0, 1),
        (BIT_BYPASS | BIT_COEF_SRC, 1, 1),
        (0, 0, 0),
    ]:
        await bus.escribir(ADDR_CTRL, valor)
        await Timer(MUESTREO_NS, unit="ns")
        assert int(dut.bypass.value) == bypass, f"bypass mal con CTRL = 0x{valor:x}"
        assert int(dut.coef_src.value) == coef_src, f"coef_src mal con CTRL = 0x{valor:x}"
        ctrl = await bus.leer(ADDR_CTRL)
        assert ctrl == valor, f"CTRL leyo 0x{ctrl:08x}, esperaba 0x{valor:08x}"


@cocotb.test()
async def test_status_lo_maneja_el_hardware(dut):
    """COMMIT_PENDING no lo escribe el host, lo escribe el wrapper."""
    bus = await arrancar(dut)

    for pendiente in (1, 0, 1):
        dut.commit_pending.value = pendiente
        await RisingEdge(dut.clk)
        st = await bus.leer(ADDR_STATUS)
        assert st & 1 == pendiente, \
            f"STATUS leyo 0x{st:08x} con commit_pending = {pendiente}"

    # Y el host no lo puede pisar: STATUS es de solo lectura.
    dut.commit_pending.value = 1
    await RisingEdge(dut.clk)
    resp = await bus.escribir_cruda(ADDR_STATUS, 0)
    assert resp == SLVERR, \
        f"escribir STATUS contesto {resp}, esperaba SLVERR: es de solo lectura"
    st = await bus.leer(ADDR_STATUS)
    assert st & 1 == 1, "la escritura rechazada igual cambio STATUS"


@cocotb.test()
async def test_accesos_invalidos_contestan_slverr(dut):
    """Un acceso fuera del mapa o del tipo equivocado se ve, no pasa callado.

    Sin esto un bug de direcciones del software se manifestaria como
    coeficientes que no llegan, que es mucho mas caro de encontrar.
    """
    bus = await arrancar(dut)

    for addr in HUECOS:
        _, resp = await bus.leer_cruda(addr)
        assert resp == SLVERR, f"leer 0x{addr:02x} contesto {resp}, esperaba SLVERR"
        resp = await bus.escribir_cruda(addr, 0xDEADBEEF)
        assert resp == SLVERR, f"escribir 0x{addr:02x} contesto {resp}, esperaba SLVERR"

    # ID y CONFIG son constantes: leerlos anda, escribirlos no.
    for addr in (ADDR_ID, ADDR_CONFIG):
        _, resp = await bus.leer_cruda(addr)
        assert resp == OKAY, f"leer 0x{addr:02x} contesto {resp}, esperaba OKAY"
        resp = await bus.escribir_cruda(addr, 0)
        assert resp == SLVERR, \
            f"escribir 0x{addr:02x} contesto {resp}: es de solo lectura"

    # Y el bloque sigue funcionando después de los rechazos.
    assert await bus.leer(ADDR_ID) == ID_ESPERADO, \
        "el bloque quedo colgado despues de un acceso invalido"


@cocotb.test()
async def test_el_reset_limpia_el_sombra(dut):
    """Despues de un reset el sombra vuelve a cero y los modos tambien."""
    bus = await arrancar(dut)

    for i, (re, im) in enumerate(valores_de_prueba()):
        await bus.escribir(addr_coef(i), (im << 16) | re)
    await bus.escribir(ADDR_CTRL, BIT_BYPASS | BIT_COEF_SRC)

    await resetear(dut)

    assert coefs_de_hw(dut) == [(0, 0)] * NTERM, "el reset no limpio el sombra"
    assert await bus.leer(ADDR_CTRL) == 0, "el reset no limpio CTRL"
    assert int(dut.bypass.value) == 0
    assert int(dut.coef_src.value) == 0


@cocotb.test()
async def test_el_mapa_no_se_solapa(dut):
    """Cada dirección del mapa es un registro distinto.

    Barrido barato que atrapa un mapa mal armado: si dos direcciones cayeran
    en el mismo almacenamiento, escribir una cambiaria la otra.
    """
    bus = await arrancar(dut)

    for i in range(NTERM):
        await bus.escribir(addr_coef(i), 0)

    for i in range(NTERM):
        marca = 0x8000_0000 | (i + 1)
        await bus.escribir(addr_coef(i), marca)
        for j in range(NTERM):
            v = await bus.leer(addr_coef(j))
            esperado = marca if j == i else 0
            assert v == esperado, \
                f"escribiendo W_COEFF[{i}] cambio W_COEFF[{j}] a 0x{v:08x}"
        await bus.escribir(addr_coef(i), 0)
