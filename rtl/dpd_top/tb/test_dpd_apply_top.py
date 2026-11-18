"""Verificacion del IP completo: datapath mas bloque de registros.

El datapath y el bloque de registros ya están verificados cada uno por su
lado. Lo que se verifica acá es lo que existe solo cuando se juntan: el doble
buffer, el armado del commit, el valor de reset del banco activo y el camino
de bypass. O sea, todo lo que puede sacar al aire una señal equivocada.
"""
import os

import cocotb
from cocotb.clock import Clock
from cocotb.queue import Queue
from cocotb.triggers import RisingEdge, Timer

from axil import AxiLite, MUESTREO_NS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
GOLDEN = os.path.join(ROOT, "fixed_point", "golden_vectors", "out")

X_W, C_W = 16, 16
M, NORD = 3, 4
NTERM = M * NORD
LOG2_G, LOG2_GOUT = 3, 7
K_ESCALA = LOG2_GOUT - LOG2_G          # 4 bits entre la escala de entrada y la de salida

MASK_C = (1 << C_W) - 1
UNO = (1 << (C_W - 1)) - 1             # 0x7FFF, el coeficiente de paso directo

ADDR_CTRL   = 0x08
ADDR_STATUS = 0x0C
ADDR_COEF   = 0x40

BIT_BYPASS   = 1 << 0
BIT_COEF_SRC = 1 << 1
BIT_COMMIT   = 1 << 8

VACIADO = 64   # ciclos para vaciar el pipeline, de sobra para cualquier profundidad


def addr_coef(i):
    return ADDR_COEF + 4 * i


def to_signed(val, width):
    val &= (1 << width) - 1
    return val - (1 << width) if val & (1 << (width - 1)) else val


def leer_hex(nombre, ancho=16):
    vals = []
    with open(os.path.join(GOLDEN, nombre)) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                n = int(ln, 16)
                vals.append(n - (1 << ancho) if n & (1 << (ancho - 1)) else n)
    return vals


def escalar(x):
    """Lo que el camino de bypass tiene que hacer con una muestra.

    Corrimiento de K_ESCALA bits a la derecha con redondeo al mas cercano y
    empate al par, que es exactamente lo que hace fxp_round_sat. No es una
    copia: la entrada esta en la escala G y la salida en G_out.
    """
    piso = x >> K_ESCALA
    guard = (x >> (K_ESCALA - 1)) & 1
    sticky = 1 if (x & ((1 << (K_ESCALA - 1)) - 1)) else 0
    return piso + (guard & (sticky | (piso & 1)))


def banco_activo(dut):
    plano_re = int(dut.act_re.value)
    plano_im = int(dut.act_im.value)
    return [(to_signed((plano_re >> (i * C_W)) & MASK_C, C_W),
             to_signed((plano_im >> (i * C_W)) & MASK_C, C_W)) for i in range(NTERM)]


PASO_DIRECTO = [(UNO, 0)] + [(0, 0)] * (NTERM - 1)


def coeficientes_golden():
    re = leer_hex("coeffs_w_re.hex")
    im = leer_hex("coeffs_w_im.hex")
    assert len(re) == len(im) == NTERM
    return list(zip(re, im))


def valores_de_prueba():
    """Doce pares distintos entre si, para que cruzar indices se note."""
    return [(to_signed((0x1234 + i * 0x0111) & MASK_C, C_W),
             to_signed((0xABCD - i * 0x0222) & MASK_C, C_W)) for i in range(NTERM)]


async def resetear(dut):
    """Reset del DUT, sin tocar el reloj.

    Vaciar el pipeline necesita un reset: el registro de desplazamiento de
    `valid` se satura, o sea que una vez que el pipeline se lleno `valid`
    queda alto. Bajar `en` lo congela pero no lo vacia, así que una segunda
    corrida sin reset leeria primero lo que quedó adentro de la etapa anterior.
    """
    dut.rst.value = 1
    dut.en.value = 0
    dut.x_re.value = 0
    dut.x_im.value = 0
    dut.lazo_we.value = 0
    dut.lazo_idx.value = 0
    dut.lazo_re.value = 0
    dut.lazo_im.value = 0
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


async def cargar(bus, coefs):
    for i, (re, im) in enumerate(coefs):
        await bus.escribir(addr_coef(i), ((im & MASK_C) << 16) | (re & MASK_C))


# Las cuatro piezas del banco de pruebas
#
# En el IP completo pasan tres cosas a la vez, el bus AXI por un lado, el
# streaming por el otro y el banco de coeficientes cambiando en el medio, y
# mezclarlas en una sola función se vuelve difícil de seguir.
#
#   Estimulos      maneja la entrada. No mira la salida.
#   MonitorSalida  mira la salida y la encola. No sabe que se espera.
#   el golden      es la prediccion, y se lee de archivo.
#   Comparador     consume la cola y compara. No toca el DUT.
#
# Ninguna de las cuatro sabe como esta hecha la otra, así que se pueden cambiar
# por separado. Por ejemplo el monitor no cambia si el estímulo pulsa `en` de
# otra forma.

class Estimulos:
    """Maneja las muestras de entrada y `en`."""

    def __init__(self, dut):
        self.dut = dut

    async def enviar(self, xi, xq, hueco=1):
        """Alimenta la secuencia, con un pulso de `en` cada `hueco` ciclos."""
        n = 0
        for ciclo in range(len(xi) * hueco):
            pulso = (ciclo % hueco == 0)
            if pulso:
                self.dut.x_re.value = xi[n]
                self.dut.x_im.value = xq[n]
                n += 1
            self.dut.en.value = 1 if pulso else 0
            await RisingEdge(self.dut.clk)

    async def vaciar(self, ciclos=VACIADO):
        """Sigue el reloj con ceros para que salga lo que quedo en el pipeline."""
        self.dut.x_re.value = 0
        self.dut.x_im.value = 0
        self.dut.en.value = 1
        for _ in range(ciclos):
            await RisingEdge(self.dut.clk)
        self.dut.en.value = 0


class MonitorSalida:
    """Encola una muestra por cada ciclo con `valid` en alto."""

    def __init__(self, dut):
        self.dut = dut
        self.cola = Queue()
        self.vistas = 0
        self._tarea = None

    async def _correr(self):
        while True:
            await RisingEdge(self.dut.clk)
            await Timer(MUESTREO_NS, unit="ns")
            if int(self.dut.valid.value):
                self.cola.put_nowait((to_signed(int(self.dut.u_re.value), X_W),
                                      to_signed(int(self.dut.u_im.value), X_W)))
                self.vistas += 1

    def arrancar(self):
        self._tarea = cocotb.start_soon(self._correr())
        return self

    def parar(self):
        if self._tarea is not None:
            self._tarea.cancel()
        return self


class Comparador:
    """Consume la cola del monitor y compara contra lo esperado."""

    TOPE = 10   # cuántas diferencias se informan antes de cortar el detalle

    def __init__(self, cola, esperados, etiquetas=None):
        self.cola, self.esperados, self.etiquetas = cola, esperados, etiquetas
        self.fallas, self.n = [], 0
        self._tarea = None

    async def _correr(self):
        while True:
            got = await self.cola.get()
            if self.n < len(self.esperados):
                esp = self.esperados[self.n]
                if got != esp and len(self.fallas) < self.TOPE:
                    quien = f" ({self.etiquetas[self.n]})" if self.etiquetas else ""
                    self.fallas.append(f"  n={self.n}{quien}: RTL={got} golden={esp}")
            self.n += 1

    def arrancar(self):
        self._tarea = cocotb.start_soon(self._correr())
        return self

    def parar(self):
        if self._tarea is not None:
            self._tarea.cancel()
        return self

    def exigir(self):
        assert self.n >= len(self.esperados), (
            f"salieron {self.n} muestras validas y se esperaban al menos "
            f"{len(self.esperados)}")
        assert not self.fallas, (
            f"{len(self.fallas)} muestras distintas del golden "
            f"(se muestran las primeras):\n" + "\n".join(self.fallas))


async def correr(dut, xi, xq, hueco=1):
    """Atajo para los tests que solo necesitan la lista de salidas.

    Arma estimulos y monitor y devuelve lo recolectado. Los tests que además
    comparan contra el golden usan el Comparador.
    """
    mon = MonitorSalida(dut).arrancar()
    est = Estimulos(dut)
    await est.enviar(xi, xq, hueco)
    await est.vaciar()
    mon.parar()
    return [mon.cola.get_nowait() for _ in range(mon.cola.qsize())]


# El estado seguro al salir del reset

@cocotb.test()
async def test_reset_es_paso_directo(dut):
    """El banco activo arranca en paso directo y no en ceros.

    Si arrancara en ceros el bloque emitiria cero desde el reset y mataria la
    transmisión hasta que el host escribiera los doce coeficientes. Saliendo
    así, la cadena funciona sin predistorsión, que es el estado seguro.
    """
    await arrancar(dut)
    assert banco_activo(dut) == PASO_DIRECTO, \
        f"el banco activo arranco en {banco_activo(dut)[:3]}..."


@cocotb.test()
async def test_paso_directo_deja_pasar_la_senal(dut):
    """Con el banco de reset la salida es la entrada, cambiada de escala.

    No es igual bit a bit al bypass porque el coeficiente de paso directo vale
    0x7FFF y no uno exacto, pero tiene que quedar dentro de un LSB.
    """
    await arrancar(dut)
    xi = [n * 137 - 2000 for n in range(64)]
    xq = [3000 - n * 91 for n in range(64)]
    salidas = await correr(dut, xi, xq)

    peor = 0
    for n in range(len(xi)):
        esp_i, esp_q = escalar(xi[n]), escalar(xq[n])
        got_i, got_q = salidas[n]
        peor = max(peor, abs(got_i - esp_i), abs(got_q - esp_q))
    assert peor <= 1, f"el paso directo se desvia hasta {peor} LSB de la entrada escalada"


# Doble buffer

@cocotb.test()
async def test_el_sombra_no_llega_al_datapath_sin_commit(dut):
    """Escribir los doce coeficientes no cambia nada hasta el commit.

    Es la razón de ser del doble buffer: durante los microsegundos que tarda
    el host en cargar, el datapath tiene que seguir usando el juego anterior.
    """
    bus = await arrancar(dut)
    await cargar(bus, valores_de_prueba())
    assert banco_activo(dut) == PASO_DIRECTO, \
        "el sombra se filtro al banco activo sin commit"


@cocotb.test()
async def test_commit_sin_armar_no_hace_nada(dut):
    """Un commit antes de la primera carga queda ignorado.

    El sombra arranca en ceros y el banco activo en paso directo. Sin el
    armado, este commit copiaría ceros sobre el paso directo y el bloque
    pasaría a emitir cero, que es justo lo que el valor de reset del activo
    estaba tratando de evitar.
    """
    bus = await arrancar(dut)
    assert int(dut.armado.value) == 0, "arranco armado"

    dut.en.value = 1
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == PASO_DIRECTO, \
        "un commit sin armar copio el sombra vacio sobre el paso directo"


@cocotb.test()
async def test_la_primera_escritura_arma_el_commit(dut):
    """Cualquier escritura de coeficiente alcanza para armar."""
    bus = await arrancar(dut)
    await bus.escribir(addr_coef(7), 0x0000_1234, strb=0x3)
    await Timer(MUESTREO_NS, unit="ns")
    assert int(dut.armado.value) == 1, "escribir un coeficiente no armo el commit"


@cocotb.test()
async def test_commit_armado_carga_el_banco(dut):
    """Con el commit armado, la transferencia es completa y de una vez."""
    bus = await arrancar(dut)
    esperados = valores_de_prueba()
    await cargar(bus, esperados)

    dut.en.value = 1
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == esperados, \
        "el banco activo no quedo igual al sombra despues del commit"


@cocotb.test()
async def test_el_commit_espera_la_frontera_de_muestra(dut):
    """Con `en` bajo el commit queda pendiente, y se aplica cuando sube.

    En la versión paralela la atomicidad ya sale gratis, porque los doce
    multiplicadores de coeficiente trabajan en el mismo ciclo. El enganche a
    `en` es para la versión serializada, donde los términos se calculan en
    ciclos distintos.
    """
    bus = await arrancar(dut)
    await cargar(bus, valores_de_prueba())

    dut.en.value = 0
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    for _ in range(6):
        await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == PASO_DIRECTO, \
        "el commit se aplico con `en` bajo, o sea fuera de una frontera de muestra"
    assert int(dut.pedido.value) == 1, "el commit no quedo pendiente"
    assert await bus.leer(ADDR_STATUS) & 1 == 1, "STATUS no reporta el commit pendiente"

    dut.en.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")
    dut.en.value = 0

    assert banco_activo(dut) == valores_de_prueba(), \
        "el commit pendiente no se aplico al subir `en`"
    assert await bus.leer(ADDR_STATUS) & 1 == 0, "STATUS quedo reportando pendiente"


@cocotb.test()
async def test_la_lectura_devuelve_el_sombra_no_el_activo(dut):
    """Despues de un commit, escribir el sombra de nuevo no toca el activo."""
    bus = await arrancar(dut)
    primero = valores_de_prueba()
    await cargar(bus, primero)

    dut.en.value = 1
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.en.value = 0

    segundo = [((re + 0x0100) & MASK_C, (im + 0x0200) & MASK_C) for re, im in primero]
    await cargar(bus, [(to_signed(a, C_W), to_signed(b, C_W)) for a, b in segundo])
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == primero, \
        "la segunda carga se filtro al banco activo sin commit"
    for i, (re, im) in enumerate(segundo):
        v = await bus.leer(addr_coef(i))
        assert v == (im << 16) | re, \
            f"leer W_COEFF[{i}] devolvio el activo y no el sombra"


# Bypass

@cocotb.test()
async def test_bypass_cambia_de_escala_no_copia_bits(dut):
    """El bypass corre la muestra a la escala de salida.

    Una copia de bits subiria el nivel físico 24 dB al conmutar, y entonces el
    bypass arruinaria la única medición para la que existe.
    """
    bus = await arrancar(dut)
    await bus.escribir(ADDR_CTRL, BIT_BYPASS)

    xi = [n * 311 - 9000 for n in range(48)]
    xq = [7000 - n * 173 for n in range(48)]
    salidas = await correr(dut, xi, xq)

    for n in range(len(xi)):
        esp = (escalar(xi[n]), escalar(xq[n]))
        assert salidas[n] == esp, \
            f"n={n}: bypass dio {salidas[n]} y se esperaba {esp} (entrada {xi[n]},{xq[n]})"
        assert salidas[n] != (xi[n], xq[n]) or xi[n] == 0, \
            f"n={n}: el bypass copio los bits en vez de cambiar de escala"


@cocotb.test()
async def test_bypass_alineado_en_latencia(dut):
    """Los dos caminos sacan la misma muestra en el mismo ciclo.

    Si no estuvieran alineados, conmutar el bypass en vivo meteria un salto.
    """
    bus = await arrancar(dut)
    xi = [n * 137 - 2000 for n in range(48)]
    xq = [3000 - n * 91 for n in range(48)]

    directo = await correr(dut, xi, xq)

    # Reset entre las dos corridas, o la segunda arrancaria con el pipeline
    # lleno de la primera. El banco activo vuelve a paso directo, que es el
    # que interesa comparar contra el bypass.
    await resetear(dut)
    await bus.escribir(ADDR_CTRL, BIT_BYPASS)
    puenteado = await correr(dut, xi, xq)

    assert len(directo) == len(puenteado), "los dos caminos dan distinta cantidad de salidas"
    peor = max(max(abs(a[0] - b[0]), abs(a[1] - b[1]))
               for a, b in zip(directo[:len(xi)], puenteado[:len(xi)]))
    assert peor <= 1, \
        f"los dos caminos difieren hasta {peor} LSB: no estan alineados en latencia"


# Escritura directa del banco activo, para el lazo de entrenamiento

@cocotb.test()
async def test_el_lazo_escribe_el_activo_directo(dut):
    """Con COEF_SRC en 1 manda el lazo y el commit del host no hace nada."""
    bus = await arrancar(dut)
    await cargar(bus, valores_de_prueba())
    await bus.escribir(ADDR_CTRL, BIT_COEF_SRC)

    esperados = list(PASO_DIRECTO)
    for i, (re, im) in [(0, (0x0111, -0x0222)), (5, (-0x0333, 0x0444)), (11, (0x0555, 0x0666))]:
        dut.lazo_idx.value = i
        dut.lazo_re.value = re
        dut.lazo_im.value = im
        dut.lazo_we.value = 1
        await RisingEdge(dut.clk)
        dut.lazo_we.value = 0
        esperados[i] = (re, im)
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == esperados, "el lazo no escribio el banco activo"

    # Y el commit del host queda sin efecto mientras el lazo manda.
    dut.en.value = 1
    await bus.escribir(ADDR_CTRL, BIT_COEF_SRC | BIT_COMMIT)
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.en.value = 0
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == esperados, \
        "con COEF_SRC en 1 el commit del host igual piso el banco activo"


@cocotb.test()
async def test_sin_coef_src_el_lazo_no_toca_nada(dut):
    """El puerto del lazo esta mudo mientras manda el host."""
    await arrancar(dut)
    dut.lazo_idx.value = 3
    dut.lazo_re.value = 0x7FFF
    dut.lazo_im.value = 0x7FFF
    dut.lazo_we.value = 1
    for _ in range(4):
        await RisingEdge(dut.clk)
    dut.lazo_we.value = 0
    await Timer(MUESTREO_NS, unit="ns")

    assert banco_activo(dut) == PASO_DIRECTO, \
        "el lazo escribio el banco activo con COEF_SRC en 0"


# El IP entero contra los golden vectors

@cocotb.test()
async def test_golden_con_coeficientes_por_el_bus(dut):
    """Los 4096 vectores funcionales, con los coeficientes cargados por AXI.

    El datapath ya se verificó contra estos mismos vectores con los
    coeficientes atados a un puerto. Lo que agrega este test es que llegar por
    el bus, pasar por el sombra y cruzar el commit da exactamente lo mismo.
    """
    bus = await arrancar(dut)
    await cargar(bus, coeficientes_golden())

    # El commit se pide con `en` bajo, así que queda pendiente y se aplica en
    # el primer flanco del streaming. Es como lo haría el software, y además
    # evita que el pipeline avance con ceros antes de los vectores.
    await bus.escribir(ADDR_CTRL, BIT_COMMIT)
    # El pulso de commit llega en el ultimo ciclo de la escritura, así que
    # `pedido` se levanta en el flanco siguiente.
    await RisingEdge(dut.clk)
    await Timer(MUESTREO_NS, unit="ns")
    assert int(dut.pedido.value) == 1, "el commit no quedo pendiente"

    # Las cuatro piezas, cada una con su trabajo.
    xi = leer_hex("functional_input_i.hex")          # estimulos
    xq = leer_hex("functional_input_q.hex")
    esperados = list(zip(leer_hex("functional_output_i.hex"),   # prediccion
                         leer_hex("functional_output_q.hex")))

    monitor = MonitorSalida(dut).arrancar()
    comparador = Comparador(monitor.cola, esperados).arrancar()

    estimulos = Estimulos(dut)
    await estimulos.enviar(xi, xq)
    await estimulos.vaciar()

    monitor.parar()
    await RisingEdge(dut.clk)      # deja que el comparador drene la cola
    comparador.parar()

    comparador.exigir()
    assert banco_activo(dut) == coeficientes_golden(), \
        "los coeficientes no llegaron enteros al banco activo"
