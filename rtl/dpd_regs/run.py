"""Genera el bloque de registros AXI-Lite desde el .rdl, lo lintea y lo corre.

Uso:
    python rtl/dpd_regs/run.py

El RTL de gen/ se regenera en cada corrida a propósito: así no puede quedar
desincronizado del .rdl, que es la única fuente de verdad del mapa.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RDL = HERE / "rdl" / "dpd_apply_regs.rdl"
GEN = HERE / "gen"
DOC = HERE / "gen" / "html"
TB = HERE / "tb"
BUILD = HERE / "sim_build"

sys.path.insert(0, str(TB))

from cocotb_tools.runner import get_runner

# PeakRDL del mismo entorno que el Python que corre este script, o el del PATH.
_junto = Path(sys.executable).with_name("peakrdl")
PEAKRDL = str(_junto) if _junto.exists() else "peakrdl"

# La variante "flat" del AXI4-Lite saca puertos sueltos en vez de una interfaz
# SystemVerilog. Hace falta por dos motivos: Verilator y cocotb no pueden ver
# las señales de adentro de una interfaz, y Vivado empaqueta mas fácil un IP
# con puertos planos.
#
# --err-if-bad-addr / --err-if-bad-rw hacen que el esclavo contesta SLVERR ante
# una dirección que no existe o un acceso del tipo equivocado, en vez de
# tragarselo en silencio. Es lo que hace que un bug del software se vea.
EXPORT = [
    PEAKRDL, "regblock", str(RDL),
    "-o", str(GEN),
    "--cpuif", "axi4-lite-flat",
    "--default-reset", "rst",
    "--err-if-bad-addr",
    "--err-if-bad-rw",
]

FUENTES = [
    GEN / "dpd_apply_regs_pkg.sv",
    GEN / "dpd_apply_regs.sv",
]

# Avisos sobre el código generado, no sobre el diseño:
#
#   GENUNNAMED   el for genvar del arreglo de coeficientes no lleva etiqueta.
#   MULTIDRIVEN  PeakRDL escribe miembros distintos de field_combo y de
#                field_storage desde always distintos. Verilator marca la
#                estructura entera como multiplemente manejada, pero cada
#                miembro tiene un solo manejador.
#   UNUSEDSIGNAL awprot/arprot, que el bloque no mira porque no implementa
#                chequeo de protección, y los dos bits bajos de la dirección,
#                que no se miran porque el mapa es todo de 32 bits alineados.
#   UNUSEDPARAM  el paquete exporta el tamaño del mapa y los parametros de la
#                instancia para que los usen el wrapper y el testbench. Al
#                linteear el bloque solo, nadie los usa todavía.
WAIVERS = [
    "-Wno-GENUNNAMED",
    "-Wno-MULTIDRIVEN",
    "-Wno-UNUSEDSIGNAL",
    "-Wno-UNUSEDPARAM",
]


def generar() -> None:
    GEN.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(EXPORT, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("fallo la generacion del bloque de registros")
    print(f"generado: {GEN}", flush=True)

    r = subprocess.run([PEAKRDL, "html", str(RDL), "-o", str(DOC)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("fallo la documentacion HTML")
    print(f"doc: {DOC / 'index.html'}", flush=True)


def mapa() -> None:
    r = subprocess.run([PEAKRDL, "dump", str(RDL)], capture_output=True, text=True)
    print(r.stdout, end="", flush=True)


def lint() -> None:
    cmd = ["verilator", "--lint-only", "-Wall", *WAIVERS,
           *[str(f) for f in FUENTES], "--top-module", "dpd_apply_regs"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("lint fallido en dpd_apply_regs")
    print("lint dpd_apply_regs: limpio", flush=True)


def correr() -> None:
    runner = get_runner("verilator")
    runner.build(
        sources=[*FUENTES, TB / "tb_dpd_apply_regs.sv"],
        hdl_toplevel="tb_dpd_apply_regs",
        build_dir=BUILD,
        # UNDRIVEN porque el reloj y el reset los maneja cocotb, y
        # TIMESCALEMOD porque PeakRDL no le pone timescale a lo que genera.
        build_args=["-Wno-fatal", "-Wno-UNDRIVEN", "-Wno-TIMESCALEMOD", *WAIVERS],
        always=True,
        waves=True,
    )
    runner.test(
        test_module="test_dpd_apply_regs",
        hdl_toplevel="tb_dpd_apply_regs",
        test_dir=TB,
        build_dir=BUILD,
        waves=True,
    )


def main() -> None:
    generar()
    mapa()
    lint()
    correr()


if __name__ == "__main__":
    main()
