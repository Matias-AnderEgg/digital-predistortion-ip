"""Compila el IP completo (datapath mas registros) y lo corre.

Uso:
    python rtl/dpd_top/run.py

El bloque de registros se regenera antes de compilar, para que el mapa que se
verifica acá sea el mismo que el del .rdl.
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAIZ = HERE.parent
HDL = HERE / "hdl"
TB = HERE / "tb"
BUILD = HERE / "sim_build"

sys.path.insert(0, str(TB))

from cocotb_tools.runner import get_runner

COMMON = RAIZ / "common" / "hdl"
GEN = RAIZ / "dpd_regs" / "gen"

FUENTES = [
    COMMON / "mp_pkg.sv",
    COMMON / "fxp_round_sat.sv",
    COMMON / "delay_reg.sv",
    COMMON / "cmul_round_sat.sv",
    COMMON / "mp_magnitude.sv",
    COMMON / "mp_acc_step.sv",
    COMMON / "mp_delay_line.sv",
    RAIZ / "dpd_apply" / "hdl" / "dpd_apply_core.sv",
    GEN / "dpd_apply_regs_pkg.sv",
    GEN / "dpd_apply_regs.sv",
    HDL / "dpd_apply_top.sv",
]

# Waivers sobre el bloque de registros generado. Están explicados en
# rtl/dpd_regs/run.py, que es donde se lintea ese bloque por su cuenta.
WAIVERS = ["-Wno-GENUNNAMED", "-Wno-MULTIDRIVEN", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM"]


def regenerar_registros() -> None:
    r = subprocess.run([sys.executable, str(RAIZ / "dpd_regs" / "run.py")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("fallo la regeneracion del bloque de registros")
    print("bloque de registros regenerado y verificado", flush=True)


def lint() -> None:
    cmd = ["verilator", "--lint-only", "-Wall", *WAIVERS,
           *[str(f) for f in FUENTES], "--top-module", "dpd_apply_top"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("lint fallido en dpd_apply_top")
    print("lint dpd_apply_top: limpio", flush=True)


def correr() -> None:
    runner = get_runner("verilator")
    runner.build(
        sources=[*FUENTES, TB / "tb_dpd_apply_top.sv"],
        hdl_toplevel="tb_dpd_apply_top",
        build_dir=BUILD,
        build_args=["-Wno-fatal", "-Wno-UNDRIVEN", "-Wno-TIMESCALEMOD", *WAIVERS],
        always=True,
        waves=True,
    )
    runner.test(
        test_module="test_dpd_apply_top",
        hdl_toplevel="tb_dpd_apply_top",
        test_dir=TB,
        build_dir=BUILD,
        waves=True,
    )


def main() -> None:
    regenerar_registros()
    lint()
    correr()


if __name__ == "__main__":
    main()
