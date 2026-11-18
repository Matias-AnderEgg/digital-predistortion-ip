"""Compila lo que va a la placa (IP mas reproductor) y lo corre.

Uso:
    python rtl/hw_test/run.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
RAIZ = HERE.parent
TB = HERE / "tb"
BUILD = HERE / "sim_build"

sys.path.insert(0, str(TB))

from cocotb_tools.runner import get_runner  # noqa: E402

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
    RAIZ / "dpd_top" / "hdl" / "dpd_apply_top.sv",
    HERE / "hdl" / "dpd_test_player.sv",
    HERE / "hdl" / "dpd_hw_wrap.v",
]

# Los del bloque de registros generado, ver rtl/dpd_regs/run.py. Además
# PINCONNECTEMPTY no aplica y el envoltorio deja `valid` del IP sin usar a
# propósito: el reproductor cuenta pulsos.
WAIVERS = ["-Wno-GENUNNAMED", "-Wno-MULTIDRIVEN", "-Wno-UNUSEDSIGNAL", "-Wno-UNUSEDPARAM"]


def lint() -> None:
    cmd = ["verilator", "--lint-only", "-Wall", *WAIVERS,
           *[str(f) for f in FUENTES], "--top-module", "dpd_hw_wrap"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("lint fallido en dpd_hw_wrap")
    print("lint dpd_hw_wrap: limpio", flush=True)


def correr() -> None:
    runner = get_runner("verilator")
    runner.build(
        sources=[*FUENTES, TB / "tb_dpd_hw.sv"],
        hdl_toplevel="tb_dpd_hw",
        build_dir=BUILD,
        build_args=["-Wno-fatal", "-Wno-TIMESCALEMOD", *WAIVERS],
        always=True,
    )
    runner.test(test_module="test_dpd_hw", hdl_toplevel="tb_dpd_hw",
                test_dir=TB, build_dir=BUILD)


if __name__ == "__main__":
    lint()
    correr()
