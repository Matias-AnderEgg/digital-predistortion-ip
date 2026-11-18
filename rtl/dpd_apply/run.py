"""Compila el bloque de aplicación del DPD y lo corre contra los golden
vectors.

Uso:
    python rtl/dpd_apply/run.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
HDL = HERE / "hdl"
COMMON_HDL = HERE.parent / "common" / "hdl"
TB = HERE / "tb"
BUILD = HERE / "sim_build"

sys.path.insert(0, str(TB))

from cocotb_tools.runner import get_runner

FUENTES = [
    COMMON_HDL / "mp_pkg.sv",
    COMMON_HDL / "fxp_round_sat.sv",
    COMMON_HDL / "delay_reg.sv",
    COMMON_HDL / "cmul_round_sat.sv",
    COMMON_HDL / "mp_magnitude.sv",
    COMMON_HDL / "mp_acc_step.sv",
    COMMON_HDL / "mp_delay_line.sv",
    HDL / "dpd_apply_core.sv",
]


def lint() -> None:
    cmd = ["verilator", "--lint-only", "-Wall",
           *[str(f) for f in FUENTES], "--top-module", "dpd_apply_core"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit("lint fallido en dpd_apply_core")
    print("lint dpd_apply_core: limpio", flush=True)


def correr(acc_per_stage: int) -> None:
    runner = get_runner("verilator")
    build = BUILD / f"acc{acc_per_stage}"
    runner.build(
        sources=FUENTES,
        hdl_toplevel="dpd_apply_core",
        build_dir=build,
        build_args=["-Wno-fatal", "-Wno-UNDRIVEN"],
        parameters={"ACC_PER_STAGE": acc_per_stage},
        always=True,
        waves=True,
    )
    runner.test(
        test_module="test_dpd_apply_core",
        hdl_toplevel="dpd_apply_core",
        test_dir=TB,
        build_dir=build,
        waves=True,
        results_xml=f"acc{acc_per_stage}.xml",
    )


def main() -> None:
    lint()
    # La profundidad del pipeline de acumulación es un parámetro, y el valor
    # correcto sale de la síntesis. Se corre con varios para verificar que
    # cambiarlo NO cambia ni un bit del resultado: si cambiara, sería que
    # alguna alineación depende del valor.
    por_defecto = [int(sys.argv[1])] if len(sys.argv) > 1 else [1, 2, 3, 4, 12]
    for n in por_defecto:
        print(f"\n=== ACC_PER_STAGE = {n} ===", flush=True)
        correr(n)


if __name__ == "__main__":
    main()
