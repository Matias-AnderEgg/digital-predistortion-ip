"""Compila los bloques comunes con Verilator y corre sus tests de cocotb.

Uso:
    python rtl/common/run.py
"""
import pathlib
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
HDL = HERE / "hdl"
TB = HERE / "tb"
BUILD = HERE / "sim_build"

sys.path.insert(0, str(TB))

from cocotb_tools.runner import get_runner


# (nombre del toplevel del testbench, fuentes que necesita)
BLOQUES = [
    ("tb_fxp_round_sat", ["fxp_round_sat.sv"]),
    ("tb_cmul_round_sat", ["fxp_round_sat.sv", "delay_reg.sv", "cmul_round_sat.sv"]),
    ("tb_mp_magnitude", ["fxp_round_sat.sv", "delay_reg.sv", "mp_magnitude.sv"]),
    ("tb_mp_acc_step", ["fxp_round_sat.sv", "mp_acc_step.sv"]),
    ("tb_mp_delay_line", ["mp_delay_line.sv"]),
    ("tb_delay_reg", ["delay_reg.sv"]),
]


def lint(fuentes: list[str], top: str) -> None:
    """Análisis estatico del módulo, antes de compilar.

    Vale la pena como paso propio: `--lint-only` con `-Wall` encuentra cosas
    que la compilacion normal deja pasar, y además nos encontro un error
    interno de Verilator 5.052 que no aparecia al compilar (ver el comentario
    en mp_magnitude.sv). Si el lint no corre, ese tipo de cosa se descubre
    tarde.
    """
    cmd = ["verilator", "--lint-only", "-Wall",
           *[str(HDL / f) for f in fuentes], "--top-module", top]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout + r.stderr)
        raise SystemExit(f"lint fallido en {top}")
    print(f"  lint {top}: limpio", flush=True)


def main() -> None:
    runner = get_runner("verilator")
    for toplevel, fuentes in BLOQUES:
        # El módulo del diseno es el ultimo de la lista; los anteriores son
        # sus dependencias.
        lint(fuentes, pathlib.Path(fuentes[-1]).stem)
    for toplevel, fuentes in BLOQUES:
        build = BUILD / toplevel
        print(f"\n=== {toplevel} ===", flush=True)
        runner.build(
            sources=[HDL / f for f in fuentes] + [TB / f"{toplevel}.sv"],
            hdl_toplevel=toplevel,
            build_dir=build,
            # UNDRIVEN es esperable: a las entradas del wrapper las maneja
            # cocotb, no el RTL. Los demás avisos queremos verlos.
            build_args=["-Wno-fatal", "-Wno-UNDRIVEN"],
            always=True,
            waves=True,
        )
        runner.test(
            test_module=toplevel.replace("tb_", "test_"),
            hdl_toplevel=toplevel,
            test_dir=TB,
            build_dir=build,
            waves=True,
            results_xml=f"{toplevel}.xml",
        )
        # El simulador siempre escribe la onda en tb/dump.vcd, así que sin
        # esto cada bloque pisa la del anterior.
        vcd = TB / "dump.vcd"
        if vcd.exists():
            vcd.replace(TB / f"{toplevel}.vcd")


if __name__ == "__main__":
    main()
