"""Tablas de recursos y tiempos para la tesis, a partir de los reportes de Vivado.

Lee los reportes que dejan sintesis.tcl, implementacion.tcl y placa.tcl en
reportes/ (se generan en la Windows, y estos cinco se versionan):

  impl_utilizacion.rpt, impl_tiempos.rpt   el IP solo, implementado fuera de
                                           contexto en la Zynq-7020 de la
                                           placa (xc7z020clg400-1). Los de la
                                           Artix-7 100T tienen el sufijo _artix
  placa_utilizacion.rpt, placa_tiempos.rpt el sistema de la Red Pitaya:
                                           IP, reproductor con sus BRAM,
                                           interconexion AXI y PS (xc7z020)
  utilizacion_jerarquica.rpt               el IP sintetizado, por módulo

y escribe en resultados/ dos tablas en LaTeX, listas para \\input:

  recursos.tex   LUT, FF, DSP48 y BRAM de los dos disenos, y el margen a 10 ns
  dsp.tex        de donde salen los 162 DSP48

Uso: python rtl/synth/tabla_recursos.py
"""
import os
import re
from collections import defaultdict

AQUI = os.path.dirname(os.path.abspath(__file__))
REPORTES = os.path.join(AQUI, "reportes")
SALIDA = os.path.join(AQUI, "resultados")
PERIODO_NS = 10.0
N_BASE = 3 * 3      # M taps por los órdenes 3, 5 y 7


def leer(nombre):
    with open(os.path.join(REPORTES, nombre)) as f:
        return f.read()


def fila(texto, recurso):
    """Usados y disponibles de la primera fila '| recurso | ...' del reporte."""
    m = re.search(rf"^\|\s*{re.escape(recurso)}\s*\|\s*([\d.]+)\s*\|\s*\d+\s*\|\s*\d+\s*\|\s*(\d+)\s*\|",
                  texto, re.M)
    if not m:
        raise ValueError(f"no esta la fila {recurso!r}")
    return float(m.group(1)), int(m.group(2))


def dispositivo(texto):
    return re.search(r"^\| Device\s*:\s*(\S+)", texto, re.M).group(1)


def familia(parte):
    """Nombre corto de la parte para el encabezado de la tabla."""
    for prefijo, nombre in [("xc7z020", "Zynq-7020"), ("xc7a100t", "Artix-7 100T")]:
        if parte.startswith(prefijo):
            return nombre
    raise ValueError(f"parte desconocida {parte}")


def wns(texto):
    """Margen de setup del resumen de tiempos (Design Timing Summary)."""
    m = re.search(r"WNS\(ns\).*\n\s*-+.*\n\s*(-?[\d.]+)", texto)
    return float(m.group(1))


def resumen(util, tiempos):
    t, u = leer(util), leer(tiempos)
    r = {"dispositivo": dispositivo(t)}
    for clave, recurso in [("LUT", "Slice LUTs"), ("FF", "Slice Registers"),
                           ("DSP48", "DSPs"), ("BRAM", "Block RAM Tile")]:
        r[clave] = fila(t, recurso)
    r["wns"] = wns(u)
    return r


def dsp_por_funcion():
    """Suma los DSP del IP sintetizado por tipo de módulo.

    En la jerarquia, (u_core) son los DSP que quedan en dpd_apply_core
    mismo: los productos de la base x * |x|^(p-1), uno por tap y por orden
    no lineal, o sea N_BASE."""
    t = leer("utilizacion_jerarquica.rpt")
    t = t[t.index("Utilization by Hierarchy"):]
    cuenta = defaultdict(lambda: [0, 0])
    for linea in t.splitlines():
        c = [x.strip() for x in linea.split("|")]
        if len(c) < 12 or not c[-2].isdigit():
            continue
        instancia, modulo, dsp = c[1], c[2], int(c[-2])
        if instancia.startswith("("):
            if instancia == "(u_core)":
                cuenta["base"] = [N_BASE, dsp]
            continue
        if modulo.startswith("cmul_round_sat"):
            cuenta["cmul"][0] += 1
            cuenta["cmul"][1] += dsp
        elif modulo.startswith("mp_magnitude"):
            cuenta["mag"][0] += 1
            cuenta["mag"][1] += dsp
        elif instancia == "dpd_apply_top":
            total = dsp
    return cuenta, total


def num(v):
    """Entero con separador de miles fino desde cinco cifras, como en el resto
    de la tesis: 7357 pero 10\\,564."""
    v = int(v)
    return f"{v:,}".replace(",", "\\,") if v >= 10000 else str(v)


def coma(v, dec=1):
    return f"{v:.{dec}f}".replace(".", ",")


def main():
    ip = resumen("impl_utilizacion.rpt", "impl_tiempos.rpt")
    placa = resumen("placa_utilizacion.rpt", "placa_tiempos.rpt")

    print(f"{'':12s}{'IP solo':>24s}{'sistema en la placa':>26s}")
    print(f"{'dispositivo':12s}{ip['dispositivo']:>24s}{placa['dispositivo']:>26s}")
    for k in ["LUT", "FF", "DSP48", "BRAM"]:
        a, b = ip[k], placa[k]
        print(f"{k:12s}{a[0]:>10.0f} de {a[1]:>6d} ({100 * a[0] / a[1]:4.1f} %)"
              f"{b[0]:>10.0f} de {b[1]:>6d} ({100 * b[0] / b[1]:4.1f} %)")
    for nombre, r in [("IP solo", ip), ("placa", placa)]:
        print(f"{nombre}: margen {r['wns']:.3f} ns a {PERIODO_NS:g} ns, "
              f"fmax {1e3 / (PERIODO_NS - r['wns']):.1f} MHz")

    os.makedirs(SALIDA, exist_ok=True)
    with open(os.path.join(SALIDA, "recursos.tex"), "w") as f:
        f.write("% Generado por rtl/synth/tabla_recursos.py. No editar.\n")
        f.write("\\begin{tabular}{lrrrr}\n\\hline\n")
        f.write(f" & \\multicolumn{{2}}{{c}}{{N\\'ucleo solo, {familia(ip['dispositivo'])}}} & "
                f"\\multicolumn{{2}}{{c}}{{Sistema, {familia(placa['dispositivo'])}}} \\\\\n")
        f.write("\\cline{2-3} \\cline{4-5}\n")
        f.write("Recurso & Usados & \\% & Usados & \\% \\\\\n\\hline\n")
        for k, nombre in [("LUT", "LUT"), ("FF", "Flip-flops"), ("DSP48", "DSP48E1"),
                          ("BRAM", "BRAM de 36 kb")]:
            a, b = ip[k], placa[k]
            f.write(f"{nombre} & {num(a[0])} & {coma(100 * a[0] / a[1])} & "
                    f"{num(b[0])} & {coma(100 * b[0] / b[1])} \\\\\n")
        f.write("\\hline\n")
        f.write(f"Margen a {PERIODO_NS:g}\\,ns & \\multicolumn{{2}}{{r}}{{{coma(ip['wns'], 2)}\\,ns}} & "
                f"\\multicolumn{{2}}{{r}}{{{coma(placa['wns'], 2)}\\,ns}} \\\\\n")
        f.write(f"$f_\\mathrm{{max}}$ & \\multicolumn{{2}}{{r}}{{{coma(1e3 / (PERIODO_NS - ip['wns']))}\\,MHz}} & "
                f"\\multicolumn{{2}}{{r}}{{{coma(1e3 / (PERIODO_NS - placa['wns']))}\\,MHz}} \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    cuenta, total = dsp_por_funcion()
    filas = [("mag", "Potencias del m\\'odulo, $|x|^2$ a $|x|^6$"),
             ("base", "Base, $x\\,|x|^{p-1}$"),
             ("cmul", "Producto por el coeficiente, $w_{p,m}\\,x\\,|x|^{p-1}$")]
    print("DSP48 por funcion:")
    for k, nombre in filas:
        n, d = cuenta[k]
        print(f"  {k:5s} {n:2d} bloques, {d // n:2d} cada uno, {d:3d}")
    suma = sum(cuenta[k][1] for k, _ in filas)
    print(f"  total {suma} (el reporte dice {total})")
    assert suma == total
    with open(os.path.join(SALIDA, "dsp.tex"), "w") as f:
        f.write("% Generado por rtl/synth/tabla_recursos.py. No editar.\n")
        f.write("\\begin{tabular}{lrrr}\n\\hline\n")
        f.write("Funci\\'on & Instancias & DSP48 por instancia & Total \\\\\n\\hline\n")
        for k, nombre in filas:
            n, d = cuenta[k]
            f.write(f"{nombre} & {n} & {d // n} & {d} \\\\\n")
        f.write(f"\\hline\nTotal & & & {suma} \\\\\n\\hline\n\\end{{tabular}}\n")
    print(f"tablas en {SALIDA}")


if __name__ == "__main__":
    main()
