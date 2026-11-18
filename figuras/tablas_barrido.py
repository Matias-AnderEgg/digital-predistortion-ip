#!/usr/bin/env python3
"""Tablas del apéndice del barrido (Apéndice ap:barrido), en LaTeX.

Uso:
    python figuras/tablas_barrido.py

Lee sys_with_acpr/resultados_barrido/barrido_mp.pkl, el mismo que usan las
figuras barrido_* de genera_figuras.py, y deja en esta carpeta
tabla_barrido_evm.tex y tabla_barrido_acpr.tex, con los 120 puntos del
barrido. Cada archivo es un tabular que el apéndice incluye con \\input.

Las filas son las ocho configuraciones de cada método. M = 3, P = 7 está en
los dos cortes y va una sola vez, en negrita la de GD, que es la elegida. Las
configuraciones de GD en las que entró el respaldo por NLMS en los siete
back-off van marcadas (con la señal LTE, P = 9, P = 11, M = 5 y M = 7).
"""

from __future__ import annotations

import pickle
from pathlib import Path

AQUI = Path(__file__).resolve().parent
REPO = AQUI.parent
BARRIDO_PKL = REPO / "sys_with_acpr" / "resultados_barrido" / "barrido_mp.pkl"

# (M, P) en el orden de las filas: el corte de memoria y después el de orden.
CONFIGS = [(1, 7), (3, 7), (5, 7), (7, 7), (3, 3), (3, 5), (3, 9), (3, 11)]
METODOS = [("ls", "LS"), ("gd", "GD")]


def _acpr_peor(r) -> float:
    return max(r["acpr_l"], r["acpr_u"])


def _evm(v: float) -> str:
    """Tres cifras significativas, como en el texto: dos decimales debajo de
    10 %, uno hasta 100 % y ninguno de ahí para arriba."""
    if v < 9.995:
        s = f"{v:.2f}"
    elif v < 100:
        s = f"{v:.1f}"
    else:
        s = f"{v:.0f}"
    return s.replace(".", "{,}")


def _acpr(v: float) -> str:
    return f"${v:.1f}$".replace(".", "{,}")


def _tabla(res: dict, bos: list, valor, fmt, con_ideal: bool) -> str:
    cols = "lll" + "r" * len(bos)
    cab = " & ".join(f"${b:+d}$" if b else "0" for b in bos)
    lineas = [
        f"\\begin{{tabular}}{{{cols}}}",
        "\\hline",
        f"Método & $M$ & $P$ & \\multicolumn{{{len(bos)}}}{{c}}{{Back-off [dB]}} \\\\",
        f"\\cline{{4-{3 + len(bos)}}}",
        f" & & & {cab} \\\\",
        "\\hline",
    ]
    if con_ideal:
        v = fmt(valor(res[("ideal",)]))
        lineas.append("Señal ideal & & & " + " & ".join([v] * len(bos)) + " \\\\")
    fila = " & ".join(fmt(valor(res[("no_dpd", b)])) for b in bos)
    lineas.append(f"Sin DPD & & & {fila} \\\\")
    for clave, nombre in METODOS:
        lineas.append("\\hline")
        for i, (M, P) in enumerate(CONFIGS):
            celdas = [fmt(valor(res[(clave, M, P, b)])) for b in bos]
            respaldo = all(res[(clave, M, P, b)].get("pa_respaldo") for b in bos)
            p_txt = f"{P}$^{{\\dagger}}$" if respaldo else f"{P}"
            m_txt = nombre if i == 0 else ""
            if clave == "gd" and (M, P) == (3, 7):
                # \boldmath para que el signo menos del ACPR también salga en negrita
                celdas = [f"{{\\bfseries\\boldmath {c}}}" for c in celdas]
                p_txt, m_lab = f"\\textbf{{{P}}}", f"\\textbf{{{M}}}"
            else:
                m_lab = f"{M}"
            lineas.append(f"{m_txt} & {m_lab} & {p_txt} & " + " & ".join(celdas) + " \\\\")
    lineas += ["\\hline", "\\end{tabular}", ""]
    return "\n".join(lineas)


def main() -> None:
    with open(BARRIDO_PKL, "rb") as f:
        d = pickle.load(f)
    bos = list(d["meta"]["BO_LIST"])
    res = d["res"]
    encabezado = "% Generado por figuras/tablas_barrido.py desde barrido_mp.pkl. No editar.\n"
    (AQUI / "tabla_barrido_evm.tex").write_text(
        encabezado + _tabla(res, bos, lambda r: r["evm"], _evm, con_ideal=False))
    # EVM con el receptor escalar, la que usa la elección de la configuración.
    (AQUI / "tabla_barrido_evm_escalar.tex").write_text(
        encabezado + _tabla(res, bos, lambda r: r["evm_escalar"], _evm, con_ideal=False))
    (AQUI / "tabla_barrido_acpr.tex").write_text(
        encabezado + _tabla(res, bos, _acpr_peor, _acpr, con_ideal=True))
    print("tabla_barrido_evm.tex, tabla_barrido_evm_escalar.tex y tabla_barrido_acpr.tex escritas en", AQUI)


if __name__ == "__main__":
    main()
