"""Genera gd_superficie.tex: descenso de gradiente sobre un costo cuadrático.

Dos paneles con el mismo costo y el mismo punto inicial, uno con mu chico y
otro con mu grande. Las trayectorias son la Ec. de actualización calculada,
w <- w - mu * grad J, y en cada panel se dibuja la curva de nivel que pasa por
cada punto de la trayectoria, así se ve que el costo baja en cada paso y que
cada paso sale perpendicular a la curva del punto donde arranca.

El encuadre se calcula: los ejes arrancan justo a la izquierda y debajo de lo
dibujado, para no dejar blanco y que la figura entre más grande en el ancho.

Uso: python gd_superficie_gen.py   (escribe gd_superficie.tex al lado)
"""
import os
import numpy as np

ANG = 38                               # inclinación del valle, en grados
TH = np.deg2rad(ANG)
R = np.array([[np.cos(TH), -np.sin(TH)], [np.sin(TH), np.cos(TH)]])
LAM = np.array([1.0, 4.0])            # valle alargado: curvatura 1 y 4
H = R @ np.diag(LAM) @ R.T
D0 = R @ np.array([-2.2, 0.7])         # w(0) - w_opt
LG = 0.75                              # largo de la flecha del gradiente

PANELES = [  # (rótulo, mu, pasos)
    ("(a) $\\mu$ chico", 0.08, 6),
    ("(b) $\\mu$ grande", 0.24, 6),
]
MARGEN = 0.35                          # entre lo dibujado y los ejes
SEP = 0.9                              # entre el fin de un panel y el eje del otro


def trayectoria(mu, n):
    d, pts = D0.copy(), [D0.copy()]
    for _ in range(n):
        d = d - mu * 2 * H @ d
        pts.append(d.copy())
    return np.array(pts)


J0 = D0 @ H @ D0
# caja de la elipse exterior (la que pasa por w(0)), relativa a w_opt
sa, sb = np.sqrt(J0 / LAM[0]), np.sqrt(J0 / LAM[1])
hx = np.hypot(sa * np.cos(TH), sb * np.sin(TH))
hy = np.hypot(sa * np.sin(TH), sb * np.cos(TH))
g = H @ D0
g /= np.linalg.norm(g)
G = D0 + LG * g                        # punta de la flecha del gradiente
xmin = min(-hx, G[0] - 0.25)           # 0.25: lugar para el rótulo
ymin = min(-hy, D0[1] - 0.3)           # 0.3: lugar para el rótulo de w(0)
C = np.array([MARGEN - xmin, MARGEN - ymin])   # w_opt respecto del origen del panel
ancho = C[0] + hx + 0.3
alto = C[1] + hy + 0.2
DX = ancho + SEP


def p(v, dx):
    return f"({v[0] + dx:.3f},{v[1]:.3f})"


lineas = []
for i, (rot, mu, n) in enumerate(PANELES):
    dx = i * DX
    pts = C + trayectoria(mu, n)
    c = C + np.array([dx, 0.0])
    lineas.append(f"  % ---------------- {rot}, mu = {mu} ----------------")
    for w in pts:  # curva de nivel por cada punto
        L = (w - C) @ H @ (w - C)
        lineas.append(f"  \\draw[black!30, rotate around={{{ANG}:({c[0]:.3f},{c[1]:.3f})}}] "
                      f"({c[0]:.3f},{c[1]:.3f}) ellipse ({np.sqrt(L / LAM[0]):.3f} and {np.sqrt(L / LAM[1]):.3f});")
    lineas.append(f"  \\fill ({c[0]:.3f},{c[1]:.3f}) circle (2pt);")
    lineas.append(f"  \\node[font=\\scriptsize, anchor=west] at ({c[0] + 0.12:.3f},{c[1] - 0.05:.3f}) "
                  f"{{$\\mathbf{{w}}_{{\\mathrm{{opt}}}}$}};")
    for a, b in zip(pts[:-1], pts[1:]):
        lineas.append(f"  \\draw[-{{Latex[length=1.4mm]}}, thick] {p(a, dx)} -- {p(b, dx)};")
    lineas.append("  \\foreach \\q in {" + ",".join(p(w, dx) for w in pts) + "} \\fill \\q circle (1.2pt);")
    lineas.append(f"  \\draw[flecha, densely dashed, black!55] {p(C + D0, dx)} -- {p(C + G, dx)}"
                  f" node[anchor=south, font=\\scriptsize, black!55] {{$\\nabla J$}};")
    lineas.append(f"  \\node[font=\\scriptsize, anchor=north] at {p(C + D0 + np.array([0.05, -0.12]), dx)} "
                  f"{{$\\mathbf{{w}}(0)$}};")
    lineas.append(f"  \\node[font=\\small, anchor=south west] at ({dx + 0.25:.3f},{alto:.3f}) {{{rot}}};")
    lineas.append(f"  \\draw[-{{Latex[length=2mm]}}, thick] ({dx:.2f},0) -- ({dx + ancho:.2f},0) "
                  f"node[below=1mm] {{$w_1$}};")
    lineas.append(f"  \\draw[-{{Latex[length=2mm]}}, thick] ({dx:.2f},0) -- ({dx:.2f},{alto:.2f}) "
                  f"node[left=1mm] {{$w_2$}};")

cab = f"""% Descenso de gradiente sobre un costo cuadrático (Sec. de descenso de
% gradiente, Cap. 5). Generado por figuras/gd_superficie_gen.py, no editar a mano.
% Costo J(w) = (w - w_opt)^T H (w - w_opt), H = R({ANG}°) diag(1, 4) R({ANG}°)^T,
% w(0) = w_opt + R({ANG}°) (-2.2, 0.7). Panel (a) mu = {PANELES[0][1]}, panel (b)
% mu = {PANELES[1][1]}. En cada panel, una curva de nivel por cada punto de la
% trayectoria, y el gradiente en w(0), normal a su curva.
% Requiere: tikz + estilos de ibextra.sty (flecha).
"""
tikz = cab + "\\begin{tikzpicture}[font=\\small, x=1cm, y=1cm]\n" + "\n".join(lineas) + "\n\\end{tikzpicture}\n"
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gd_superficie.tex"), "w") as f:
    f.write(tikz)
print(f"ancho total {DX + ancho:.2f}, alto {alto:.2f}")
for rot, mu, n in PANELES:
    t = trayectoria(mu, n)
    print(rot, mu, "J:", " ".join(f"{w @ H @ w:.2f}" for w in t))
