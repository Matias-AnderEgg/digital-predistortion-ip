"""Lazo vuelta a vuelta con la señal LTE de la tesis: la placa aplica y
entrena, la Mac hace de PA.

  - cada vuelta pasa una subtrama nueva (14 símbolos, 61440 muestras) de la
    señal de evaluación de la tesis (semilla 456, la de lte_placa.py), por el
    IP de a trozos (lazo_lte.c);
  - el modelo del PA se identifica una sola vez, en la vuelta 0, que es sin
    DPD, con las primeras --modelo muestras de esa subtrama (16000, como el
    entrenamiento fuera de línea);
  - en cada vuelta el ARM hace una pasada del DPD sobre las primeras
    --muestras muestras de la subtrama, partiendo de los coeficientes
    vigentes, y los carga divididos por el RMS de la salida del DPD sobre la
    subtrama entera, para que el PA reciba la misma potencia media que sin
    DPD.

La medición de cada vuelta es la de lte_placa.py (NMSE con la ganancia
alineada, EVM con el ecualizador por subportadora sobre los 14 símbolos de
la subtrama, ACPR del adyacente y del alterno).

Con la placa, la Mac repite cada paso con el mismo C y el modelo de Python
del IP, y verifica en cada vuelta que la salida del IP y los coeficientes
que entrenó el ARM sean los mismos, código por código.

Uso:
  python lazo_lte.py --sin-placa [--bo -9] [--vueltas 15] [--muestras 8000]
  python lazo_lte.py --bo -9                 con la placa por USB

Deja resultados/lte_bo<BO>_n<muestras>/vueltas.csv y, en la terminal, una
línea por vuelta.
"""
import argparse
import ctypes
import hashlib
import os
import re
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.abspath(os.path.join(AQUI, "..", ".."))
sys.path.insert(0, os.path.join(RAIZ, "sw", "hw_test"))

import entrenamiento as E  # noqa: E402
import lte_placa as LP  # noqa: E402

T, GV, fxp = E.T, E.GV, E.fxp
N_SIM = 14
N_BLOQUE = LP.lte.largo(N_SIM)     # 61440, igual que lazo_lte.c
G_Y = 8.0                          # escala con que viaja y, igual que lazo_lte.c
SALIDA = os.path.join(AQUI, "salida")
RESULTADOS = os.path.join(AQUI, "resultados")
REMOTO = "/root/dpd_hw"


def subtramas(n):
    """Las primeras n subtramas de la senal de evaluacion de la tesis."""
    x, grilla = LP.senal(456)
    assert n * N_BLOQUE <= len(x)
    return [(x[k * N_BLOQUE:(k + 1) * N_BLOQUE], grilla[k * N_SIM:(k + 1) * N_SIM])
            for k in range(n)]


def codigos_x(x):
    xi, xq = GV.to_code(fxp.quantize(x / GV.G, GV.CFG.data, GV.CFG))
    return E.empaquetar(xi, xq).astype("<u4")


def de_codigos(c, g):
    re_, im_ = E.desempaquetar(c)
    return (re_ + 1j * im_) * g / 32768.0


def codigos_y(y):
    re_ = np.clip(np.round(y.real / G_Y * 32768), -32768, 32767).astype(np.int64)
    im_ = np.clip(np.round(y.imag / G_Y * 32768), -32768, 32767).astype(np.int64)
    return E.empaquetar(re_, im_).astype("<u4")


def pa(u, bo):
    """El PA con ganancia fija: la potencia media la mantiene el ARM al
    cargar w / c."""
    kw = dict(T.pa_kwargs(LP.PARAMS), bo_db=bo)
    y, _, _ = T.apply_selected_pa(u, **kw, normalize_input=False, normalize_output=False)
    return y


class EnMac:
    def __init__(self, x_codigos):
        self.lib = T.cargar_lib()
        self.cfg = T.Cfg()
        self.lib.dpd_train_cfg_default(ctypes.byref(self.cfg))
        self.x = [de_codigos(c, GV.G) for c in x_codigos]
        self.w = np.zeros(12, np.complex128)
        self.h = np.zeros(12, np.complex128)
        self.c = 1.0

    def codigos_w(self):
        re_ = np.zeros(12, np.int16)
        im_ = np.zeros(12, np.int16)
        self.lib.dpd_coef_to_q15(np.ascontiguousarray(self.w / self.c), re_, im_)
        return E.empaquetar(re_, im_).astype("<u4")

    def inicio(self):
        self.w[:] = 0
        self.w[0] = 1
        self.c = 1.0
        return self.codigos_w()

    def aplicar(self, k):
        re_, im_ = E.desempaquetar(self.codigos_w())
        ui, uq = E.salida_esperada(self.x[k], (re_ + 1j * im_) / 32768.0)
        return E.empaquetar(ui, uq).astype("<u4")

    def modelo(self, u_cod, y_cod, n):
        self.cfg.warm_start = 0
        u = np.ascontiguousarray(de_codigos(u_cod[:n], GV.G_OUT))
        y = np.ascontiguousarray(de_codigos(y_cod[:n], G_Y))
        ep = self.lib.dpd_train_pa_model(u, y, n, ctypes.byref(self.cfg), self.h, np.zeros(64))
        return f"modelo en la Mac, {n} muestras, {abs(ep)} epocas{' (con el respaldo)' if ep < 0 else ''}"

    def entrenar(self, k, n):
        self.cfg.warm_start = 1
        self.cfg.ep_dpd = 1
        t = time.time()
        self.lib.dpd_train_dla(np.ascontiguousarray(self.x[k][:n]), n, self.h,
                               ctypes.byref(self.cfg), self.w, np.zeros(64))
        self.c = self.lib.dpd_rms_salida(np.ascontiguousarray(self.x[k]), N_BLOQUE, self.w,
                                         ctypes.byref(self.cfg))
        return self.codigos_w(), time.time() - t


class EnPlaca:
    def __init__(self, x_codigos):
        import consola
        self.c = consola.Consola()
        archivos = {
            "lazo_lte.c": os.path.join(AQUI, "lazo_lte.c"),
            "dpd_train.c": os.path.join(RAIZ, "sw", "dpd_train", "dpd_train.c"),
            "dpd_train.h": os.path.join(RAIZ, "sw", "dpd_train", "dpd_train.h"),
            "ip.c": os.path.join(RAIZ, "sw", "hw_test", "ip.c"),
            "ip.h": os.path.join(RAIZ, "sw", "hw_test", "ip.h"),
            "placa_mapa.h": os.path.join(RAIZ, "sw", "hw_test", "placa_mapa.h"),
        }
        self.cmd(f"mkdir -p {REMOTO}")
        for nombre, local in archivos.items():
            if not self.c.enviar(local, f"{REMOTO}/{nombre}"):
                sys.exit(f"no llego bien {nombre}")
        # Las subtramas son unos MB por la consola: si la placa ya las tiene, no
        # se vuelven a mandar.
        x_bin = os.path.join(SALIDA, "x_subtramas.bin")
        np.concatenate(x_codigos).astype("<u4").tofile(x_bin)
        md5 = hashlib.md5(open(x_bin, "rb").read()).hexdigest()
        if md5 not in self.cmd("md5sum x_bloques.bin || true"):
            t = time.time()
            if not self.c.enviar(x_bin, f"{REMOTO}/x_bloques.bin"):
                sys.exit("no llego bien x_bloques.bin")
            print(f"subtramas enviadas en {time.time() - t:.0f} s", flush=True)
        self.cmd("gcc -O2 -std=gnu99 -Wall -ffp-contract=off -o lazo_lte lazo_lte.c ip.c "
                 "dpd_train.c -lm")

    def cmd(self, linea):
        codigo, salida = self.c.cmd(f"cd {REMOTO} && {linea} 2>&1")
        if codigo != 0:
            sys.exit(f"fallo en la placa: {linea}\n{salida}")
        return salida

    def w_de(self, salida):
        m = re.search(r"^w((?: [0-9a-f]{8}){12})", salida, re.M)
        return np.array([int(v, 16) for v in m.group(1).split()], dtype="<u4")

    def inicio(self):
        return self.w_de(self.cmd("./lazo_lte inicio"))

    def aplicar(self, k):
        self.cmd(f"./lazo_lte aplicar {k}")
        local = os.path.join(SALIDA, "u.bin")
        self.c.traer(f"{REMOTO}/u.bin", local)
        return np.fromfile(local, dtype="<u4")

    def modelo(self, u_cod, y_cod, n):
        """u_cod no se usa: la placa tiene la suya en u.bin."""
        local = os.path.join(SALIDA, "y.bin")
        y_cod[:n].astype("<u4").tofile(local)
        if not self.c.enviar(local, f"{REMOTO}/y.bin"):
            sys.exit("no llego bien y.bin")
        return self.cmd(f"./lazo_lte modelo {n}").strip()

    def entrenar(self, k, n):
        salida = self.cmd(f"./lazo_lte entrenar {k} {n}")
        t = float(re.search(r"entrenar ([0-9.]+) s", salida).group(1))
        return self.w_de(salida), t


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sin-placa", action="store_true")
    ap.add_argument("--vueltas", type=int, default=15)
    ap.add_argument("--muestras", type=int, default=8000)
    ap.add_argument("--modelo", type=int, default=16000)
    ap.add_argument("--bo", type=float, default=-9.0)
    args = ap.parse_args()
    n_ent = max(1, min(args.muestras, N_BLOQUE))
    n_mod = max(1, min(args.modelo, N_BLOQUE))
    global SALIDA, RESULTADOS
    nombre = f"lte_bo{args.bo:g}_n{n_ent}"
    SALIDA = os.path.join(SALIDA, nombre)
    RESULTADOS = os.path.join(RESULTADOS, nombre)
    os.makedirs(SALIDA, exist_ok=True)

    senal = subtramas(args.vueltas)
    x_cod = [codigos_x(x) for x, _ in senal]
    sombra = EnMac(x_cod)
    motor = sombra if args.sin_placa else EnPlaca(x_cod)
    print("todo en la Mac" if args.sin_placa else "placa lista", flush=True)

    w_m = motor.inicio()
    if motor is not sombra:
        assert np.array_equal(w_m, sombra.inicio())

    filas, fallas = [], 0
    for k in range(args.vueltas):
        x, grilla = senal[k]
        t0 = time.time()
        u_cod = motor.aplicar(k)
        if motor is not sombra:
            iguales = int(np.sum(u_cod == sombra.aplicar(k)))
            fallas += N_BLOQUE - iguales
        u = de_codigos(u_cod, GV.G_OUT)
        p_pa = 10 * np.log10(np.mean(np.abs(u) ** 2))
        y = pa(u, args.bo)
        nm, evm, acpr, alt, _, _ = LP.metricas(x, grilla, y)
        if k == 0:
            y_cod = codigos_y(y)
            print("  " + motor.modelo(u_cod, y_cod, n_mod), flush=True)
            if motor is not sombra:
                sombra.modelo(u_cod, y_cod, n_mod)
        w_m, t_ent = motor.entrenar(k, n_ent)
        linea = (f"vuelta {k:2d}: NMSE {nm:6.2f} dB, EVM {evm:5.2f} %, ACPR {acpr:6.1f} dB, "
                 f"alterno {alt:6.1f} dB, PA a {p_pa:+5.2f} dB, entrenar {t_ent * 1e3:4.0f} ms")
        if motor is not sombra:
            w_s, _ = sombra.entrenar(k, n_ent)
            igual_w = int(np.sum(w_m == w_s))
            fallas += 12 - igual_w
            linea += f", IP {iguales}/{N_BLOQUE} y w {igual_w}/12 iguales a la Mac"
        linea += f", vuelta {time.time() - t0:.1f} s"
        print(linea, flush=True)
        filas.append((k, nm, evm, acpr, alt, p_pa, t_ent))

    os.makedirs(RESULTADOS, exist_ok=True)
    with open(os.path.join(RESULTADOS, "vueltas.csv"), "w") as fh:
        fh.write("vuelta,nmse_db,evm_pct,acpr_db,alterno_db,p_pa_db,t_entrenar_s\n")
        for fila in filas:
            fh.write("%d,%.3f,%.3f,%.2f,%.2f,%.3f,%.4f\n" % fila)
    print(f"resultados en {RESULTADOS}")
    if motor is not sombra:
        print("TODO IGUAL A LA MAC" if fallas == 0 else f"HAY {fallas} DIFERENCIAS CON LA MAC")


if __name__ == "__main__":
    main()
