"""EVM de la validación en la placa con el receptor escalar, sin tocar el repo.

Parchea lte_placa.ecualizar con el escalar de evm_rms_percent (el de las
simulaciones) y repite:
  1) separar a -9 y +3 dB (subtrama y 364 símbolos, flotante y punto fijo);
  2) la salida real de la placa (salida_bo*.bin);
  3) el lazo vuelta a vuelta en la Mac, comprobando el NMSE contra vueltas.csv;
  4) el modelo del PA en Q0.15 (experimentos/h_q15_lte.py).
Uso: PYTHONDONTWRITEBYTECODE=1 python figuras/cap7_evm_escalar.py

Antes de parchear mide una vez con el ecualizador para comprobar que da lo
mismo que lo publicado."""
import csv
import os
import sys

import numpy as np

# La raíz del repositorio, calculada desde este archivo (como figuras_placa.py).
RAIZ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "sw", "hw_test"))
sys.path.insert(0, os.path.join(RAIZ, "sw", "lazo_simulado"))
sys.path.insert(0, os.path.join(RAIZ, "experimentos"))
os.environ.setdefault("MPLBACKEND", "Agg")

import entrenamiento as E  # noqa: E402
import lte_placa as LP  # noqa: E402
import lazo_lte as LL  # noqa: E402
import h_q15_lte as HQ  # noqa: E402

GV = E.GV
ECUALIZADOR = LP.ecualizar


def escalar(ref, rx):
    """Como evm_rms_percent de sys_with_acpr/rx.py, la EVM escalar de las
    simulaciones: un coeficiente complejo que ajusta rx a ref por mínimos cuadrados."""
    return rx * (np.vdot(rx, ref) / np.vdot(rx, rx))


def placa(bo):
    d = np.load(os.path.join(LP.DIR, f"esperada_bo{bo:g}.npz"))
    hw = np.fromfile(os.path.join(LP.DIR, f"salida_bo{bo:g}.bin"), dtype="<u4")
    assert np.array_equal(hw, d["salida"])
    re_, im_ = E.desempaquetar(hw)
    u_hw = (re_ + 1j * im_) / 32768.0 * GV.G_OUT
    x, grilla = d["x"], d["grilla"]
    for nombre, y in (("sin DPD", LP.pa(x, bo, True)), ("núcleo en la placa", LP.pa(u_hw, bo, True))):
        nm, ev, ac, al, _, _ = LP.metricas(x, grilla, y)
        print(f"  placa {bo:+g} dB {nombre:20s} NMSE {nm:7.2f}  EVM {ev:5.2f} %  ACPR {ac:6.1f}  alt {al:6.1f}")


def lazo(bo, vueltas=15, n_ent=2000, n_mod=16000):
    ruta = os.path.join(RAIZ, "sw", "lazo_simulado", "resultados", f"lte_bo{bo:g}_n{n_ent}", "vueltas.csv")
    with open(ruta) as fh:
        filas = list(csv.DictReader(fh))
    senal = LL.subtramas(vueltas)
    m = LL.EnMac([LL.codigos_x(x) for x, _ in senal])
    m.inicio()
    nm_l, ev_l, ev_eq = [], [], []
    for k in range(vueltas):
        x, grilla = senal[k]
        u_cod = m.aplicar(k)
        y = LL.pa(LL.de_codigos(u_cod, GV.G_OUT), bo)
        LP.ecualizar = ECUALIZADOR
        ev_eq.append(LP.metricas(x, grilla, y)[1])
        LP.ecualizar = escalar
        nm, ev, *_ = LP.metricas(x, grilla, y)
        nm_l.append(nm)
        ev_l.append(ev)
        if k == 0:
            m.modelo(u_cod, LL.codigos_y(y), n_mod)
        m.entrenar(k, n_ent)
    nm_csv = np.array([float(f["nmse_db"]) for f in filas])
    ev_csv = np.array([float(f["evm_pct"]) for f in filas])
    print(f"  lazo {bo:+g} dB: dif. NMSE con el CSV {np.max(np.abs(np.array(nm_l) - nm_csv)):.4f} dB, "
          f"dif. EVM ecualizada con el CSV {np.max(np.abs(np.array(ev_eq) - ev_csv)):.4f} %")
    print(f"    vuelta 0: NMSE {nm_l[0]:.2f}, EVM escalar {ev_l[0]:.2f} %")
    print(f"    vueltas 10 a 14: NMSE {np.mean(nm_csv[10:]):.2f}, EVM escalar {np.mean(ev_l[10:]):.2f} % "
          f"(ecualizada {np.mean(ev_csv[10:]):.2f} %)")
    print("    EVM escalar por vuelta: " + " ".join(f"{v:.2f}" for v in ev_l))


def h_q15():
    x, grilla = LP.senal(456)
    casos = [("sin DPD", x)]
    for nombre, cambios in (("h flotante", {}), ("h Q0.15", {"quantize_h": 1})):
        w = HQ.entrenar(**cambios)
        casos.append((nombre, HQ.dpd.apply_memory_poly_fxp(x, w, Pmax=GV.PMAX, M=GV.M_TAPS, cfg=GV.CFG)))
    for nombre, u in casos:
        nm, ev, ac, al, _, _ = LP.metricas(x, grilla, LP.pa(u, HQ.BO, True))
        print(f"  {nombre:12s} NMSE {nm:7.2f}  EVM {ev:5.2f} %  ACPR {ac:6.1f}  alt {al:6.1f}")


if __name__ == "__main__":
    print("== Con el ecualizador (control, tiene que dar lo publicado)")
    placa(-9.0)
    print("== Receptor escalar")
    LP.ecualizar = escalar
    for bo in (-9.0, 3.0):
        print(f"-- separar {bo:+g} dB")
        LP.separar(bo)
    print("-- salida real de la placa")
    for bo in (-9.0, 3.0):
        placa(bo)
    print("-- lazo vuelta a vuelta")
    for bo in (-9.0, 3.0):
        lazo(bo)
    print("-- modelo del PA en Q0.15, -9 dB, 364 símbolos")
    LP.ecualizar = escalar
    h_q15()
