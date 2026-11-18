"""El entrenamiento en C no tiene que depender de G.

Corre el mismo C con G = 1 y con G = 8 sobre la señal de los vectores
dorados (semilla 123, BO = -9 dB) y compara h y w. Con la convención
normalizada salen iguales, porque G es potencia de 2. Para eso precond_delta
tiene que escalar con G.

Uso: python test/escala_g.py [libdpd_train] [n]
Sin argumentos usa build/libdpd_train y 16000 muestras.
"""
import ctypes
import os
import sys

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import test_dpd_train as T  # noqa: E402

LSB = 2.0 ** -15


def main():
    nombre = "libdpd_train.dylib" if sys.platform == "darwin" else "libdpd_train.so"
    ruta = sys.argv[1] if len(sys.argv) > 1 else os.path.join(AQUI, "..", "build", nombre)
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16000
    lib = ctypes.CDLL(os.path.abspath(ruta))
    pc = np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS")
    pd = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
    lib.dpd_train_cfg_default.argtypes = [ctypes.POINTER(T.Cfg)]
    lib.dpd_train_pa_model.argtypes = [pc, pc, ctypes.c_int, ctypes.POINTER(T.Cfg), pc, pd]
    lib.dpd_train_dla.argtypes = [pc, ctypes.c_int, pc, ctypes.POINTER(T.Cfg), pc, pd]
    _, x, y = T.senal(n)

    def correr(G, **kw):
        cfg = T.Cfg()
        lib.dpd_train_cfg_default(ctypes.byref(cfg))
        cfg.G, cfg.mu_h_respaldo = G, 0.0
        for k, v in kw.items():
            setattr(cfg, k, v)
        h, w = np.zeros(12, complex), np.zeros(12, complex)
        lib.dpd_train_pa_model(x, y, n, ctypes.byref(cfg), h, np.zeros(cfg.ep_pa))
        lib.dpd_train_dla(x, n, h, ctypes.byref(cfg), w, np.zeros(cfg.ep_dpd))
        return h, w

    print(f"{n} muestras, G = 1 contra G = 8, diferencia máxima en LSB de Q0.15")
    for nombre, kw in (("LMS sin límite", dict(h_ganancia_max=0.0)),
                       ("LMS con límite 1", dict(h_ganancia_max=1.0)),
                       ("NLMS (el respaldo)", dict(use_nlms_h=1, mu_h=0.05))):
        h1, w1 = correr(1.0, **kw)
        h8, w8 = correr(8.0, **kw)
        print(f"  {nombre:20s} h {np.max(np.abs(h1 - h8)) / LSB:7.3g}   "
              f"w {np.max(np.abs(w1 - w8)) / LSB:7.3g}")


if __name__ == "__main__":
    main()
