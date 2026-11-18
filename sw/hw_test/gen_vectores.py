"""Pasa los golden vectors a un header de C para el programa del ARM.

Uso: python gen_vectores.py      (escribe vectores.h al lado)

Empaqueta cada muestra como la espera el reproductor, {im, re} en 32 bits, y
los coeficientes como los espera W_COEFF. No se versiona: sale de los
archivos de fixed_point/golden_vectors/out con este script.
"""
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
GOLDEN = os.path.join(AQUI, "..", "..", "fixed_point", "golden_vectors", "out")


def leer(nombre):
    with open(os.path.join(GOLDEN, nombre)) as f:
        return [int(ln, 16) for ln in f if ln.strip()]


def empaquetar(re, im):
    return [((i & 0xFFFF) << 16) | (r & 0xFFFF) for r, i in zip(re, im)]


def arreglo(fh, nombre, valores):
    fh.write(f"static const uint32_t {nombre}[{len(valores)}] = {{\n")
    for k in range(0, len(valores), 8):
        fh.write("    " + ", ".join(f"0x{v:08X}U" for v in valores[k:k + 8]) + ",\n")
    fh.write("};\n\n")


def main():
    with open(os.path.join(AQUI, "vectores.h"), "w") as fh:
        fh.write("/* Generado por gen_vectores.py desde los golden vectors. No editar. */\n")
        fh.write("#ifndef VECTORES_H\n#define VECTORES_H\n\n#include <stdint.h>\n\n")
        arreglo(fh, "COEFS", empaquetar(leer("coeffs_w_re.hex"), leer("coeffs_w_im.hex")))
        for tipo in ["functional", "directed"]:
            arreglo(fh, f"{tipo.upper()}_IN", empaquetar(leer(f"{tipo}_input_i.hex"), leer(f"{tipo}_input_q.hex")))
            arreglo(fh, f"{tipo.upper()}_OUT", empaquetar(leer(f"{tipo}_output_i.hex"), leer(f"{tipo}_output_q.hex")))
        fh.write("#endif\n")
    print("vectores.h escrito")


if __name__ == "__main__":
    main()
