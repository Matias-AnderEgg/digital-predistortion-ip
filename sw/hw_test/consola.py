"""Maneja la consola serie de la Red Pitaya por el micro USB.

La placa no tiene red, así que todo pasa por la consola: los comandos y
también los archivos, que viajan comprimidos y en base64 como texto.

Usa el primer /dev/cu.usbserial-*, o el puerto que diga RP_PUERTO.

Uso:
  python consola.py ver                         lo que haya en la consola
  python consola.py cmd "uname -a"              corre un comando, sale con su código
  python consola.py enviar archivo /root/destino
  python consola.py traer /root/origen archivo

No inicia sesion: eso se hace una vez a mano, con screen, y la sesion queda
abierta en la placa.

Solo usa la biblioteca estándar, para no instalar pyserial.
"""
import base64
import glob
import gzip
import hashlib
import os
import re
import select
import sys
import termios
import time

BAUDIOS = termios.B115200


def puerto():
    p = os.environ.get("RP_PUERTO") or next(iter(sorted(glob.glob("/dev/cu.usbserial-*"))), None)
    if not p:
        sys.exit("no hay ninguna placa conectada por USB")
    return p


class Consola:
    def __init__(self):
        self.fd = self.fd_r = os.open(puerto(), os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        a = termios.tcgetattr(self.fd)
        a[0] = 0                                            # iflag
        a[1] = 0                                            # oflag
        a[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # cflag, 8N1
        a[3] = 0                                            # lflag: crudo
        a[4] = a[5] = BAUDIOS
        a[6][termios.VMIN] = 0
        a[6][termios.VTIME] = 0
        termios.tcsetattr(self.fd, termios.TCSANOW, a)

    def escribir(self, datos):
        if isinstance(datos, str):
            datos = datos.encode()
        while datos:
            select.select([], [self.fd], [])
            n = os.write(self.fd, datos)
            datos = datos[n:]
        termios.tcdrain(self.fd)

    def leer(self, hasta=None, espera=2.0, total=600.0):
        """Lee hasta que aparece `hasta` o pasan `espera` segundos sin datos."""
        buf = b""
        t0 = ultimo = time.time()
        while True:
            r, _, _ = select.select([self.fd_r], [], [], 0.1)
            if r:
                try:
                    d = os.read(self.fd_r, 4096)
                except BlockingIOError:
                    d = b""
                if d:
                    buf += d
                    ultimo = time.time()
                    if hasta is not None and re.search(hasta, buf):
                        return buf
            ahora = time.time()
            if hasta is None and ahora - ultimo > espera:
                return buf
            if ahora - t0 > total:
                raise TimeoutError(buf.decode(errors="replace")[-2000:])

    def cmd(self, linea, total=600.0):
        """Corre `linea` y devuelve (codigo, salida)."""
        self.leer(espera=0.2)                       # descarta lo pendiente
        self.escribir(linea + "; echo __FIN_$?__\n")
        buf = self.leer(hasta=rb"__FIN_(\d+)__", total=total)
        texto = buf.decode(errors="replace").replace("\r", "")
        # Saca los codigos de escape de la terminal (colores, modo de pegado).
        texto = re.sub(r"\x1b(\[[0-9;?]*[A-Za-z]|\][^\x07]*\x07)", "", texto)
        codigo = int(re.search(r"__FIN_(\d+)__", texto).group(1))
        cuerpo = texto[:texto.rfind("__FIN_")]
        # La primera línea es el eco del comando.
        cuerpo = cuerpo.split("\n", 1)[1] if "\n" in cuerpo else ""
        return codigo, cuerpo

    def enviar(self, local, remoto):
        datos = open(local, "rb").read()
        b64 = base64.encodebytes(gzip.compress(datos, 9)).decode()
        md5 = hashlib.md5(datos).hexdigest()
        self.leer(espera=0.2)
        # Sin eco, para no mandar todo dos veces por el cable.
        self.escribir(f"stty -echo; base64 -d | gzip -dc > {remoto}; stty echo\n")
        time.sleep(0.5)
        t0 = time.time()
        lineas = b64.splitlines(keepends=True)
        for i in range(0, len(lineas), 32):
            self.escribir("".join(lineas[i:i + 32]))
        self.escribir("\x04")                       # fin de la entrada
        codigo, salida = self.cmd(f"md5sum {remoto}")
        ok = codigo == 0 and md5 in salida.split()
        print(f"{local} -> {remoto}: {len(datos)} bytes, {len(b64)} por el cable, "
              f"{time.time() - t0:.0f} s, {'md5 ok' if ok else 'MD5 DISTINTO'}")
        return ok

    def traer(self, remoto, local):
        codigo, salida = self.cmd(f"gzip -9c {remoto} | base64", total=1200)
        if codigo != 0:
            sys.exit(salida)
        open(local, "wb").write(gzip.decompress(base64.b64decode(salida)))
        print(f"{remoto} -> {local}")


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c = Consola()
    orden = sys.argv[1]
    if orden == "ver":
        c.escribir("\n")
        sys.stdout.write(c.leer(espera=1.5).decode(errors="replace"))
    elif orden == "cmd":
        codigo, salida = c.cmd(sys.argv[2])
        sys.stdout.write(salida)
        sys.exit(codigo)
    elif orden == "enviar":
        sys.exit(0 if c.enviar(sys.argv[2], sys.argv[3]) else 1)
    elif orden == "traer":
        c.traer(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
