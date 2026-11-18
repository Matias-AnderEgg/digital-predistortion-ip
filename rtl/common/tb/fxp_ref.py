"""Espejo en Python de los bloques comunes de punto fijo del RTL.

`round_sat` replica `rtl/common/hdl/fxp_round_sat.sv` operacion por
operacion, con enteros. `round_sat_fxp` hace la misma cuenta delegando en
fxpmath, la libreria con la que se decidieron los formatos del DPD.

Que los tres coincidan (RTL, este espejo, fxpmath) es lo que garantiza que
el hardware implementa la convención del modelo y no una parecida.
"""


def to_signed(val: int, width: int) -> int:
    """Bits crudos -> entero con signo en complemento a dos."""
    mask = (1 << width) - 1
    val &= mask
    if val & (1 << (width - 1)):
        val -= 1 << width
    return val


def limits(width: int) -> tuple[int, int]:
    """Minimo y maximo de un entero con signo de `width` bits."""
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def round_sat(din: int, in_frac: int, out_w: int, out_frac: int) -> int:
    """Redondeo al mas cercano (empate al par) y saturación.

    `din` es un entero con signo que representa din / 2^in_frac. Devuelve el
    código entero en el formato de salida.
    """
    shift = in_frac - out_frac
    assert shift >= 0, "el bloque saca bits fraccionarios, no los agrega"

    # En Python el corrimiento a la derecha sobre negativos ya es piso
    # (hacia menos infinito), igual que el >>> de SystemVerilog.
    piso = din >> shift

    if shift >= 1:
        descartado = din & ((1 << shift) - 1)
        mitad = 1 << (shift - 1)
        if descartado > mitad:
            piso += 1
        elif descartado == mitad and (piso & 1):
            # Empate exacto: sube solo si el piso es impar, para terminar par.
            piso += 1

    lo, hi = limits(out_w)
    return max(lo, min(hi, piso))


def round_sat_fxp(din: int, in_frac: int, out_w: int, out_frac: int) -> int:
    """Misma cuenta, delegada en fxpmath."""
    from fxpmath import Fxp

    # La division es exacta en doble precisión mientras din entre en 53
    # bits, que cubre de sobra los anchos del datapath.
    exacto = din / float(1 << in_frac)
    q = Fxp(
        exacto,
        signed=True,
        n_word=out_w,
        n_frac=out_frac,
        rounding="around",
        overflow="saturate",
    )
    return int(q.raw())


def cmul_round_sat(a_re: int, a_im: int, b_re: int, b_im: int,
                   a_frac: int, b_frac: int, o_w: int, o_frac: int):
    """Espejo de `rtl/common/hdl/cmul_round_sat.sv`.

    Productos parciales y sumas al ancho completo, una sola reducción de
    formato al final por componente. Redondear cada parcial por separado
    daría distinto (y peor).
    """
    p_frac = a_frac + b_frac
    s_re = a_re * b_re - a_im * b_im
    s_im = a_re * b_im + a_im * b_re
    return (round_sat(s_re, p_frac, o_w, o_frac),
            round_sat(s_im, p_frac, o_w, o_frac))


def cmul_round_sat_fxp(a_re: int, a_im: int, b_re: int, b_im: int,
                       a_frac: int, b_frac: int, o_w: int, o_frac: int):
    """Lo mismo, delegado en fxpmath sobre el producto complejo completo."""
    p_frac = a_frac + b_frac
    s_re = a_re * b_re - a_im * b_im
    s_im = a_re * b_im + a_im * b_re
    return (round_sat_fxp(s_re, p_frac, o_w, o_frac),
            round_sat_fxp(s_im, p_frac, o_w, o_frac))


def mp_magnitude(x_re: int, x_im: int, x_frac: int,
                 a_w: int, a_frac: int, npow: int = 3):
    """Espejo de `rtl/common/hdl/mp_magnitude.sv`.

    Devuelve [a2, a4, a6, ...]: |x|^2 y sus potencias enteras, cada una
    reducida al formato del acumulador. No hay raíz cuadrada.
    """
    suma = x_re * x_re + x_im * x_im          # exacto
    a = [round_sat(suma, 2 * x_frac, a_w, a_frac)]
    for k in range(1, npow):
        a.append(round_sat(a[k - 1] * a[0], 2 * a_frac, a_w, a_frac))
    return a


def acc_step(acc: int, term: int, shift: int, a_w: int, a_frac: int) -> int:
    """Espejo de `rtl/common/hdl/mp_acc_step.sv`.

    El corrimiento se hace sin perdida y la suma es exacta; lo único que
    puede pasar es que el resultado no entre en el formato del acumulador.
    Como la entrada y la salida tienen los mismos bits fraccionarios,
    `round_sat` queda como pura saturación.
    """
    return round_sat((term << shift) + acc, a_frac, a_w, a_frac)
