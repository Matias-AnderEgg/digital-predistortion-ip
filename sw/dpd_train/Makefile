# Biblioteca compartida para verificar el entrenamiento en la Mac contra el
# modelo de Python. Para el ARM se compila dpd_train.c dentro del proyecto
# del PS, no hace falta nada de aca.
#
# -ffp-contract=off: sin esto el compilador fusiona multiplicacion y suma
# en una sola instruccion (FMA), que redondea distinto que Python. El
# Cortex-A9 no tiene FMA en doble precision, asi que tambien es lo que
# va a pasar en la placa.

CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -pedantic -ffp-contract=off
BUILD   := build

ifeq ($(shell uname),Darwin)
LIB := $(BUILD)/libdpd_train.dylib
else
LIB := $(BUILD)/libdpd_train.so
endif

PY ?= python3

all: $(LIB)

$(LIB): dpd_train.c dpd_train.h | $(BUILD)
	$(CC) $(CFLAGS) -fPIC -shared -o $@ dpd_train.c -lm

$(BUILD):
	mkdir -p $@

test: $(LIB)
	$(PY) test/test_dpd_train.py

rapido: $(LIB)
	$(PY) test/test_dpd_train.py --rapido

clean:
	rm -rf $(BUILD)

.PHONY: all test rapido clean
