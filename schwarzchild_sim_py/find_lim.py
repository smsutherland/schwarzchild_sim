import schwarz.util
import numpy as np
import time
import astropy.units as u
import sys
from math import ceil

m = schwarz.presets.mercury_orbit

def calc_prec(f, dt, t, n = 100000):
    r = f(m, dt, t, n)
    return schwarz.util.calculate_precession(r), r[-1, :-1]

eps = 1e-2

f = eval(f"schwarz.simulate.{sys.argv[1]}")

n = 1
t = 1e9
dt = t/n
prevr = f(m, dt, t, 100000)[-1, :-1]
while True:
    n = ceil(n * 1.1)
    dt = t/n
    start = time.perf_counter()
    thisr = f(m, dt, t, 100000)[-1, :-1]
    end = time.perf_counter()
    elapsed = end - start
    print(f"{dt:.3e}: {thisr} - {elapsed:.3f}s")
    if np.all(np.abs((thisr - prevr)/prevr) < eps):
        break

print(f"it took {elapsed}s to reach a precision of {eps}")
