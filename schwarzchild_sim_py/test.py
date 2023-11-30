import schwarz.util
import astropy.units as u
import matplotlib.pyplot as plt

m = schwarz.presets.mercury_orbit

r = schwarz.simulate.conservation(m, 5e-1, 1e9, 10000)
