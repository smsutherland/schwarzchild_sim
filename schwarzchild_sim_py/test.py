import schwarz.util
import astropy.units as u
import matplotlib.pyplot as plt

m = schwarz.presets.mercury_orbit

r = schwarz.simulate.modified_midpoint(m, 5e-1, 1e9, 10000)
