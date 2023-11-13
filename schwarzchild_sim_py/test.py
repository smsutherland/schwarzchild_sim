import schwarz.util
import astropy.units as u

m = schwarz.presets.mercury_orbit

res1 = schwarz.simulate.euler_1(m, 5e-1, 1e9, 10000)
res2 = schwarz.simulate.euler_2(m, 5e-1, 1e9, 10000)
res3 = schwarz.simulate.euler_3(m, 5e-1, 1e9, 10000)
res4 = schwarz.simulate.euler_4(m, 5e-1, 1e9, 10000)

precession1 = schwarz.util.calculate_precession(res1).to(u.arcsec / u.century)
precession2 = schwarz.util.calculate_precession(res2).to(u.arcsec / u.century)
precession3 = schwarz.util.calculate_precession(res3).to(u.arcsec / u.century)
precession4 = schwarz.util.calculate_precession(res4).to(u.arcsec / u.century)
print(precession1)
print(precession2)
print(precession3)
print(precession4)
