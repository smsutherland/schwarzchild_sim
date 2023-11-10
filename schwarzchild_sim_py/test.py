import matplotlib.pyplot as plt
import schwarz.util

m = schwarz.mercury_orbit

res = schwarz.simulate.simulate_euler_1(m, 1e-1, 10000, max_theta=2*3.14)

schwarz.util.plot_orbit(res)
plt.show()
