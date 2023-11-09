import matplotlib.pyplot as plt
import schwarz
import numpy as np
import astropy.units as u
import astropy.constants as c
m = schwarz.mercury_orbit
result = schwarz.simulate.simulate_euler_1(m, 1e-1, max_theta=2*3.14, history_interval=10000)
print(result)

def plot_orbit(orbit, mass = None):
    """
    Plots an orbit as returned from schwarzchild_sim.simulate_conditions_rel.
    If mass is provided, the schwarzchild radius is also plotted.
    """
    orbit = orbit.T

    r = orbit[0]
    theta = orbit[1]

    plt.polar(theta, r, label="orbit")

    if mass is not None:
        r = (2.*c.G*mass/c.c**2 * u.kg).to_value(u.m) * np.ones(100)
        theta = np.linspace(0, 2.*np.pi, 100, endpoint=True)
        plt.polar(theta, r, label="Schwarzchild radius")
        plt.legend()

plot_orbit(result)
plt.show()
