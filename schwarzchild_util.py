from typing import Optional
import numpy as np
import pylab as pl
from astropy import units as u, constants as c
from schwarzchild_sim import BodyParameters

u.century = u.def_unit("century", 100*u.yr, "A century. 100 years.")

def plot_orbit(orbit: np.ndarray, mass: Optional[float] = None):
    orbit = orbit.T

    r = orbit[0]
    theta = orbit[1]

    pl.polar(theta, r)

def calculate_precession(orbit: np.ndarray, per_orbit: bool = False):
    """
    calculate the precession of the periapsis of the orbit,
    as returned by schwarzchild_sim.simulate_conditions_rel
    """
    orbit = orbit.T
    r = orbit[0]
    theta = orbit[1]

    r_maxima = np.r_[True, r[1:] > r[:-1]] & np.r_[r[:-1] > r[1:], True]
    apoapsies = theta[r_maxima][1:-1]
    apoapsies -= 2*np.pi*np.arange(0, apoapsies.shape[0])
    average_change = apoapsies[-1] - apoapsies[1]
    if per_orbit:
        return average_change * u.rad
    else:
        time = orbit[2]
        times = time[r_maxima]
        average_change_time = np.mean(times[-1] - times[1])
        return average_change/average_change_time * u.rad / u.s

def calculate_expected_precession(params: BodyParameters):
    """
    calculate the expected precession of the periapsis of the orbit.
    Takes in a BodyParameters object
    """
    return 6*np.pi*c.G**2*(params.M*u.kg)**2/params.get_h()**2/c.c**2

def calculate_orbital_period(orbit: np.ndarray):
    """
    calculate the period of the orbit,
    as returned by schwarzchild_sim.simulate_conditions_rel
    """
    orbit = orbit.T
    r = orbit[0]
    r_maxima = np.r_[True, r[1:] > r[:-1]] & np.r_[r[:-1] > r[1:], True]
    time = orbit[2][r_maxima][1:-1]
    average_time = (time[-1] - time[0])/time.shape[0]
    return average_time
