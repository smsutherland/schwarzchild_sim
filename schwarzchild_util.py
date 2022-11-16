from typing import Optional
import numpy as np
import pylab as pl
from astropy import units as u, constants as c
import schwarzchild_sim

u.century = u.def_unit("century", 100*u.yr, "A century. 100 years.")

def plot_orbit(orbit: np.ndarray, mass: Optional[float] = None):
    orbit = orbit.T

    r = orbit[0]
    theta = orbit[1]

    pl.polar(theta, r, label="orbit")

    if mass is not None:
        r = (2.*c.G*mass/c.c**2 * u.kg).to_value(u.m) * np.ones(100)
        theta = np.linspace(0, 2.*np.pi, 100, endpoint=True)
        pl.polar(theta, r, label="Schwarzchild radius")
        pl.legend()

def calculate_precession(orbit: np.ndarray, per_orbit: bool = False):
    """
    calculate the precession of the periapsis of the orbit,
    as returned by schwarzchild_sim.simulate_conditions_rel
    """
    orbit = orbit.T
    r = orbit[0]
    theta = orbit[1]
    time = orbit[2]

    r_maxima = np.r_[True, r[1:] > r[:-1]] & np.r_[r[:-1] > r[1:], True]
    apoapsies = theta[r_maxima][1:-1]
    apoapsies -= 2*np.pi*np.arange(0, apoapsies.shape[0])
    average_change = apoapsies[-1] - apoapsies[0]
    if per_orbit:
        return average_change/(apoapsies.shape[0] - 1) * u.rad
    else:
        times = time[r_maxima][1:-1]
        average_change_time = times[-1] - times[0]
        return average_change/average_change_time * u.rad / u.s

def calculate_expected_precession(params):
    """
    calculate the expected precession of the periapsis of the orbit.
    Takes in a BodyParameters object
    """
    return (6*np.pi*u.rad)*c.G**2*(params.M*u.kg)**2/(params.get_h()*u.m**2/u.s)**2/c.c**2

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

small_precession_preset = (
    schwarzchild_sim.small_precession.clone(),
    {
        "max_theta": 10*2*np.pi,
        "time_step": 1e-6,
        "history_interval": 10000,
    }
)

two_lobe_zoom_whirl_preset = (
    schwarzchild_sim.zoom_whirl.clone(),
    {
        "max_theta": 10*2*np.pi,
        "time_step": 1e-8,
        "history_interval": 1000,
    }
)

three_lobe_zoom_whirl_preset_params = schwarzchild_sim.zoom_whirl.clone()
three_lobe_zoom_whirl_preset_params.omega *= 1.062
three_lobe_zoom_whirl_preset = (
    three_lobe_zoom_whirl_preset_params,
    {
        "max_theta": 10*2*np.pi,
        "time_step": 1e-8,
        "history_interval": 1000,
    }
)

mercury_orbit_preset = (
    schwarzchild_sim.mercury_orbit.clone(),
    {
        "max_theta": 10*2*np.pi,
        "time_step": 1e-1,
    }
)
