from typing import Optional, Union
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u, constants as c

from schwarz._inner import BodyParameters, schwarzchild_radius
from schwarz import _inner

u.century = u.def_unit("century", 100 * u.yr, "A century. 100 years.")


def plot_orbit(orbit: np.ndarray, mass: Optional[float] = None) -> plt.Figure:
    """
    Plots an orbit as returned from schwarzchild_sim.simulate_conditions_rel.
    If mass is provided, the schwarzchild radius is also plotted.
    """
    orbit = orbit.T

    r = orbit[0]
    theta = orbit[1]

    fig = plt.figure()
    ax = plt.subplot(111, projection="polar")

    ax.plot(theta, r, label="orbit")

    if mass is not None:
        r = (2.0 * c.G * mass / c.c**2 * u.kg).to_value(u.m) * np.ones(100)
        theta = np.linspace(0, 2.0 * np.pi, 100, endpoint=True)
        ax.polar(theta, r, label="Schwarzchild radius")
        ax.legend()
    return fig


def plot_effective_potential(
    orbit, kind: str = "r", xvals: Optional[Union[np.ndarray, u.Quantity]] = None
):
    """
    Plots the effective potential of the orbit.
    Kind: "r", "n", or "rn" will plot relativistic, newtonian, or both effective potentials
    xvals: if provided, will determine the domain over which the effective potential is plotted,
        otherwise it will be plotted on the domain 1-10 schwarzchild radii.
    """
    h: u.Quantity = orbit.get_h() * u.m**2 / u.s
    rs = u.def_unit("schwarzchild radius", orbit.rs() * u.m, format={"latex": "r_s"})
    M: u.Quantity = orbit.M * u.kg

    def v_newtonian(r: u.Quantity) -> u.Quantity:
        return (-c.G * M / r + h**2 / (2 * r**2)).to(u.J / u.kg)

    def v_relativistic(r: u.Quantity) -> u.Quantity:
        return (-c.G * M / r + h**2 / (2 * r**2) - c.G * M * h**2 / (c.c**2 * r**3)).to(
            u.J / u.kg
        )

    if xvals is None:
        xvals = np.arange(1.0, 10.0, 0.1) * rs
    elif isinstance(xvals, np.ndarray):
        xvals *= rs

    if "r" in kind:
        plt.plot(xvals, v_relativistic(xvals), label="relativistic")
    if "n" in kind:
        plt.plot(xvals, v_newtonian(xvals), label="newtonian")
    plt.legend()


def calculate_precession(orbit: np.ndarray, per_orbit: bool = False) -> float:
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
    apoapsies -= 2 * np.pi * np.arange(0, apoapsies.shape[0])
    average_change = apoapsies[-1] - apoapsies[0]
    if per_orbit:
        return average_change / (apoapsies.shape[0] - 1) * u.rad
    else:
        times = time[r_maxima][1:-1]
        average_change_time = times[-1] - times[0]
        return average_change / average_change_time * u.rad / u.s


def calculate_expected_precession(params: BodyParameters) -> float:
    """
    calculate the expected precession of the periapsis of the orbit.
    Takes in a BodyParameters object
    """
    return (
        (6 * np.pi * u.rad)
        * c.G**2
        * (params.mass * u.kg) ** 2
        / ((params.radius * params.radius * params.angular_velocity) * u.m**2 / u.s)
        ** 2
        / c.c**2
    )


def calculate_orbital_period(orbit: np.ndarray) -> float:
    """
    calculate the period of the orbit,
    as returned by schwarzchild_sim.simulate_conditions_rel
    """
    orbit = orbit.T
    r = orbit[0]
    r_maxima = np.r_[True, r[1:] > r[:-1]] & np.r_[r[:-1] > r[1:], True]
    time = orbit[2][r_maxima][1:-1]
    average_time = (time[-1] - time[0]) / time.shape[0]
    return average_time


__all__ = [
    "plot_orbit",
    "plot_effective_potential",
    "calculate_precession",
    "calculate_expected_precession",
    "calculate_orbital_period",
]
