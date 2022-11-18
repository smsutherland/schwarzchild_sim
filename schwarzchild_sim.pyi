from typing import Optional
import numpy as np

def circular_omega(mass: float, radius: float) -> float: ...
def schwarzchild_radius(mass: float) -> float: ...
def gen_initial_condition(
    m: float,
    r: float,
    v: float,
    theta: float,
    omega: Optional[float] = None,
    h: Optional[float] = None,
) -> BodyParameters: ...
def simulate_conditions_rel(
    initial_condition: BodyParameters,
    max_theta: float = 2*np.pi,
    max_r: float = 10*1.5e11,
    history_interval: int = 100000,
    time_step: float = 1e-5,
    version: int = 4,
) -> np.ndarray: ...

class BodyParameters:
    M: float
    r: float
    v: float
    theta: float
    omega: float

    def clone(self) -> BodyParameters: ...
    def get_h(self) -> float: ...
    def rs(self) -> float: ...

earth_orbit: BodyParameters
small_precession: BodyParameters
zoom_whirl: BodyParameters