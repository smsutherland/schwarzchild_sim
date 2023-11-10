from ._inner import BodyParameters, M_SUN, schwarzchild_radius

mercury_orbit = BodyParameters(M_SUN, 6.9818e10, 0., 0., 3.886e4 / 6.9818e10)
small_precession = BodyParameters(M_SUN, schwarzchild_radius(M_SUN) * 200., 0., 0., 35.)

__all__ = ["mercury_orbit", "small_precession"]
