mod solvers;
pub use solvers::*;

/// Gravitational constant in SI units [m^3 kg^-1 s^-2]
pub const G: f64 = 6.674e-11;

/// Speed of light in SI units [m s^-1]
pub const C: f64 = 3e8;

/// Mass of the Sun in SI units [kg]
pub const M_SUN: f64 = 2e30;

/// Astronomical unit in SI units [m]
pub const AU: f64 = 1.5e11;

#[derive(Debug, Clone)]
pub struct BodyParameters {
    pub mass: f64,
    pub radius: f64,
    pub radial_velocity: f64,
    pub angle: f64,
    pub angular_velocity: f64,
}

pub fn schwarzchild_radius(mass: f64) -> f64 {
    2. * mass * G / (C * C)
}
