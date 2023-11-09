use ndarray::Array2;

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

pub fn simulate_conditions<S: Solver>(
    mut initial_condition: BodyParameters,
    end: impl Fn(&BodyParameters, f64) -> bool,
    history_interval: usize,
    mut time_step: f64,
    mut solver: S,
) -> Result<Array2<f64>, &'static str> {
    let time_step_in_s = time_step;

    let mut history = Vec::new();
    let mut steps = 0;
    let mut positive_v = true;

    solver.init(&mut initial_condition, &mut time_step);

    while !end(&initial_condition, steps as f64 * time_step_in_s) {
        solver.step(&mut initial_condition, time_step);
        if steps % history_interval == 0 || {
            let changed = (initial_condition.radial_velocity >= 0.) ^ positive_v;
            if changed {
                positive_v = !positive_v;
            }
            changed
        } {
            history.push([
                initial_condition.radius,
                initial_condition.angle,
                steps as f64 * time_step_in_s,
            ]);
        }
        steps += 1;
    }

    Ok(Array2::from(history))
}
