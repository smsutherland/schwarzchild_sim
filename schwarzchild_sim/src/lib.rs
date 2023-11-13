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
    max_t: f64,
    history_interval: usize,
    mut time_step: f64,
    mut solver: S,
) -> Result<Array2<f64>, &'static str> {
    let time_step_in_s = time_step;
    let iters = (max_t / time_step).ceil() as usize;

    let mut history = Vec::with_capacity(iters / history_interval + 100);
    let mut positive_v = true;

    solver.init(&mut initial_condition, &mut time_step);

    for i in 0..iters {
        solver.step(&mut initial_condition, time_step);
        if i % history_interval == 0 || {
            let changed = (initial_condition.radial_velocity >= 0.) ^ positive_v;
            if changed {
                positive_v = !positive_v;
            }
            changed
        } {
            history.push([
                initial_condition.radius,
                initial_condition.angle,
                i as f64 * time_step_in_s,
            ]);
            print!("\r{:.2}%", 100. * i as f32 / iters as f32);
        }
    }

    println!("\r100.00%");
    Ok(Array2::from(history))
}
