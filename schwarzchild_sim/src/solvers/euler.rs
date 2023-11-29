use ndarray::Array2;

use crate::{BodyParameters, C, G};

pub trait Euler {
    fn init(&mut self, initial_condition: &mut BodyParameters, time_step: &mut f64);
    fn step(&mut self, condition: &mut BodyParameters, time_step: f64);
}

pub fn simulate_euler(
    mut initial_condition: BodyParameters,
    max_t: f64,
    history_interval: usize,
    mut time_step: f64,
    mut solver: impl Euler,
) -> Array2<f64> {
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
    Array2::from(history)
}

pub struct EulerSolve1 {
    r_s: f64,
}

impl Euler for EulerSolve1 {
    fn init(&mut self, initial_condition: &mut BodyParameters, _: &mut f64) {
        self.r_s = 2. * initial_condition.mass * G / (C * C);
    }

    fn step(&mut self, condition: &mut BodyParameters, delta_s: f64) {
        let dr = condition.radial_velocity * delta_s;
        let dtheta = condition.angular_velocity * delta_s;
        let domega = -2. * condition.angular_velocity * dr / condition.radius;
        let dv = -0.5
            * ((self.r_s / (condition.radius * condition.radius))
                * (C * C
                    + condition.radius
                        * condition.radius
                        * condition.angular_velocity
                        * condition.angular_velocity)
                + (1. - self.r_s / condition.radius)
                    * (-2.
                        * condition.radius
                        * condition.angular_velocity
                        * condition.angular_velocity))
            * delta_s;

        condition.radius += dr;
        condition.angle += dtheta;
        condition.angular_velocity += domega;
        condition.radial_velocity += dv;
    }
}

impl EulerSolve1 {
    pub fn new() -> Self {
        Self { r_s: 0. }
    }
}

impl Default for EulerSolve1 {
    fn default() -> Self {
        Self::new()
    }
}

pub struct EulerSolve2;

impl Euler for EulerSolve2 {
    fn init(&mut self, initial_condition: &mut BodyParameters, time_step: &mut f64) {
        geometrizish_quantities(initial_condition, time_step);
    }

    fn step(&mut self, condition: &mut BodyParameters, time_step: f64) {
        let r2 = condition.radius * condition.radius;
        let omega2 = condition.angular_velocity * condition.angular_velocity;

        let dr = condition.radial_velocity * time_step;
        let dtheta = condition.angular_velocity * time_step;
        let dometa = -2. * condition.angular_velocity * dr / condition.radius;
        let dv = (-0.5 * condition.mass * (1. / r2 + omega2)
            + (condition.radius - condition.mass) * omega2)
            * time_step;

        condition.radius += dr;
        condition.angle += dtheta;
        condition.angular_velocity += dometa;
        condition.radial_velocity += dv;
    }
}

fn geometrizish_quantities(initial_condition: &mut BodyParameters, time_step: &mut f64) {
    initial_condition.mass *= 2. * G / (C * C);
    initial_condition.radial_velocity /= C;
    initial_condition.angular_velocity /= C;
    *time_step *= C;
}

pub struct EulerSolve3;

impl Euler for EulerSolve3 {
    fn init(&mut self, initial_condition: &mut BodyParameters, time_step: &mut f64) {
        geometrizish_quantities(initial_condition, time_step);
    }

    fn step(&mut self, condition: &mut BodyParameters, time_step: f64) {
        let r2 = condition.radius * condition.radius;
        let omega2 = condition.angular_velocity * condition.angular_velocity;

        let dr = condition.radial_velocity * time_step;
        let dtheta = condition.angular_velocity * time_step;
        let domega = -2. * condition.angular_velocity * dr / condition.radius;
        let dv = (-0.5 * condition.mass * (1. / r2 + omega2)
            + (condition.radius - condition.mass) * omega2)
            * time_step;

        condition.radius += dr + 0.5 * dv * time_step;
        condition.angle += dtheta + 0.5 * domega * time_step;
        condition.angular_velocity += domega;
        condition.radial_velocity += dv;
    }
}

pub struct EulerSolve4;

impl Euler for EulerSolve4 {
    fn init(&mut self, initial_condition: &mut BodyParameters, time_step: &mut f64) {
        geometrizish_quantities(initial_condition, time_step);
    }

    fn step(&mut self, condition: &mut BodyParameters, time_step: f64) {
        let r2 = condition.radius * condition.radius;
        let omega2 = condition.angular_velocity * condition.angular_velocity;

        let dr = condition.radial_velocity * time_step;
        let dtheta = condition.angular_velocity * time_step;
        let dv = (-0.5 * condition.mass * (1. / r2 + omega2)
            + (condition.radius - condition.mass) * omega2)
            * time_step;

        let old_h = condition.radius * condition.radius * condition.angular_velocity;
        condition.radius += dr + 0.5 * dv * time_step;
        condition.angle += dtheta;
        condition.angular_velocity = old_h / (condition.radius * condition.radius);
        condition.radial_velocity += dv;
    }
}
