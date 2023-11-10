use super::Solver;
use crate::{BodyParameters, C, G};

pub struct EulerSolve1 {
    r_s: f64,
}

impl Solver for EulerSolve1 {
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

impl Solver for EulerSolve2 {
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