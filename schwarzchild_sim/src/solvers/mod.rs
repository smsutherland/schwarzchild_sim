use crate::BodyParameters;

mod euler;
pub use euler::{EulerSolve1, EulerSolve2};

pub trait Solver {
    fn init(&mut self, initial_condition: &mut BodyParameters, time_step: &mut f64);
    fn step(&mut self, condition: &mut BodyParameters, time_step: f64);
}
