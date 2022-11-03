use numpy::ndarray::Array2;
use numpy::IntoPyArray;
use pyo3::{exceptions::PyValueError, prelude::*};

const G: f64 = 6.674e-11;
const C: f64 = 3e8;
const M_SUN: f64 = 2e30;
const AU: f64 = 1.5e11;

#[allow(non_snake_case)]
#[derive(Debug, Clone)]
#[pyclass]
pub struct BodyParameters {
    #[pyo3(get, set)]
    M: f64,
    #[pyo3(get, set)]
    r: f64,
    #[pyo3(get, set)]
    v: f64,
    #[pyo3(get, set)]
    theta: f64,
    #[pyo3(get, set)]
    omega: f64,
}

impl BodyParameters {
    #[allow(dead_code)]
    fn update_newtonian(&mut self, delta_t: f64) {
        let dr = self.v * delta_t;
        let dtheta = self.omega * delta_t;
        let domega = -2. * self.omega * dr / self.r;
        let dv1 = -0.5 * self.r * domega;
        let dv2 = self.r * self.omega * self.omega * delta_t;
        let dv3 = -G * self.M * delta_t / (self.r * self.r);
        let dv4 = -self.omega * dr;
        let dv = dv1 + dv2 + dv3 + dv4;

        self.r += dr;
        self.theta += dtheta;
        self.omega += domega;
        self.v += dv;
    }

    fn update_relativity_v1(&mut self, delta_s: f64) {
        let rs = 2. * self.M * G / (C * C);

        let dr = self.v * delta_s;
        let dtheta = self.omega * delta_s;
        let domega = -2. * self.omega * dr / self.r;
        let dv = -0.5
            * ((rs / (self.r * self.r)) * (C * C + self.r * self.r * self.omega * self.omega)
                + (1. - rs / self.r) * (-2. * self.r * self.omega * self.omega))
            * delta_s;

        self.r += dr;
        self.theta += dtheta;
        self.omega += domega;
        self.v += dv;
    }

    /// Assumes self has geometrizished quantities and delta_s scaled by `C`.
    fn update_relativity_v2(&mut self, delta_s: f64) {
        let r2 = self.r * self.r;
        let omega2 = self.omega * self.omega;

        let dr = self.v * delta_s;
        let dtheta = self.omega * delta_s;
        let domega = -2. * self.omega * dr / self.r;
        let dv = (-0.5 * self.M * (1. / r2 + omega2) + (self.r - self.M) * omega2) * delta_s;

        self.r += dr;
        self.theta += dtheta;
        self.omega += domega;
        self.v += dv;
    }

    /// Assumes self has geometrizished quantities and delta_s scaled by `C`.
    fn update_relativity_v3(&mut self, delta_s: f64) {
        let r2 = self.r * self.r;
        let omega2 = self.omega * self.omega;

        let dr = self.v * delta_s;
        let dtheta = self.omega * delta_s;
        let domega = -2. * self.omega * dr / self.r;
        let dv = (-0.5 * self.M * (1. / r2 + omega2) + (self.r - self.M) * omega2) * delta_s;

        self.r += dr + 0.5 * dv * delta_s;
        self.theta += dtheta + 0.5 * domega * delta_s;
        self.omega += domega;
        self.v += dv;
    }

    fn geometrizish_quantities(&mut self) {
        self.M *= 2. * G / (C * C);
        self.v /= C;
        self.omega /= C;
    }
}

#[pymethods]
impl BodyParameters {
    fn __str__(&self) -> String {
        format!(
            "M: {:.2e}, r: {:.2e}, v: {:.2e}, theta: {:.2e}, omega: {:.2e},",
            self.M, self.r, self.v, self.theta, self.omega
        )
    }

    fn clone(&self) -> Self {
        <Self as Clone>::clone(self)
    }
}

fn circular_omega(mass: f64, radius: f64) -> f64 {
    f64::sqrt(G * mass / radius) / radius
}

fn schwarzchild_radius(mass: f64) -> f64 {
    2. * mass * G / (C * C)
}

fn gen_initial_condition(
    m: f64,
    r: f64,
    v: f64,
    theta: f64,
    omega: Option<f64>,
    h: Option<f64>,
) -> Result<BodyParameters, &'static str> {
    match (omega, h) {
        (None, None) => Err("Either omega or h must be specified."),
        (Some(omega), None) => Ok(BodyParameters {
            M: m,
            r,
            v,
            theta,
            omega,
        }),
        (None, Some(h)) => Ok(BodyParameters {
            M: m,
            r,
            v,
            theta,
            omega: h / (r * r),
        }),
        (Some(_), Some(_)) => Err("omega and h cannot both be specified."),
    }
}

fn simulate_conditions_rel(
    mut initial_condition: BodyParameters,
    max_theta: f64,
    max_r: f64,
    history_interval: usize,
    mut time_step: f64,
    version: usize,
) -> Result<Array2<f64>, &'static str> {
    let time_step_in_s = time_step;
    if let 1..=3 = version {
    } else {
        return Err("version must be 1 - 3");
    }
    let update_fns = [
        BodyParameters::update_relativity_v1,
        BodyParameters::update_relativity_v2,
        BodyParameters::update_relativity_v3,
    ];
    let mut history = Vec::new();
    let mut steps = 0;
    let mut positive_v = true;
    if version != 1 {
        initial_condition.geometrizish_quantities();
        time_step *= C;
    }
    while initial_condition.theta <= max_theta
        && initial_condition.r > 0.
        && initial_condition.r <= max_r
    {
        update_fns[version - 1](&mut initial_condition, time_step);
        if steps % history_interval == 0 || {
            let changed = (initial_condition.v >= 0.) ^ positive_v;
            if changed {
                positive_v = !positive_v;
            }
            changed
        } {
            history.push([
                initial_condition.r,
                initial_condition.theta,
                steps as f64 * time_step_in_s,
            ]);
            print!("\r{:.2}%", 100. * initial_condition.theta / max_theta);
        }
        steps += 1;
    }
    println!("\r100.00%");
    Ok(Array2::from(history))
}

/// Simulates orbits around a Schwarzchild black hole.
#[pymodule]
fn schwarzchild_sim(_py: Python, m: &PyModule) -> PyResult<()> {
    /// circular_omega(mass, radius, /)
    /// --
    ///
    /// Find the angular velocity for a circular orbit (under newtonian gravity)
    /// for a given radius and central mass.
    #[pyfn(m)]
    fn circular_omega(mass: f64, radius: f64) -> f64 {
        crate::circular_omega(mass, radius)
    }

    /// schwarzchild_radius(mass, /)
    /// --
    ///
    /// Find the schwarzchild radius of an object of mass `mass`.
    #[pyfn(m)]
    fn schwarzchild_radius(mass: f64) -> f64 {
        crate::schwarzchild_radius(mass)
    }

    /// gen_initial_condition(m, r, v, theta, omega, h, /)
    /// --
    ///
    /// Generate a `BodyParameters` object matching the conditions given.
    /// Exactly one of `omega` or `h` must be specified.
    #[pyfn(m, theta = "0.")]
    fn gen_initial_condition(
        m: f64,
        r: f64,
        v: f64,
        theta: f64,
        omega: Option<f64>,
        h: Option<f64>,
    ) -> PyResult<BodyParameters> {
        crate::gen_initial_condition(m, r, v, theta, omega, h).map_err(|s| PyValueError::new_err(s))
    }

    /// simulate_conditions_rel(initial_condition, max_theta, max_r, history_interval, time_step, /)
    /// --
    ///
    /// Iterates the state of the test particle specified by `initial_condition`.
    /// Stops iteration if theta > `max_theta`, r > `max_r`, or r < 0.
    /// Saves a point every `history_interval` iterations, which then gets returned as an ndarray.
    #[pyfn(
        m,
        max_theta = "2.*std::f64::consts::PI",
        max_r = "10.*AU",
        history_interval = "100000",
        time_step = "1e-5",
        version = "1"
    )]
    fn simulate_conditions_rel(
        py: Python,
        initial_condition: BodyParameters,
        max_theta: f64,
        max_r: f64,
        history_interval: usize,
        time_step: f64,
        version: usize,
    ) -> PyResult<&numpy::PyArray2<f64>> {
        crate::simulate_conditions_rel(
            initial_condition,
            max_theta,
            max_r,
            history_interval,
            time_step,
            version,
        )
        .map_err(|s| PyValueError::new_err(s))
        .map(|history| history.into_pyarray(py))
    }

    m.add("M_sun", M_SUN)?;
    m.add("AU", AU)?;

    m.add(
        "small_precession",
        BodyParameters {
            M: M_SUN,
            r: schwarzchild_radius(M_SUN) * 200.,
            v: 0.,
            theta: 0.,
            omega: 100.,
        },
    )?;
    m.add(
        "earth_orbit",
        BodyParameters {
            M: M_SUN,
            r: AU,
            v: 0.,
            theta: 0.,
            omega: circular_omega(M_SUN, AU),
        },
    )?;
    m.add(
        "zoom_whirl",
        BodyParameters {
            M: M_SUN,
            r: schwarzchild_radius(M_SUN) * 20.,
            v: 0.,
            theta: 0.,
            omega: 8000. * 0.064,
        },
    )?;
    m.add(
        "mercury_orbit",
        BodyParameters {
            M: M_SUN,
            r: 6.9818e10,
            v: 0.,
            theta: 0.,
            omega: 3.886e4 / 6.9818e10,
        },
    )?;

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn run_schwarzchild_sim() {
        simulate_conditions_rel(
            BodyParameters {
                M: M_SUN,
                r: 6.9818e10,
                v: 0.,
                theta: 0.,
                omega: 3.886e4 / 6.9818e10,
            },
            100.*2.*std::f64::consts::PI,
            10.*AU,
            10000,
            1e-1,
            3,
        ).unwrap();
    }
}
