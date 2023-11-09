use numpy::ToPyArray;
use pyo3::{exceptions::PyValueError, prelude::*};
use schwarzchild_sim as schwarz_rs;

/// Some docs
#[pyclass(module = "schwarz")]
#[derive(Debug, Clone)]
struct BodyParameters(schwarz_rs::BodyParameters);

#[pymethods]
impl BodyParameters {
    fn __str__(&self) -> String {
        format!("{:?}", self.0)
    }

    fn __repr__(&self) -> String {
        format!("{:#?}", self.0)
    }

    #[getter]
    fn mass(&self) -> f64 {
        self.0.mass
    }

    #[setter]
    fn set_mass(&mut self, mass: f64) {
        self.0.mass = mass;
    }

    #[getter]
    fn radius(&self) -> f64 {
        self.0.radius
    }

    #[setter]
    fn set_radius(&mut self, radius: f64) {
        self.0.radius = radius;
    }

    #[getter]
    fn radial_velocity(&self) -> f64 {
        self.0.radial_velocity
    }

    #[setter]
    fn set_radial_velocity(&mut self, radial_velocity: f64) {
        self.0.radial_velocity = radial_velocity;
    }

    #[getter]
    fn angle(&self) -> f64 {
        self.0.angle
    }

    #[setter]
    fn set_angle(&mut self, angle: f64) {
        self.0.angle = angle;
    }

    #[getter]
    fn angular_velocity(&self) -> f64 {
        self.0.angular_velocity
    }

    #[setter]
    fn set_angular_velocity(&mut self, angular_velocity: f64) {
        self.0.angular_velocity = angular_velocity;
    }

    #[new]
    fn new(m: f64, r: f64, v: f64, theta: f64, omega: f64) -> Self {
        Self(schwarz_rs::BodyParameters {
            mass: m,
            radius: r,
            radial_velocity: v,
            angle: theta,
            angular_velocity: omega,
        })
    }
}

impl From<schwarz_rs::BodyParameters> for BodyParameters {
    fn from(b: schwarz_rs::BodyParameters) -> Self {
        Self(b)
    }
}

impl Into<schwarz_rs::BodyParameters> for BodyParameters {
    fn into(self) -> schwarz_rs::BodyParameters {
        self.0
    }
}

#[pyfunction]
fn schwarzchild_radius(mass: f64) -> f64 {
    schwarz_rs::schwarzchild_radius(mass)
}

#[pymodule]
fn schwarz(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BodyParameters>()?;
    m.add_class::<Solvers>()?;

    m.add_function(wrap_pyfunction!(schwarzchild_radius, m)?)?;

    m.add("G", schwarz_rs::G)?;
    m.add("C", schwarz_rs::C)?;
    m.add("M_SUN", schwarz_rs::M_SUN)?;
    m.add("AU", schwarz_rs::AU)?;

    m.add::<BodyParameters>(
        "mercury_orbit",
        schwarz_rs::BodyParameters {
            mass: schwarz_rs::M_SUN,
            radius: 6.9818e10,
            radial_velocity: 0.,
            angle: 0.,
            angular_velocity: 3.886e4 / 6.9818e10,
        }
        .into(),
    )?;
    m.add::<BodyParameters>(
        "small_precession",
        schwarz_rs::BodyParameters {
            mass: schwarz_rs::M_SUN,
            radius: schwarzchild_radius(schwarz_rs::M_SUN) * 200.,
            radial_velocity: 0.,
            angle: 0.,
            angular_velocity: 35.,
        }
        .into(),
    )?;

    m.add_submodule(simulate_module(py)?)?;

    Ok(())
}

fn simulate_module(py: Python) -> PyResult<&'_ PyModule> {
    let m = PyModule::new(py, "simulate")?;
    m.add_function(wrap_pyfunction!(simulate_conditions, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_euler_1, m)?)?;
    Ok(m)
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
enum Solvers {
    Euler1,
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, solver, history_interval = 1, *, max_time, max_r, max_theta))]
fn simulate_conditions(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    solver: Solvers,
    history_interval: usize,
    max_time: Option<f64>,
    max_r: Option<f64>,
    max_theta: Option<f64>,
) -> PyResult<&numpy::PyArray2<f64>> {
    match solver {
        Solvers::Euler1 => simulate_euler_1(
            py,
            initial_condition,
            time_step,
            history_interval,
            max_time,
            max_r,
            max_theta,
        ),
    }
}

fn make_end_condition(
    max_time: Option<f64>,
    max_r: Option<f64>,
    max_theta: Option<f64>,
) -> PyResult<impl Fn(&schwarz_rs::BodyParameters, f64) -> bool> {
    if max_time.is_none() && max_r.is_none() && max_theta.is_none() {
        return Err(PyValueError::new_err(
            "At least one of `max_time`, `max_r`, or `max_theta` must be provided",
        ));
    }

    Ok(move |state: &schwarz_rs::BodyParameters, time: f64| {
        let mut end = false;
        if let Some(max_time) = max_time {
            end |= time >= max_time;
        }
        if let Some(max_r) = max_r {
            end |= state.radius >= max_r;
        }
        if let Some(max_theta) = max_theta {
            end |= state.angle >= max_theta;
        }
        end
    })
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, history_interval = 1, *, max_time = None, max_r = None, max_theta = None))]
fn simulate_euler_1(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    history_interval: usize,
    max_time: Option<f64>,
    max_r: Option<f64>,
    max_theta: Option<f64>,
) -> PyResult<&numpy::PyArray2<f64>> {
    let end_condition = make_end_condition(max_time, max_r, max_theta)?;
    let history = schwarz_rs::simulate_conditions(
        initial_condition.into(),
        end_condition,
        history_interval,
        time_step,
        schwarz_rs::EulerSolve1::new(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(history.to_pyarray(py))
}
