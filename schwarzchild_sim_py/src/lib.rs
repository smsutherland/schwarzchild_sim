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
#[pyo3(name = "_inner")]
fn schwarz(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<BodyParameters>()?;
    m.add_class::<Solvers>()?;

    m.add_function(wrap_pyfunction!(schwarzchild_radius, m)?)?;

    m.add("G", schwarz_rs::G)?;
    m.add("C", schwarz_rs::C)?;
    m.add("M_SUN", schwarz_rs::M_SUN)?;
    m.add("AU", schwarz_rs::AU)?;

    m.add_submodule(simulate_module(py)?)?;

    Ok(())
}

fn simulate_module(py: Python) -> PyResult<&'_ PyModule> {
    let m = PyModule::new(py, "simulate")?;
    m.add_function(wrap_pyfunction!(run, m)?)?;
    m.add_function(wrap_pyfunction!(euler_1, m)?)?;
    m.add_function(wrap_pyfunction!(euler_2, m)?)?;
    m.add_function(wrap_pyfunction!(euler_3, m)?)?;
    m.add_function(wrap_pyfunction!(euler_4, m)?)?;
    Ok(m)
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
enum Solvers {
    Euler1,
    Euler2,
    Euler3,
    Euler4,
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, solver, end_time, history_interval = 1))]
fn run(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    solver: Solvers,
    end_time: f64,
    history_interval: usize,
) -> PyResult<&numpy::PyArray2<f64>> {
    match solver {
        Solvers::Euler1 => euler_1(py, initial_condition, time_step, end_time, history_interval),
        Solvers::Euler2 => euler_2(py, initial_condition, time_step, end_time, history_interval),
        Solvers::Euler3 => euler_3(py, initial_condition, time_step, end_time, history_interval),
        Solvers::Euler4 => euler_4(py, initial_condition, time_step, end_time, history_interval),
    }
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, end_time, history_interval = 1))]
fn euler_1(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    end_time: f64,
    history_interval: usize,
) -> PyResult<&numpy::PyArray2<f64>> {
    let history = schwarz_rs::simulate_euler(
        initial_condition.into(),
        end_time,
        history_interval,
        time_step,
        schwarz_rs::EulerSolve1::new(),
    )
    .map_err(PyValueError::new_err)?;
    Ok(history.to_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, end_time, history_interval = 1))]
fn euler_2(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    end_time: f64,
    history_interval: usize,
) -> PyResult<&numpy::PyArray2<f64>> {
    let history = schwarz_rs::simulate_euler(
        initial_condition.into(),
        end_time,
        history_interval,
        time_step,
        schwarz_rs::EulerSolve2,
    )
    .map_err(PyValueError::new_err)?;
    Ok(history.to_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, end_time, history_interval = 1))]
fn euler_3(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    end_time: f64,
    history_interval: usize,
) -> PyResult<&numpy::PyArray2<f64>> {
    let history = schwarz_rs::simulate_euler(
        initial_condition.into(),
        end_time,
        history_interval,
        time_step,
        schwarz_rs::EulerSolve3,
    )
    .map_err(PyValueError::new_err)?;
    Ok(history.to_pyarray(py))
}

#[pyfunction]
#[pyo3(signature = (initial_condition, time_step, end_time, history_interval = 1))]
fn euler_4(
    py: Python,
    initial_condition: BodyParameters,
    time_step: f64,
    end_time: f64,
    history_interval: usize,
) -> PyResult<&numpy::PyArray2<f64>> {
    let history = schwarz_rs::simulate_euler(
        initial_condition.into(),
        end_time,
        history_interval,
        time_step,
        schwarz_rs::EulerSolve4,
    )
    .map_err(PyValueError::new_err)?;
    Ok(history.to_pyarray(py))
}
