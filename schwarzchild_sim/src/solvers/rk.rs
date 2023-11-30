use std::ops::{Add, AddAssign, Mul};

use ndarray::Array2;

use crate::{BodyParameters, C, G};

#[derive(Debug, Clone, Copy)]
struct State {
    r: f64,
    v: f64,
    a: f64,
    w: f64,
}

impl Add for State {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            v: self.v + other.v,
            a: self.a + other.a,
            w: self.w + other.w,
        }
    }
}

impl AddAssign for State {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.v += other.v;
        self.a += other.a;
        self.w += other.w;
    }
}

impl Mul<f64> for State {
    type Output = Self;

    #[inline]
    fn mul(self, other: f64) -> Self {
        Self {
            r: self.r * other,
            v: self.v * other,
            a: self.a * other,
            w: self.w * other,
        }
    }
}

impl Mul<State> for f64 {
    type Output = State;

    #[inline]
    fn mul(self, other: State) -> State {
        State {
            r: self * other.r,
            v: self * other.v,
            a: self * other.a,
            w: self * other.w,
        }
    }
}

impl From<BodyParameters> for State {
    fn from(body: BodyParameters) -> Self {
        Self {
            r: body.radius,
            v: body.radial_velocity,
            a: body.angle,
            w: body.angular_velocity,
        }
    }
}

pub fn simulate_rk4(
    initial_condition: BodyParameters,
    max_t: f64,
    history_interval: usize,
    time_step: f64,
) -> Array2<f64> {
    let n = (max_t / time_step).ceil() as usize;
    let h = time_step;
    let r_s = 2. * G * initial_condition.mass / (C * C);
    let mut x = State::from(initial_condition);
    let l = x.r * x.r * x.w;

    let mut history = vec![[x.r, x.a, 0.]];

    let f = |x: State| State {
        r: x.v,
        v: -0.5
            * (r_s / x.r / x.r * (C * C + x.r * x.r * x.w * x.w)
                - 2. * x.r * x.w * x.w * (1. - r_s / x.r)),
        a: x.w,
        w: 0.,
    };

    let conserve_l = |x: &mut State| {
        x.w = l / x.r / x.r;
    };

    let mut positive_v = true;
    for i in 0..n {
        let k1 = f(x);
        let k2 = f(x + 0.5 * h * k1);
        let k3 = f(x + 0.5 * h * k2);
        let k4 = f(x + h * k3);
        x += h * (k1 + 2. * k2 + 2. * k3 + k4) * (1. / 6.);
        conserve_l(&mut x);
        if i % history_interval == 0 || {
            let changed = (x.v >= 0.) ^ positive_v;
            if changed {
                positive_v = !positive_v;
            }
            changed
        } {
            history.push([x.r, x.a, i as f64 * h]);
            print!("\r{:.2}%", 100. * i as f32 / n as f32);
        }
    }
    println!("\r100.00%");

    history.push([x.r, x.a, n as f64 * h]);
    Array2::from(history)
}
