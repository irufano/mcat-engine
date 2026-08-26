use super::StoppingInput;

pub fn check(i: &StoppingInput) -> (bool, &'static str) {
    let diff = (i.theta_now - i.theta_prev).norm();
    if diff <= i.delta {
        (true, "convergence")
    } else {
        (false, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn input<'a>(
        theta_now: &'a DVector<f64>,
        theta_prev: &'a DVector<f64>,
        delta: f64,
    ) -> StoppingInput<'a> {
        StoppingInput {
            n: 5,
            n_min: 0,
            n_max: 100,
            se_threshold: 0.0,
            delta,
            se: &[],
            theta_now,
            theta_prev,
        }
    }

    #[test]
    fn stops_when_theta_change_is_within_delta() {
        let now = DVector::from_vec(vec![1.00]);
        let prev = DVector::from_vec(vec![1.005]);
        let (stop, reason) = check(&input(&now, &prev, 0.01));
        assert!(stop);
        assert_eq!(reason, "convergence");
    }

    #[test]
    fn does_not_stop_when_theta_change_exceeds_delta() {
        let now = DVector::from_vec(vec![1.0]);
        let prev = DVector::from_vec(vec![0.5]);
        let (stop, reason) = check(&input(&now, &prev, 0.01));
        assert!(!stop);
        assert_eq!(reason, "");
    }

    #[test]
    fn multidimensional_change_uses_euclidean_norm() {
        // ||[0.3, 0.4]|| = 0.5
        let now = DVector::from_vec(vec![0.3, 0.4]);
        let prev = DVector::from_vec(vec![0.0, 0.0]);
        assert!(check(&input(&now, &prev, 0.5)).0);
        assert!(!check(&input(&now, &prev, 0.49)).0);
    }

    #[test]
    fn exact_equality_at_delta_counts_as_convergence() {
        let now = DVector::from_vec(vec![1.0]);
        let prev = DVector::from_vec(vec![0.0]);
        let (stop, reason) = check(&input(&now, &prev, 1.0));
        assert!(stop);
        assert_eq!(reason, "convergence");
    }
}
