use super::StoppingInput;

pub fn check(i: &StoppingInput) -> (bool, &'static str) {
    if i.n >= i.n_max {
        (true, "fixed_length")
    } else {
        (false, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn input(n: usize, n_max: usize, theta: &DVector<f64>) -> StoppingInput<'_> {
        StoppingInput {
            n,
            n_min: 0,
            n_max,
            se_threshold: 0.0,
            delta: 0.0,
            se: &[],
            theta_now: theta,
            theta_prev: theta,
        }
    }

    #[test]
    fn stops_once_n_reaches_n_max() {
        let theta = DVector::from_vec(vec![0.0]);
        let (stop, reason) = check(&input(30, 30, &theta));
        assert!(stop);
        assert_eq!(reason, "fixed_length");
    }

    #[test]
    fn stops_when_n_exceeds_n_max() {
        let theta = DVector::from_vec(vec![0.0]);
        assert!(check(&input(31, 30, &theta)).0);
    }

    #[test]
    fn does_not_stop_before_n_max() {
        let theta = DVector::from_vec(vec![0.0]);
        let (stop, reason) = check(&input(29, 30, &theta));
        assert!(!stop);
        assert_eq!(reason, "");
    }
}
