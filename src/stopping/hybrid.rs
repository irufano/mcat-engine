use super::{StoppingInput, max_se};

pub fn check(i: &StoppingInput) -> (bool, &'static str) {
    if i.n < i.n_min {
        return (false, "");
    }
    if i.n >= i.n_max {
        return (true, "max_length");
    }
    if max_se(i.se) <= i.se_threshold {
        return (true, "se_threshold");
    }
    (false, "")
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[allow(clippy::too_many_arguments)]
    fn input<'a>(
        n: usize,
        n_min: usize,
        n_max: usize,
        se: &'a [Option<f64>],
        threshold: f64,
        theta: &'a DVector<f64>,
    ) -> StoppingInput<'a> {
        StoppingInput {
            n,
            n_min,
            n_max,
            se_threshold: threshold,
            delta: 0.0,
            se,
            theta_now: theta,
            theta_prev: theta,
        }
    }

    #[test]
    fn never_stops_before_n_min_even_if_se_is_already_low() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.01)];
        let (stop, reason) = check(&input(2, 5, 30, &se, 0.30, &theta));
        assert!(!stop);
        assert_eq!(reason, "");
    }

    #[test]
    fn stops_at_n_max_regardless_of_se() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(5.0)];
        let (stop, reason) = check(&input(30, 5, 30, &se, 0.30, &theta));
        assert!(stop);
        assert_eq!(reason, "max_length");
    }

    #[test]
    fn stops_on_se_threshold_once_past_n_min() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.25)];
        let (stop, reason) = check(&input(10, 5, 30, &se, 0.30, &theta));
        assert!(stop);
        assert_eq!(reason, "se_threshold");
    }

    #[test]
    fn continues_when_past_n_min_but_se_still_high() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.5)];
        let (stop, reason) = check(&input(10, 5, 30, &se, 0.30, &theta));
        assert!(!stop);
        assert_eq!(reason, "");
    }
}
