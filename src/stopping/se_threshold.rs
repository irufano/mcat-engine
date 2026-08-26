use super::{StoppingInput, max_se};

pub fn check(i: &StoppingInput) -> (bool, &'static str) {
    if max_se(i.se) <= i.se_threshold {
        (true, "se_threshold")
    } else {
        (false, "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    fn input<'a>(
        se: &'a [Option<f64>],
        threshold: f64,
        theta: &'a DVector<f64>,
    ) -> StoppingInput<'a> {
        StoppingInput {
            n: 5,
            n_min: 0,
            n_max: 100,
            se_threshold: threshold,
            delta: 0.0,
            se,
            theta_now: theta,
            theta_prev: theta,
        }
    }

    #[test]
    fn stops_when_worst_dimension_se_is_at_or_below_threshold() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.2), Some(0.29)];
        let (stop, reason) = check(&input(&se, 0.30, &theta));
        assert!(stop);
        assert_eq!(reason, "se_threshold");
    }

    #[test]
    fn does_not_stop_when_any_dimension_exceeds_threshold() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.2), Some(0.35)];
        assert!(!check(&input(&se, 0.30, &theta)).0);
    }

    #[test]
    fn undefined_se_counts_as_infinite_and_never_stops() {
        let theta = DVector::from_vec(vec![0.0]);
        let se = [Some(0.1), None];
        assert!(!check(&input(&se, 0.30, &theta)).0);
    }
}
