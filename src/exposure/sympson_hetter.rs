use rand::Rng;

/// Accept item with probability `sh_r_param` - the classic Sympson & Hetter (1985)
/// control parameter `P(A|S)` (a.k.a. `K_i`), used directly and unmodified as the
/// Bernoulli gate. Matches the reference implementation in `mirtCAT`
/// (`findNextCATItem.R`, `exposure_type == 'SH'`: `design@exposure[item] >= runif(1,0,1)`).
///
/// `sh_r_param` must be pre-calibrated offline (per item, via simulation) so that the
/// resulting administration rate `P(A)` converges to the desired target exposure rate —
/// this function performs no such calibration itself; it only applies the parameter.
/// See `experimental/exposure/exposure.md` §2.2 for the theory writeup and the
/// `mirtCAT` comparison this implementation was aligned to.
pub fn is_eligible(sh_r_param: f64) -> bool {
    rand::thread_rng().gen_bool(sh_r_param.clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn r_param_of_one_always_admits() {
        for _ in 0..200 {
            assert!(is_eligible(1.0));
        }
    }

    #[test]
    fn r_param_of_zero_never_admits() {
        for _ in 0..200 {
            assert!(!is_eligible(0.0));
        }
    }

    #[test]
    fn out_of_range_r_param_is_clamped() {
        for _ in 0..200 {
            assert!(is_eligible(5.0)); // clamps to 1.0
            assert!(!is_eligible(-5.0)); // clamps to 0.0
        }
    }
}
