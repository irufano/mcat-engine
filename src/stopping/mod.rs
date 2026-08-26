use nalgebra::DVector;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub mod convergence;
pub mod fixed_length;
pub mod hybrid;
pub mod se_threshold;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StoppingRule {
    FixedLength,
    SeThreshold,
    Convergence,
    Hybrid,
}

impl FromStr for StoppingRule {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "fixed_length" => Ok(Self::FixedLength),
            "se_threshold" => Ok(Self::SeThreshold),
            "convergence" => Ok(Self::Convergence),
            "hybrid" => Ok(Self::Hybrid),
            other => Err(format!("Unknown stopping rule: {other}")),
        }
    }
}

pub struct StoppingInput<'a> {
    pub n: usize,
    pub n_min: usize,
    pub n_max: usize,
    pub se_threshold: f64,
    pub delta: f64,
    /// Per-dimension SE, already computed by the method-appropriate formula
    /// (`McatEngine::compute_se`) — MLE/MAP/EAP each mean something different by
    /// "information", so stopping rules only ever see the final SE, never a raw FIM.
    pub se: &'a [Option<f64>],
    pub theta_now: &'a DVector<f64>,
    pub theta_prev: &'a DVector<f64>,
}

pub fn should_stop(rule: &StoppingRule, input: &StoppingInput) -> (bool, &'static str) {
    match rule {
        StoppingRule::FixedLength => fixed_length::check(input),
        StoppingRule::SeThreshold => se_threshold::check(input),
        StoppingRule::Convergence => convergence::check(input),
        StoppingRule::Hybrid => hybrid::check(input),
    }
}

/// Worst-case (largest) per-dimension SE — `None` (undefined SE) counts as infinity, so a
/// stopping rule checking `max_se <= se_threshold` never fires on an undefined dimension.
pub fn max_se(se: &[Option<f64>]) -> f64 {
    se.iter()
        .map(|s| s.unwrap_or(f64::INFINITY))
        .fold(0.0_f64, f64::max)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_known_rules() {
        assert_eq!(
            "fixed_length".parse::<StoppingRule>().unwrap(),
            StoppingRule::FixedLength
        );
        assert_eq!(
            "se_threshold".parse::<StoppingRule>().unwrap(),
            StoppingRule::SeThreshold
        );
        assert_eq!(
            "convergence".parse::<StoppingRule>().unwrap(),
            StoppingRule::Convergence
        );
        assert_eq!(
            "hybrid".parse::<StoppingRule>().unwrap(),
            StoppingRule::Hybrid
        );
    }

    #[test]
    fn rejects_unknown_rule() {
        assert!("bogus".parse::<StoppingRule>().is_err());
    }

    #[test]
    fn max_se_treats_none_as_infinity() {
        assert_eq!(max_se(&[Some(0.1), None, Some(0.2)]), f64::INFINITY);
    }

    #[test]
    fn max_se_of_empty_is_zero() {
        assert_eq!(max_se(&[]), 0.0);
    }

    #[test]
    fn max_se_picks_largest_defined_value() {
        assert_eq!(max_se(&[Some(0.1), Some(0.4), Some(0.2)]), 0.4);
    }

    #[test]
    fn dispatch_routes_to_the_matching_rule() {
        let theta = DVector::from_vec(vec![0.0]);
        let input = StoppingInput {
            n: 30,
            n_min: 0,
            n_max: 30,
            se_threshold: 1.0,
            delta: 1.0,
            se: &[],
            theta_now: &theta,
            theta_prev: &theta,
        };
        let (stop, reason) = should_stop(&StoppingRule::FixedLength, &input);
        assert!(stop);
        assert_eq!(reason, "fixed_length");
    }
}
