use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub mod eap;
pub mod map;
pub mod mle;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum EstimationMethod {
    Mle,
    Map,
    Eap,
}

impl FromStr for EstimationMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "mle" => Ok(Self::Mle),
            "map" => Ok(Self::Map),
            "eap" => Ok(Self::Eap),
            other => Err(format!("Unknown estimation method: {other}")),
        }
    }
}

pub struct EstimationInput<'a> {
    pub item_a_vecs: &'a [DVector<f64>],
    pub item_d: &'a [f64],
    pub item_c: &'a [f64],
    pub responses: &'a [u8],
    pub prior_mean: &'a DVector<f64>,
    pub prior_cov_inv: &'a DMatrix<f64>,
    pub eap_quad_pts: usize,
    pub k: usize,
}

pub fn estimate(
    method: &EstimationMethod,
    input: &EstimationInput,
) -> (DVector<f64>, crate::debug::EstimationDebug) {
    match method {
        EstimationMethod::Mle => mle::estimate(input),
        EstimationMethod::Map => map::estimate(input),
        EstimationMethod::Eap => eap::estimate(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_known_methods() {
        assert_eq!(
            "mle".parse::<EstimationMethod>().unwrap(),
            EstimationMethod::Mle
        );
        assert_eq!(
            "map".parse::<EstimationMethod>().unwrap(),
            EstimationMethod::Map
        );
        assert_eq!(
            "eap".parse::<EstimationMethod>().unwrap(),
            EstimationMethod::Eap
        );
    }

    #[test]
    fn rejects_unknown_method() {
        assert!("bogus".parse::<EstimationMethod>().is_err());
    }

    #[test]
    fn dispatch_routes_to_the_matching_method() {
        let a = [DVector::from_vec(vec![1.0])];
        let d = [0.0];
        let c = [0.0];
        let r = [1u8];
        let prior_mean = DVector::from_vec(vec![0.0]);
        let prior_cov_inv = DMatrix::<f64>::identity(1, 1);
        let input = EstimationInput {
            item_a_vecs: &a,
            item_d: &d,
            item_c: &c,
            responses: &r,
            prior_mean: &prior_mean,
            prior_cov_inv: &prior_cov_inv,
            eap_quad_pts: 21,
            k: 1,
        };

        assert_eq!(estimate(&EstimationMethod::Mle, &input).1.method, "mle");
        assert_eq!(estimate(&EstimationMethod::Map, &input).1.method, "map");
        assert_eq!(estimate(&EstimationMethod::Eap, &input).1.method, "eap");
    }
}
