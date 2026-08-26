#![allow(dead_code)]

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};
use std::str::FromStr;

pub mod a_optimal;
pub mod d_optimal;
pub mod kl_information;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SelectionMethod {
    DOptimal,
    AOptimal,
    KlInformation,
}

impl FromStr for SelectionMethod {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "d_optimal" => Ok(Self::DOptimal),
            "a_optimal" => Ok(Self::AOptimal),
            "kl_information" => Ok(Self::KlInformation),
            other => Err(format!("Unknown selection method: {other}")),
        }
    }
}

pub struct SelectionContext<'a> {
    pub theta: &'a DVector<f64>,
    pub cum_fim: &'a DMatrix<f64>,
    pub k: usize,
    /// Number of items already administered in the session (m in Chang & Ying's (1996)
    /// shrinking interval δ_m = C / sqrt(m + 1)); used only by KL information.
    pub administered: usize,
}

/// Score an item for selection; higher = better.
///
/// `a`, `d`, `c` are the candidate item's own M3PL parameters — required by KL information,
/// which (unlike D-optimal/A-optimal) is not a function of `item_fim` alone.
pub fn score_item(
    method: &SelectionMethod,
    ctx: &SelectionContext,
    item_fim: &DMatrix<f64>,
    a: &DVector<f64>,
    d: f64,
    c: f64,
) -> f64 {
    let updated = ctx.cum_fim + item_fim;
    match method {
        SelectionMethod::DOptimal => d_optimal::score(&updated),
        SelectionMethod::AOptimal => a_optimal::score(&updated),
        SelectionMethod::KlInformation => {
            kl_information::score(ctx.theta, a, d, c, ctx.administered)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_known_methods() {
        assert_eq!(
            "d_optimal".parse::<SelectionMethod>().unwrap(),
            SelectionMethod::DOptimal
        );
        assert_eq!(
            "a_optimal".parse::<SelectionMethod>().unwrap(),
            SelectionMethod::AOptimal
        );
        assert_eq!(
            "kl_information".parse::<SelectionMethod>().unwrap(),
            SelectionMethod::KlInformation
        );
    }

    #[test]
    fn rejects_unknown_method() {
        assert!("bogus".parse::<SelectionMethod>().is_err());
    }

    #[test]
    fn d_optimal_dispatch_matches_direct_call() {
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::identity(1, 1);
        let ctx = SelectionContext {
            theta: &theta,
            cum_fim: &cum_fim,
            k: 1,
            administered: 0,
        };
        let item_fim = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0]));
        let a = DVector::from_vec(vec![1.0]);
        let via_dispatch = score_item(&SelectionMethod::DOptimal, &ctx, &item_fim, &a, 0.0, 0.0);
        let expected = d_optimal::score(&(&cum_fim + &item_fim));
        assert!((via_dispatch - expected).abs() < 1e-9);
    }

    #[test]
    fn a_optimal_dispatch_matches_direct_call() {
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::identity(1, 1);
        let ctx = SelectionContext {
            theta: &theta,
            cum_fim: &cum_fim,
            k: 1,
            administered: 0,
        };
        let item_fim = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0]));
        let a = DVector::from_vec(vec![1.0]);
        let via_dispatch = score_item(&SelectionMethod::AOptimal, &ctx, &item_fim, &a, 0.0, 0.0);
        let expected = a_optimal::score(&(&cum_fim + &item_fim));
        assert!((via_dispatch - expected).abs() < 1e-9);
    }

    #[test]
    fn kl_information_dispatch_matches_direct_call() {
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let ctx = SelectionContext {
            theta: &theta,
            cum_fim: &cum_fim,
            k: 1,
            administered: 3,
        };
        let item_fim = DMatrix::<f64>::zeros(1, 1);
        let a = DVector::from_vec(vec![1.2]);
        let via_dispatch = score_item(
            &SelectionMethod::KlInformation,
            &ctx,
            &item_fim,
            &a,
            0.1,
            0.0,
        );
        let expected = kl_information::score(&theta, &a, 0.1, 0.0, 3);
        assert!((via_dispatch - expected).abs() < 1e-9);
    }
}
