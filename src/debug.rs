#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::str::FromStr;

/// How much detail `select_next_item`/`estimate_theta` output gets persisted to
/// `trans_session_item_debug`. Only affects what's *written* — the underlying
/// computation (full candidate scoring, full Newton-Raphson trace) always runs in
/// full regardless of mode; `playground.rs` also always returns the untrimmed debug
/// structs in its API response, independent of any preset's `debug_mode`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum DebugMode {
    /// No `trans_session_item_debug` row is written at all.
    Off,
    /// Row is written, but `selection_detail.candidates` is capped to the preset's
    /// `debug_sample_size` best-scoring candidates and `estimation_detail.iterations`
    /// to that many trailing entries — see `SelectionDebug::sampled`/`EstimationDebug::sampled`.
    Sample,
    /// Every candidate and every iteration is persisted, unabridged.
    Full,
}

impl FromStr for DebugMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "off" => Ok(Self::Off),
            "sample" => Ok(Self::Sample),
            "full" => Ok(Self::Full),
            other => Err(format!("Unknown debug mode: {other}")),
        }
    }
}

/// Default `master_test_settings.debug_sample_size` for new presets that don't specify
/// one. The actual cap applied in `DebugMode::Sample` is always the preset's own
/// `debug_sample_size`, not this constant — see `TestSettings::debug_sample_size`.
pub const DEFAULT_DEBUG_SAMPLE_SIZE: i32 = 10;

#[derive(Debug, Serialize)]
pub struct CandidateDebug {
    pub item_id: String,
    pub eligible: bool,
    /// Raw method score (D-optimal/A-optimal/KL); `None` if ineligible.
    pub raw_score: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct SelectionDebug {
    pub method: String,
    pub exposure_method: String,
    /// Active, not-yet-administered items considered this round.
    pub candidates: Vec<CandidateDebug>,
    /// `None` when no eligible item was found (bank exhausted).
    pub winner_item_id: Option<String>,
    pub winner_score: Option<f64>,
}

impl SelectionDebug {
    /// Trims `candidates` down to the `n` closest to the winner (i.e. the strongest
    /// contenders): sorted by `raw_score` descending, then truncated. Ineligible
    /// candidates carry no score, so they sort last and are dropped first — sample mode
    /// is about showing the near-ties, not the exclusions.
    pub fn sampled(mut self, n: usize) -> Self {
        self.candidates.sort_by(|a, b| {
            let score_a = a.raw_score.unwrap_or(f64::NEG_INFINITY);
            let score_b = b.raw_score.unwrap_or(f64::NEG_INFINITY);
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        self.candidates.truncate(n);
        self
    }
}

#[derive(Debug, Serialize)]
pub struct EstimationIteration {
    pub iter: usize,
    pub theta: Vec<f64>,
    pub delta_norm: f64,
}

#[derive(Debug, Serialize)]
pub struct EstimationDebug {
    pub method: String,
    pub items_used: usize,
    /// Newton-Raphson trace (MLE/MAP); empty for EAP, which has no iterative refinement.
    pub iterations: Vec<EstimationIteration>,
    /// `None` for EAP, where "convergence" isn't applicable.
    pub converged: Option<bool>,
    pub final_theta: Vec<f64>,
    /// Per-dimension posterior SD, when the estimator can produce it directly from its
    /// own uncertainty representation instead of the shared inverse-information formula:
    /// `Some` for EAP (SD of the quadrature posterior), `None` for MLE/MAP (whose SE is
    /// derived afterward from the cumulative Fisher Information Matrix instead — see
    /// `McatEngine::compute_se`).
    pub posterior_se: Option<Vec<Option<f64>>>,
    /// Method-specific extras, e.g. EAP's quadrature point count / posterior mass.
    pub extra: serde_json::Value,
}

impl EstimationDebug {
    /// Trims `iterations` down to the last `n` (the ones nearest convergence). EAP's
    /// `iterations` is already empty, so this is a no-op for it.
    pub fn sampled(mut self, n: usize) -> Self {
        let len = self.iterations.len();
        if len > n {
            self.iterations = self.iterations.split_off(len - n);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_mode_from_str_roundtrips_known_values() {
        assert_eq!(DebugMode::from_str("off").unwrap(), DebugMode::Off);
        assert_eq!(DebugMode::from_str("sample").unwrap(), DebugMode::Sample);
        assert_eq!(DebugMode::from_str("full").unwrap(), DebugMode::Full);
        assert!(DebugMode::from_str("bogus").is_err());
    }

    fn candidate(id: &str, score: Option<f64>) -> CandidateDebug {
        CandidateDebug {
            item_id: id.to_string(),
            eligible: score.is_some(),
            raw_score: score,
        }
    }

    #[test]
    fn selection_debug_sampled_keeps_top_n_by_score_descending() {
        let debug = SelectionDebug {
            method: "d_optimal".to_string(),
            exposure_method: "none".to_string(),
            candidates: vec![
                candidate("low", Some(1.0)),
                candidate("high", Some(3.0)),
                candidate("mid", Some(2.0)),
            ],
            winner_item_id: Some("high".to_string()),
            winner_score: Some(3.0),
        };

        let sampled = debug.sampled(2);
        let ids: Vec<&str> = sampled
            .candidates
            .iter()
            .map(|c| c.item_id.as_str())
            .collect();
        assert_eq!(ids, vec!["high", "mid"]);
    }

    #[test]
    fn selection_debug_sampled_sorts_ineligible_candidates_last() {
        let debug = SelectionDebug {
            method: "d_optimal".to_string(),
            exposure_method: "sympson_hetter".to_string(),
            candidates: vec![
                candidate("ineligible", None),
                candidate("eligible", Some(1.0)),
            ],
            winner_item_id: Some("eligible".to_string()),
            winner_score: Some(1.0),
        };

        let sampled = debug.sampled(1);
        assert_eq!(sampled.candidates.len(), 1);
        assert_eq!(sampled.candidates[0].item_id, "eligible");
    }

    #[test]
    fn selection_debug_sampled_is_a_noop_when_n_exceeds_candidate_count() {
        let debug = SelectionDebug {
            method: "d_optimal".to_string(),
            exposure_method: "none".to_string(),
            candidates: vec![candidate("only", Some(1.0))],
            winner_item_id: Some("only".to_string()),
            winner_score: Some(1.0),
        };

        assert_eq!(debug.sampled(10).candidates.len(), 1);
    }

    fn estimation_debug(n_iterations: usize) -> EstimationDebug {
        EstimationDebug {
            method: "mle".to_string(),
            items_used: n_iterations,
            iterations: (0..n_iterations)
                .map(|iter| EstimationIteration {
                    iter,
                    theta: vec![0.0],
                    delta_norm: 0.0,
                })
                .collect(),
            converged: Some(true),
            final_theta: vec![0.0],
            posterior_se: None,
            extra: serde_json::Value::Null,
        }
    }

    #[test]
    fn estimation_debug_sampled_keeps_the_last_n_iterations() {
        let sampled = estimation_debug(5).sampled(2);
        let iters: Vec<usize> = sampled.iterations.iter().map(|it| it.iter).collect();
        assert_eq!(iters, vec![3, 4]);
    }

    #[test]
    fn estimation_debug_sampled_is_a_noop_when_n_exceeds_iteration_count() {
        let sampled = estimation_debug(2).sampled(10);
        assert_eq!(sampled.iterations.len(), 2);
    }
}

#[derive(Debug, Serialize)]
pub struct StoppingDebug {
    pub rule: String,
    pub n: usize,
    pub n_min: usize,
    pub n_max: usize,
    pub se: Vec<Option<f64>>,
    pub se_threshold: f64,
    pub delta_norm: f64,
    pub convergence_delta: f64,
    pub decision: bool,
    pub reason: String,
}
