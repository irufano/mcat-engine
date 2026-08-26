#![allow(dead_code)]

use crate::error::{EngineError, EngineResult};
use crate::{
    debug::{CandidateDebug, EstimationDebug, SelectionDebug, StoppingDebug},
    estimation::{self, EstimationInput, EstimationMethod},
    exposure::ExposureMethod,
    mirt::{item_fim, se_vector},
    selection::{SelectionContext, SelectionMethod, score_item},
    stopping::{StoppingInput, StoppingRule, should_stop},
    types::{EngineItem, EngineSettings},
};
use nalgebra::{DMatrix, DVector};

pub struct McatEngine;

impl McatEngine {
    /// Select the best next item from the eligible pool given current session state.
    ///
    /// `candidates` pairs each eligible item with its a-vector *already
    /// projected* into the session's tested-dimension order (see
    /// `crate::dimensions::project_item`) — this function never reads
    /// `item.a_params` directly, so masking/projection is entirely the
    /// caller's responsibility (`session_handler`, `playground`). An item
    /// that measures a dimension outside what's being tested must simply be
    /// absent from `candidates`, not present with a truncated vector.
    pub fn select_next_item(
        settings: &EngineSettings,
        candidates: &[(&EngineItem, DVector<f64>)],
        administered_ids: &[String],
        theta: &DVector<f64>,
        cum_fim: &DMatrix<f64>,
        k: usize,
    ) -> EngineResult<(Option<(usize, f64)>, SelectionDebug)> {
        let selection_method: SelectionMethod =
            settings.selection_method.parse().map_err(|_| {
                EngineError::InvalidConfig(format!(
                    "Unknown selection method: {}",
                    settings.selection_method
                ))
            })?;

        let exposure_method: ExposureMethod = settings.exposure_method.parse().map_err(|_| {
            EngineError::InvalidConfig(format!(
                "Unknown exposure method: {}",
                settings.exposure_method
            ))
        })?;

        let ctx = SelectionContext {
            theta,
            cum_fim,
            k,
            administered: administered_ids.len(),
        };
        let mut best: Option<(usize, f64)> = None;
        let mut candidate_debug: Vec<CandidateDebug> = Vec::new();

        for (idx, (item, a)) in candidates.iter().enumerate() {
            if administered_ids.contains(&item.id) {
                continue;
            }
            if !item.is_active {
                continue;
            }

            // The Sympson-Hetter gate matches theory (Sympson & Hetter, 1985) and
            // mirtCAT by using only the pre-calibrated per-item `sh_r_param` directly —
            // exposure history is not an input to the gate itself (see
            // `exposure::is_eligible` / `sympson_hetter::is_eligible`).
            let eligible = crate::exposure::is_eligible(&exposure_method, item.sh_r_param);
            if !eligible {
                candidate_debug.push(CandidateDebug {
                    item_id: item.id.clone(),
                    eligible: false,
                    raw_score: None,
                });
                continue;
            }

            let fim = item_fim(theta, a, item.d_param, item.c_param);
            let raw_score =
                score_item(&selection_method, &ctx, &fim, a, item.d_param, item.c_param);
            let score = raw_score;

            candidate_debug.push(CandidateDebug {
                item_id: item.id.clone(),
                eligible: true,
                raw_score: Some(raw_score),
            });

            if best.map_or(true, |(_, s)| score > s) {
                best = Some((idx, score));
            }
        }

        let debug = SelectionDebug {
            method: settings.selection_method.clone(),
            exposure_method: settings.exposure_method.clone(),
            candidates: candidate_debug,
            winner_item_id: best.map(|(idx, _)| candidates[idx].0.id.clone()),
            winner_score: best.map(|(_, s)| s),
        };

        Ok((best, debug))
    }

    /// Re-estimate theta after a new response has been recorded.
    pub fn estimate_theta(
        settings: &EngineSettings,
        item_a_vecs: &[DVector<f64>],
        item_d: &[f64],
        item_c: &[f64],
        responses: &[u8],
        k: usize,
    ) -> EngineResult<(DVector<f64>, EstimationDebug)> {
        let method: EstimationMethod = settings.estimation_method.parse().map_err(|_| {
            EngineError::InvalidConfig(format!(
                "Unknown estimation method: {}",
                settings.estimation_method
            ))
        })?;

        // Defensive: the host application rejects an infeasible EAP grid at write time, but
        // this covers presets that predate that check and playground overrides, so a
        // session can never block on a multi-million-point quadrature sum.
        if method == EstimationMethod::Eap {
            estimation::eap::check_feasible(k, settings.eap_quadrature_points as usize)
                .map_err(EngineError::InvalidRequest)?;
        }

        let prior_mean = DVector::from_vec(settings.prior_mean.clone());
        let prior_cov_inv = DMatrix::from_diagonal(&DVector::from_vec(
            settings.prior_cov_diag.iter().map(|v| 1.0 / v).collect(),
        ));

        let input = EstimationInput {
            item_a_vecs,
            item_d,
            item_c,
            responses,
            prior_mean: &prior_mean,
            prior_cov_inv: &prior_cov_inv,
            eap_quad_pts: settings.eap_quadrature_points as usize,
            k,
        };

        Ok(estimation::estimate(&method, &input))
    }

    /// Per-dimension SE using the formula appropriate to `settings.estimation_method`:
    ///
    /// - **MLE**: `sqrt(diag(inv(cum_fim)))` — the standard asymptotic SE (Cramér-Rao).
    /// - **MAP**: `sqrt(diag(inv(cum_fim + prior_cov_inv)))` — same as MLE but folding in
    ///   the prior's curvature, matching the Hessian MAP's own Newton-Raphson already
    ///   converges on (see `estimation::map::estimate`). Without the prior term here, SE
    ///   would be biased too large and undefined (`None`) before any item is answered,
    ///   even though the posterior (= prior at n=0) has perfectly well-defined variance.
    /// - **EAP**: the posterior SD produced directly by the quadrature
    ///   (`EstimationDebug.posterior_se`) — EAP's whole point is not needing an
    ///   asymptotic-normal approximation, so reusing the MLE/MAP FIM-based formula for it
    ///   would throw that away. `eap_posterior_se` is `None` only when no fresh EAP
    ///   estimate is on hand (e.g. reporting SE without re-estimating); in that case the
    ///   MAP-style formula is used as a Laplace-approximation stand-in, which is exact at
    ///   n=0 (posterior == prior) and an approximation otherwise.
    pub fn compute_se(
        settings: &EngineSettings,
        cum_fim: &DMatrix<f64>,
        k: usize,
        eap_posterior_se: Option<&[Option<f64>]>,
    ) -> EngineResult<Vec<Option<f64>>> {
        let method: EstimationMethod = settings.estimation_method.parse().map_err(|_| {
            EngineError::InvalidConfig(format!(
                "Unknown estimation method: {}",
                settings.estimation_method
            ))
        })?;

        let map_style_se = || {
            let prior_cov_inv = DMatrix::from_diagonal(&DVector::from_vec(
                settings.prior_cov_diag.iter().map(|v| 1.0 / v).collect(),
            ));
            se_vector(&(cum_fim + prior_cov_inv), k)
        };

        Ok(match method {
            EstimationMethod::Mle => se_vector(cum_fim, k),
            EstimationMethod::Map => map_style_se(),
            EstimationMethod::Eap => match eap_posterior_se {
                Some(se) => se.to_vec(),
                None => map_style_se(),
            },
        })
    }

    /// Check whether the session should stop.
    pub fn should_stop(
        settings: &EngineSettings,
        n: usize,
        se: &[Option<f64>],
        theta_now: &DVector<f64>,
        theta_prev: &DVector<f64>,
    ) -> EngineResult<(bool, &'static str, StoppingDebug)> {
        let rule: StoppingRule = settings.stopping_rule.parse().map_err(|_| {
            EngineError::InvalidConfig(format!("Unknown stopping rule: {}", settings.stopping_rule))
        })?;

        let input = StoppingInput {
            n,
            n_min: settings.n_min as usize,
            n_max: settings.n_max as usize,
            se_threshold: settings.se_threshold,
            delta: settings.convergence_delta,
            se,
            theta_now,
            theta_prev,
        };

        let (decision, reason) = should_stop(&rule, &input);
        let debug = StoppingDebug {
            rule: settings.stopping_rule.clone(),
            n,
            n_min: settings.n_min as usize,
            n_max: settings.n_max as usize,
            se: se.to_vec(),
            se_threshold: settings.se_threshold,
            delta_norm: (theta_now - theta_prev).norm(),
            convergence_delta: settings.convergence_delta,
            decision,
            reason: reason.to_string(),
        };

        Ok((decision, reason, debug))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> EngineSettings {
        EngineSettings {
            selection_method: "d_optimal".to_string(),
            estimation_method: "mle".to_string(),
            stopping_rule: "fixed_length".to_string(),
            exposure_method: "none".to_string(),
            n_min: 1,
            n_max: 10,
            se_threshold: 0.3,
            convergence_delta: 0.01,
            prior_mean: vec![0.0],
            prior_cov_diag: vec![1.0],
            eap_quadrature_points: 21,
        }
    }

    fn item(id: &str, sh_r_param: f64, is_active: bool) -> EngineItem {
        EngineItem {
            id: id.to_string(),
            a_params: vec![1.0],
            d_param: 0.0,
            c_param: 0.0,
            sh_r_param,
            is_active,
        }
    }

    // ── select_next_item ────────────────────────────────────────────────

    #[test]
    fn select_next_item_picks_the_highest_scoring_eligible_candidate() {
        let s = settings(); // d_optimal
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let weak = item("weak", 1.0, true);
        let strong = item("strong", 1.0, true);
        let candidates = vec![
            (&weak, DVector::from_vec(vec![0.5])),
            (&strong, DVector::from_vec(vec![2.0])),
        ];
        let (result, debug) =
            McatEngine::select_next_item(&s, &candidates, &[], &theta, &cum_fim, 1).unwrap();
        let (idx, _score) = result.expect("an eligible item should be found");
        assert_eq!(candidates[idx].0.id, "strong");
        assert_eq!(debug.winner_item_id.as_deref(), Some("strong"));
    }

    #[test]
    fn select_next_item_skips_already_administered_items() {
        let s = settings();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let strong = item("strong", 1.0, true);
        let weak = item("weak", 1.0, true);
        let candidates = vec![
            (&strong, DVector::from_vec(vec![2.0])),
            (&weak, DVector::from_vec(vec![0.5])),
        ];
        let administered = vec!["strong".to_string()];
        let (result, _) =
            McatEngine::select_next_item(&s, &candidates, &administered, &theta, &cum_fim, 1)
                .unwrap();
        let (idx, _) = result.unwrap();
        assert_eq!(candidates[idx].0.id, "weak");
    }

    #[test]
    fn select_next_item_skips_inactive_items() {
        let s = settings();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let inactive_strong = item("strong", 1.0, false);
        let active_weak = item("weak", 1.0, true);
        let candidates = vec![
            (&inactive_strong, DVector::from_vec(vec![2.0])),
            (&active_weak, DVector::from_vec(vec![0.5])),
        ];
        let (result, _) =
            McatEngine::select_next_item(&s, &candidates, &[], &theta, &cum_fim, 1).unwrap();
        let (idx, _) = result.unwrap();
        assert_eq!(candidates[idx].0.id, "weak");
    }

    #[test]
    fn select_next_item_returns_none_when_all_candidates_are_ineligible() {
        let mut s = settings();
        s.exposure_method = "sympson_hetter".to_string();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let never = item("never", 0.0, true); // sh_r_param=0.0 -> never eligible
        let candidates = vec![(&never, DVector::from_vec(vec![1.0]))];
        let (result, debug) =
            McatEngine::select_next_item(&s, &candidates, &[], &theta, &cum_fim, 1).unwrap();
        assert!(result.is_none());
        assert_eq!(debug.candidates.len(), 1);
        assert!(!debug.candidates[0].eligible);
    }

    #[test]
    fn select_next_item_rejects_unknown_selection_method() {
        let mut s = settings();
        s.selection_method = "bogus".to_string();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let it = item("a", 1.0, true);
        let candidates = vec![(&it, DVector::from_vec(vec![1.0]))];
        let err =
            McatEngine::select_next_item(&s, &candidates, &[], &theta, &cum_fim, 1).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn select_next_item_rejects_unknown_exposure_method() {
        let mut s = settings();
        s.exposure_method = "bogus".to_string();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let it = item("a", 1.0, true);
        let candidates = vec![(&it, DVector::from_vec(vec![1.0]))];
        let err =
            McatEngine::select_next_item(&s, &candidates, &[], &theta, &cum_fim, 1).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)));
    }

    // ── estimate_theta ──────────────────────────────────────────────────

    #[test]
    fn estimate_theta_dispatches_to_the_configured_method() {
        let s = settings(); // mle
        let a = DVector::from_vec(vec![1.0]);
        let (_, debug) = McatEngine::estimate_theta(&s, &[a], &[0.0], &[0.0], &[1], 1).unwrap();
        assert_eq!(debug.method, "mle");
    }

    #[test]
    fn estimate_theta_rejects_unknown_estimation_method() {
        let mut s = settings();
        s.estimation_method = "bogus".to_string();
        let a = DVector::from_vec(vec![1.0]);
        let err = McatEngine::estimate_theta(&s, &[a], &[0.0], &[0.0], &[1], 1).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)));
    }

    #[test]
    fn estimate_theta_rejects_an_infeasible_eap_grid() {
        let mut s = settings();
        s.estimation_method = "eap".to_string();
        s.eap_quadrature_points = 1_000_000; // pts^k = 1_000_000 > MAX_GRID_POINTS at k=1
        let a = DVector::from_vec(vec![1.0]);
        let err = McatEngine::estimate_theta(&s, &[a], &[0.0], &[0.0], &[1], 1).unwrap_err();
        assert!(matches!(err, EngineError::InvalidRequest(_)));
    }

    // ── compute_se ──────────────────────────────────────────────────────

    #[test]
    fn compute_se_mle_uses_cum_fim_alone() {
        let s = settings(); // mle
        let cum_fim = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0]));
        let se = McatEngine::compute_se(&s, &cum_fim, 1, None).unwrap();
        // sqrt(diag(inv(diag(4)))) = sqrt(0.25) = 0.5
        assert!((se[0].unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_se_map_folds_in_prior_precision() {
        let mut s = settings();
        s.estimation_method = "map".to_string();
        s.prior_cov_diag = vec![1.0]; // prior_cov_inv = 1.0
        let cum_fim = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0]));
        let se = McatEngine::compute_se(&s, &cum_fim, 1, None).unwrap();
        // sqrt(diag(inv(diag(3 + 1)))) = sqrt(0.25) = 0.5
        assert!((se[0].unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_se_eap_uses_supplied_posterior_se_when_present() {
        let mut s = settings();
        s.estimation_method = "eap".to_string();
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let posterior = [Some(0.42)];
        let se = McatEngine::compute_se(&s, &cum_fim, 1, Some(&posterior)).unwrap();
        assert_eq!(se[0], Some(0.42));
    }

    #[test]
    fn compute_se_eap_falls_back_to_map_style_without_a_posterior() {
        let mut s = settings();
        s.estimation_method = "eap".to_string();
        s.prior_cov_diag = vec![1.0];
        let cum_fim = DMatrix::from_diagonal(&DVector::from_vec(vec![3.0]));
        let se = McatEngine::compute_se(&s, &cum_fim, 1, None).unwrap();
        assert!((se[0].unwrap() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn compute_se_rejects_unknown_estimation_method() {
        let mut s = settings();
        s.estimation_method = "bogus".to_string();
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let err = McatEngine::compute_se(&s, &cum_fim, 1, None).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)));
    }

    // ── should_stop ─────────────────────────────────────────────────────

    #[test]
    fn should_stop_dispatches_and_reports_the_rules_reason() {
        let s = settings(); // fixed_length, n_max = 10
        let theta = DVector::from_vec(vec![0.0]);
        let (stop, reason, debug) =
            McatEngine::should_stop(&s, 10, &[Some(0.1)], &theta, &theta).unwrap();
        assert!(stop);
        assert_eq!(reason, "fixed_length");
        assert_eq!(debug.rule, "fixed_length");
        assert_eq!(debug.n, 10);
    }

    #[test]
    fn should_stop_rejects_unknown_rule() {
        let mut s = settings();
        s.stopping_rule = "bogus".to_string();
        let theta = DVector::from_vec(vec![0.0]);
        let err = McatEngine::should_stop(&s, 1, &[], &theta, &theta).unwrap_err();
        assert!(matches!(err, EngineError::InvalidConfig(_)));
    }
}
