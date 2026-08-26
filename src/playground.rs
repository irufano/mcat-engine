#![allow(dead_code)]

//! Pure, DB-free orchestration for the algorithm playground: lets callers
//! run the same `McatEngine` functions production sessions use, but with
//! item selection / ability estimation / stopping independently toggled on
//! or off, and with responses simulated from a caller-supplied "true theta"
//! instead of coming from a real examinee. Never touches the database —
//! callers pass in already-loaded items and get back updated in-memory
//! state to feed into the next call.
//!
//! Building an `EngineSettings` from request input (e.g. a playground
//! "settings override" body merged with defaults) is the host application's
//! job, not this crate's — see `mcat-api`'s `modules::playground::service::build_settings`,
//! which constructs its own DB-shaped settings type and converts it to
//! `EngineSettings` right before calling [`step`].

use std::collections::HashMap;

use nalgebra::{DMatrix, DVector};
use rand::Rng;
use serde::Deserialize;

use crate::error::{EngineError, EngineResult};
use crate::{
    debug::{EstimationDebug, SelectionDebug, StoppingDebug},
    engine::McatEngine,
    mirt::{item_fim, probability},
    types::{EngineItem, EngineSettings},
};

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(default)]
pub struct StageToggles {
    pub selection: bool,
    pub estimation: bool,
    pub stopping: bool,
}

impl Default for StageToggles {
    fn default() -> Self {
        Self {
            selection: true,
            estimation: true,
            stopping: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, serde::Serialize)]
pub struct AdministeredItem {
    pub item_id: String,
    /// `None` when this item was administered during a step where
    /// `stages.estimation` was disabled — it still counts for selection
    /// exclusion (never re-administered) but is excluded from the
    /// estimation refit below, since it was never actually scored.
    pub response: Option<u8>,
}

/// Where the response for a newly-administered item should come from, when
/// ability estimation is enabled.
pub enum ResponseSource<'a> {
    /// Draw a response probabilistically from the item's own M3PL params at
    /// this "true theta" — the standard way to test whether an estimator
    /// recovers a known ability.
    Simulate { true_theta: &'a DVector<f64> },
    /// Caller supplies the response directly (deterministic testing).
    Fixed(u8),
    /// No response source given — an error if estimation is enabled.
    None,
}

#[derive(Debug, Default, Deserialize)]
pub struct PlaygroundSettingsInput {
    pub selection_method: Option<String>,
    pub estimation_method: Option<String>,
    pub stopping_rule: Option<String>,
    pub exposure_method: Option<String>,
    pub n_min: Option<i32>,
    pub n_max: Option<i32>,
    pub se_threshold: Option<f64>,
    pub convergence_delta: Option<f64>,
    pub prior_mean: Option<Vec<f64>>,
    pub prior_cov_diag: Option<Vec<f64>>,
    pub eap_quadrature_points: Option<i32>,
    /// Subset (and order) of the bank's dimension universe to actually test — mirrors
    /// `TestSettings.tested_dimensions` in production. `None` defaults to the bank's
    /// full universe (today's playground behavior). Inert here: this struct is just
    /// override data for the host application's `build_settings`; no function in this
    /// crate reads it directly — see `step`'s `tested_a` parameter for how masking
    /// actually happens once this has been resolved into a concrete dimension list.
    pub tested_dimensions: Option<Vec<String>>,
}

/// Simulate a dichotomous response from a "true theta" using the item's own
/// M3PL parameters.
pub fn simulate_response(
    true_theta: &DVector<f64>,
    a: &DVector<f64>,
    d: f64,
    c: f64,
    rng: &mut impl Rng,
) -> u8 {
    let p = probability(true_theta, a, d, c).clamp(0.0, 1.0);
    u8::from(rng.gen_bool(p))
}

pub struct StepOutcome {
    /// Index into the `items` slice of the item administered this step;
    /// `None` when selection was enabled but the bank/pool was exhausted.
    pub item_idx: Option<usize>,
    pub selection_score: Option<f64>,
    pub selection_debug: Option<SelectionDebug>,
    pub response: Option<u8>,
    /// Equal to the input `theta` when estimation is disabled or no
    /// response was produced.
    pub theta_after: DVector<f64>,
    pub estimation_debug: Option<EstimationDebug>,
    pub cum_fim_after: DMatrix<f64>,
    /// Per-dimension SE via the method-appropriate formula (`McatEngine::compute_se`),
    /// kept alongside `cum_fim_after` since EAP's SE can't be reconstructed from the FIM
    /// alone. Equal to the input state's SE when estimation was disabled this step.
    pub se_after: Vec<Option<f64>>,
    pub stop_decision: Option<bool>,
    pub stop_reason: Option<&'static str>,
    pub stopping_debug: Option<StoppingDebug>,
}

fn find_item<'a>(items: &'a [EngineItem], item_id: &str) -> EngineResult<&'a EngineItem> {
    items
        .iter()
        .find(|it| it.id == item_id)
        .ok_or_else(|| EngineError::InvalidRequest(format!("Item '{item_id}' not found in bank")))
}

/// Run a single playground step: optionally select an item, optionally
/// produce/estimate a response, optionally evaluate the stopping rule —
/// exactly mirroring the production `McatEngine` calls used by
/// `session_handler`, just without any database access.
///
/// `tested_a` maps item id → that item's a-vector already projected into the caller's
/// tested-dimension order (built by `mcat-api`'s `playground::service::build_projected_a`
/// via `mcat_engine::dimensions::project_item`, once per request — this crate stays
/// dimension-name-free and never computes projections itself, only consumes them). An
/// item id absent from this map doesn't project onto the current tested-dimension set:
/// it's silently excluded from selection candidates (mirrors production's "excluded
/// entirely" eligibility rule), and it's an `EngineError::InvalidRequest` if it turns up
/// in `administered` or as `item_override` — `administered` is caller-resent, in-flight
/// state that can legitimately go stale if `tested_dimensions` changed between calls.
#[allow(clippy::too_many_arguments)]
pub fn step(
    items: &[EngineItem],
    settings: &EngineSettings,
    stages: &StageToggles,
    theta: &DVector<f64>,
    cum_fim: &DMatrix<f64>,
    administered: &[AdministeredItem],
    tested_a: &HashMap<String, DVector<f64>>,
    k: usize,
    item_override: Option<&str>,
    response_source: ResponseSource,
    rng: &mut impl Rng,
) -> EngineResult<StepOutcome> {
    let administered_ids: Vec<String> = administered.iter().map(|a| a.item_id.clone()).collect();

    // Items absent from `tested_a` measure a dimension outside the current
    // tested-dimension set and are excluded entirely — never truncated/zero-padded.
    let candidates: Vec<(&EngineItem, DVector<f64>)> = items
        .iter()
        .filter_map(|it| tested_a.get(&it.id).map(|a| (it, a.clone())))
        .collect();

    let (item_idx, selection_score, selection_debug) = if stages.selection {
        let (best, debug) = McatEngine::select_next_item(
            settings,
            &candidates,
            &administered_ids,
            theta,
            cum_fim,
            k,
        )?;
        match best {
            Some((cand_idx, score)) => {
                // `cand_idx` indexes into `candidates`, which — now that it's built via
                // `filter_map` — is no longer 1:1 with `items` whenever any item was
                // excluded by `tested_a`. Re-resolve the winning candidate's position in
                // the original `items` slice by id, since `StepOutcome.item_idx` is
                // documented (and relied on by callers, e.g. `mcat-api`'s
                // `shape_step_response`) as an index into `items`, not `candidates`.
                let item_id = &candidates[cand_idx].0.id;
                let idx = items
                    .iter()
                    .position(|it| &it.id == item_id)
                    .expect("a selected candidate always originates from `items`");
                (Some(idx), Some(score), Some(debug))
            }
            None => {
                // Bank exhausted — nothing was administered this step, so no fresh EAP
                // posterior SD is available either; `compute_se` falls back to the
                // MAP-style formula in that case.
                return Ok(StepOutcome {
                    item_idx: None,
                    selection_score: None,
                    selection_debug: Some(debug),
                    response: None,
                    theta_after: theta.clone(),
                    estimation_debug: None,
                    cum_fim_after: cum_fim.clone(),
                    se_after: McatEngine::compute_se(settings, cum_fim, k, None)?,
                    stop_decision: None,
                    stop_reason: Some("bank_exhausted"),
                    stopping_debug: None,
                });
            }
        }
    } else {
        let override_id = item_override.ok_or_else(|| {
            EngineError::InvalidRequest(
                "item_id_override is required when stages.selection is false".into(),
            )
        })?;
        let idx = items
            .iter()
            .position(|it| it.id == override_id && it.is_active)
            .ok_or_else(|| {
                EngineError::InvalidRequest(format!(
                    "Override item '{override_id}' not found or inactive in bank"
                ))
            })?;
        (Some(idx), None, None)
    };

    let item = &items[item_idx.expect("item_idx resolved above")];
    // Only reachable via `item_id_override` — normal selection only ever picks from
    // `tested_a`-backed candidates above, so this can't fail on that path.
    let item_a = tested_a.get(&item.id).ok_or_else(|| {
        EngineError::InvalidRequest(format!(
            "Item '{}' does not project onto the current tested-dimension set",
            item.id
        ))
    })?;

    let response = if stages.estimation {
        match response_source {
            ResponseSource::Fixed(r) => Some(r),
            ResponseSource::Simulate { true_theta } => Some(simulate_response(
                true_theta,
                item_a,
                item.d_param,
                item.c_param,
                rng,
            )),
            ResponseSource::None => {
                return Err(EngineError::InvalidRequest(
                    "estimation is enabled but no response source (true_theta or response_override) was provided".into(),
                ));
            }
        }
    } else {
        None
    };

    let (theta_after, estimation_debug) = if let Some(resp) = response {
        let mut all_a: Vec<DVector<f64>> = Vec::with_capacity(administered.len() + 1);
        let mut all_d: Vec<f64> = Vec::with_capacity(administered.len() + 1);
        let mut all_c: Vec<f64> = Vec::with_capacity(administered.len() + 1);
        let mut all_r: Vec<u8> = Vec::with_capacity(administered.len() + 1);

        for prior in administered {
            let Some(prior_response) = prior.response else {
                continue;
            };
            let prior_item = find_item(items, &prior.item_id)?;
            let prior_a = tested_a.get(&prior.item_id).ok_or_else(|| {
                EngineError::InvalidRequest(format!(
                    "Previously administered item '{}' no longer projects onto the current \
                     tested-dimension set — tested_dimensions must stay consistent across \
                     playground calls that resend the same in-flight state",
                    prior.item_id
                ))
            })?;
            all_a.push(prior_a.clone());
            all_d.push(prior_item.d_param);
            all_c.push(prior_item.c_param);
            all_r.push(prior_response);
        }
        all_a.push(item_a.clone());
        all_d.push(item.d_param);
        all_c.push(item.c_param);
        all_r.push(resp);

        let (theta_after, debug) =
            McatEngine::estimate_theta(settings, &all_a, &all_d, &all_c, &all_r, k)?;
        (theta_after, Some(debug))
    } else {
        (theta.clone(), None)
    };

    let cum_fim_after = cum_fim + item_fim(&theta_after, item_a, item.d_param, item.c_param);
    // Method-appropriate SE, exactly mirroring session_handler::submit_response.
    let se_after = McatEngine::compute_se(
        settings,
        &cum_fim_after,
        k,
        estimation_debug
            .as_ref()
            .and_then(|d| d.posterior_se.as_deref()),
    )?;

    let (stop_decision, stop_reason, stopping_debug) = if stages.stopping {
        let n = administered.len() + 1;
        let (decision, reason, debug) =
            McatEngine::should_stop(settings, n, &se_after, &theta_after, theta)?;
        (Some(decision), Some(reason), Some(debug))
    } else {
        (None, None, None)
    };

    Ok(StepOutcome {
        item_idx,
        selection_score,
        selection_debug,
        response,
        theta_after,
        estimation_debug,
        cum_fim_after,
        se_after,
        stop_decision,
        stop_reason,
        stopping_debug,
    })
}

/// Outcome of simulating one examinee to completion via [`run_examinee`]. Deliberately
/// lean — no per-step `SelectionDebug`/`EstimationDebug`/`StoppingDebug` — see that
/// function's doc comment for why.
pub struct ExamineeRunOutcome {
    pub theta_hat: DVector<f64>,
    pub se_vector: Vec<Option<f64>>,
    pub n_administered: usize,
    pub administered_item_ids: Vec<String>,
    /// `None` when `max_steps` was exhausted without the stopping rule (or bank
    /// exhaustion) ever firing.
    pub stop_reason: Option<String>,
}

/// Run one simulated examinee to completion: repeatedly calls [`step`] with
/// `stages = {selection: true, estimation: true, stopping: true}` and a response drawn
/// from `true_theta`, stopping when the stopping rule fires, the bank is exhausted, or
/// `max_steps` is reached.
///
/// Exists as a DB-free, allocation-light primitive for callers that need to run *many*
/// simulated examinees (e.g. a Monte-Carlo calibration study) without paying for or
/// carrying forward each step's full debug trace — `step()` still computes those debug
/// structs internally (nothing about the underlying computation changes), this function
/// just declines to keep them, since retaining `SelectionDebug`/`EstimationDebug` for
/// every step of every examinee in an N-examinee batch would multiply an already
/// non-trivial per-step payload by N for no caller benefit. Callers that need the full
/// per-step trace for a single examinee should call `step()` directly in their own loop
/// instead (see `mcat-api`'s `modules::playground::service::run`), not this function.
///
/// `tested_a` is `step`'s own tested-dimension-projection map, reused unchanged for
/// every step of this one examinee — since every administered item was itself drawn
/// from candidates `step` already filtered through this same map, its "administered
/// item missing from tested_a" error case is structurally unreachable here.
#[allow(clippy::too_many_arguments)]
pub fn run_examinee(
    items: &[EngineItem],
    settings: &EngineSettings,
    initial_theta: &DVector<f64>,
    true_theta: &DVector<f64>,
    tested_a: &HashMap<String, DVector<f64>>,
    k: usize,
    max_steps: usize,
    rng: &mut impl Rng,
) -> EngineResult<ExamineeRunOutcome> {
    let stages = StageToggles::default();
    let mut theta = initial_theta.clone();
    let mut cum_fim = DMatrix::<f64>::zeros(k, k);
    let mut se_vector = McatEngine::compute_se(settings, &cum_fim, k, None)?;
    let mut administered: Vec<AdministeredItem> = Vec::new();
    let mut stop_reason: Option<String> = None;

    for _ in 0..max_steps {
        let outcome = step(
            items,
            settings,
            &stages,
            &theta,
            &cum_fim,
            &administered,
            tested_a,
            k,
            None,
            ResponseSource::Simulate { true_theta },
            rng,
        )?;

        let Some(idx) = outcome.item_idx else {
            stop_reason = Some("bank_exhausted".to_string());
            break;
        };

        theta = outcome.theta_after;
        cum_fim = outcome.cum_fim_after;
        se_vector = outcome.se_after;
        administered.push(AdministeredItem {
            item_id: items[idx].id.clone(),
            response: outcome.response,
        });

        if outcome.stop_decision.unwrap_or(false) {
            stop_reason = outcome.stop_reason.map(str::to_string);
            break;
        }
    }

    Ok(ExamineeRunOutcome {
        theta_hat: theta,
        se_vector,
        n_administered: administered.len(),
        administered_item_ids: administered.into_iter().map(|a| a.item_id).collect(),
        stop_reason,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn simulate_response_high_true_theta_easy_item_is_almost_always_correct() {
        // theta=6, a=1, d=0 → p = sigmoid(6) ≈ 0.9975; over 500 draws the
        // binomial std dev is ~1.1, so 480+ is safely within range.
        let true_theta = DVector::from_vec(vec![6.0]);
        let a = DVector::from_vec(vec![1.0]);
        let mut rng = StdRng::seed_from_u64(42);

        let hits: u32 = (0..500)
            .map(|_| simulate_response(&true_theta, &a, 0.0, 0.0, &mut rng) as u32)
            .sum();

        assert!(
            hits > 480,
            "expected near-certain correct responses, got {hits}/500"
        );
    }

    fn settings(stopping_rule: &str, n_min: i32, n_max: i32, se_threshold: f64) -> EngineSettings {
        EngineSettings {
            selection_method: "d_optimal".to_string(),
            estimation_method: "map".to_string(),
            stopping_rule: stopping_rule.to_string(),
            exposure_method: "none".to_string(),
            n_min,
            n_max,
            se_threshold,
            convergence_delta: 0.01,
            prior_mean: vec![0.0],
            prior_cov_diag: vec![1.0],
            eap_quadrature_points: 21,
        }
    }

    /// A bank of well-discriminating items spread across a wide difficulty range, so
    /// `d_optimal` selection always has a well-targeted item available and the bank
    /// doesn't run out within a handful of steps.
    fn wide_bank(n: usize) -> Vec<EngineItem> {
        (0..n)
            .map(|i| {
                let d = -4.0 + 8.0 * (i as f64) / ((n - 1) as f64);
                EngineItem {
                    id: format!("item-{i}"),
                    a_params: vec![1.5],
                    d_param: d,
                    c_param: 0.0,
                    sh_r_param: 1.0,
                    is_active: true,
                }
            })
            .collect()
    }

    /// Identity `tested_a`: every item's own `a_params`, unprojected — models the
    /// "tested_dimensions omitted" default (tested set == bank universe) that all the
    /// pre-masking tests below exercise, so they keep passing unchanged now that `step`/
    /// `run_examinee` require a `tested_a` map instead of reading `a_params` directly.
    fn identity_tested_a(items: &[EngineItem]) -> HashMap<String, DVector<f64>> {
        items
            .iter()
            .map(|it| (it.id.clone(), DVector::from_vec(it.a_params.clone())))
            .collect()
    }

    #[test]
    fn run_examinee_recovers_a_known_true_theta_given_enough_items() {
        let items = wide_bank(41);
        let s = settings("fixed_length", 20, 20, 0.3);
        let true_theta = DVector::from_vec(vec![1.5]);
        let initial_theta = DVector::from_vec(vec![0.0]);
        let mut rng = StdRng::seed_from_u64(42);

        let tested_a = identity_tested_a(&items);
        let outcome = run_examinee(
            &items,
            &s,
            &initial_theta,
            &true_theta,
            &tested_a,
            1,
            20,
            &mut rng,
        )
        .unwrap();

        assert_eq!(outcome.n_administered, 20);
        assert_eq!(outcome.administered_item_ids.len(), 20);
        assert_eq!(outcome.stop_reason.as_deref(), Some("fixed_length"));
        assert!(
            (outcome.theta_hat[0] - 1.5).abs() < 0.5,
            "expected theta_hat near 1.5 after 20 items, got {}",
            outcome.theta_hat[0]
        );
    }

    #[test]
    fn run_examinee_reports_bank_exhausted_when_items_run_out() {
        let items = wide_bank(2);
        let s = settings("fixed_length", 1, 10, 0.3);
        let true_theta = DVector::from_vec(vec![0.0]);
        let initial_theta = DVector::from_vec(vec![0.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let tested_a = identity_tested_a(&items);
        let outcome = run_examinee(
            &items,
            &s,
            &initial_theta,
            &true_theta,
            &tested_a,
            1,
            10,
            &mut rng,
        )
        .unwrap();

        assert_eq!(outcome.n_administered, 2);
        assert_eq!(outcome.stop_reason.as_deref(), Some("bank_exhausted"));
    }

    #[test]
    fn run_examinee_returns_none_stop_reason_when_max_steps_exhausted_without_stopping() {
        let items = wide_bank(41);
        // se_threshold effectively unreachable, n_max far beyond max_steps — nothing
        // should trigger the stopping rule before the loop's own max_steps cap does.
        let s = settings("se_threshold", 1, 100, 0.0001);
        let true_theta = DVector::from_vec(vec![0.0]);
        let initial_theta = DVector::from_vec(vec![0.0]);
        let mut rng = StdRng::seed_from_u64(3);

        let tested_a = identity_tested_a(&items);
        let outcome = run_examinee(
            &items,
            &s,
            &initial_theta,
            &true_theta,
            &tested_a,
            1,
            3,
            &mut rng,
        )
        .unwrap();

        assert_eq!(outcome.n_administered, 3);
        assert_eq!(outcome.stop_reason, None);
    }

    #[test]
    fn step_excludes_item_absent_from_tested_a_map() {
        // "excluded" (d=0) is the perfect d_optimal target at theta=0 — it would win
        // selection if eligible. Only "included" (a worse-targeted d=3) is in `tested_a`,
        // proving exclusion isn't accidental (a coincidentally-worse score), it's absolute.
        let excluded = EngineItem {
            id: "excluded".to_string(),
            a_params: vec![1.5],
            d_param: 0.0,
            c_param: 0.0,
            sh_r_param: 1.0,
            is_active: true,
        };
        let included = EngineItem {
            id: "included".to_string(),
            a_params: vec![1.5],
            d_param: 3.0,
            c_param: 0.0,
            sh_r_param: 1.0,
            is_active: true,
        };
        let items = vec![excluded, included];
        let s = settings("fixed_length", 1, 10, 0.3);
        let tested_a: HashMap<String, DVector<f64>> =
            [("included".to_string(), DVector::from_vec(vec![1.5]))]
                .into_iter()
                .collect();
        let stages = StageToggles::default();
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let mut rng = StdRng::seed_from_u64(1);

        let outcome = step(
            &items,
            &s,
            &stages,
            &theta,
            &cum_fim,
            &[],
            &tested_a,
            1,
            None,
            ResponseSource::Simulate { true_theta: &theta },
            &mut rng,
        )
        .unwrap();

        let idx = outcome.item_idx.expect("the eligible item should be found");
        assert_eq!(items[idx].id, "included");
    }

    #[test]
    fn step_returns_invalid_request_when_historical_administered_item_missing_from_tested_a() {
        let items = wide_bank(3);
        let mut tested_a = identity_tested_a(&items);
        tested_a.remove("item-0"); // simulates tested_dimensions changing since it was administered
        let administered = vec![AdministeredItem {
            item_id: "item-0".to_string(),
            response: Some(1),
        }];
        let s = settings("fixed_length", 1, 10, 0.3);
        let stages = StageToggles {
            selection: false,
            estimation: true,
            stopping: false,
        };
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let mut rng = StdRng::seed_from_u64(1);

        let result = step(
            &items,
            &s,
            &stages,
            &theta,
            &cum_fim,
            &administered,
            &tested_a,
            1,
            Some("item-1"), // present in tested_a — isolates the failure to the historical item
            ResponseSource::Fixed(1),
            &mut rng,
        );

        assert!(matches!(result, Err(EngineError::InvalidRequest(_))));
    }

    #[test]
    fn step_uses_projected_a_not_raw_item_a_params() {
        let items = vec![EngineItem {
            id: "item-0".to_string(),
            // Raw "universe order" a_params, deliberately different from tested_a below —
            // if `step` ever fell back to reading this instead of the projected vector,
            // the FIM assertion below would fail.
            a_params: vec![1.0],
            d_param: 0.0,
            c_param: 0.0,
            sh_r_param: 1.0,
            is_active: true,
        }];
        let s = settings("fixed_length", 1, 10, 0.3);
        let projected_a = DVector::from_vec(vec![2.0]);
        let tested_a: HashMap<String, DVector<f64>> = [(items[0].id.clone(), projected_a.clone())]
            .into_iter()
            .collect();
        let stages = StageToggles {
            selection: false,
            estimation: true,
            stopping: false,
        };
        let theta = DVector::from_vec(vec![0.0]);
        let cum_fim = DMatrix::<f64>::zeros(1, 1);
        let mut rng = StdRng::seed_from_u64(1);

        let outcome = step(
            &items,
            &s,
            &stages,
            &theta,
            &cum_fim,
            &[],
            &tested_a,
            1,
            Some("item-0"),
            ResponseSource::Fixed(1),
            &mut rng,
        )
        .unwrap();

        let expected_fim = item_fim(&outcome.theta_after, &projected_a, 0.0, 0.0);
        assert_eq!(outcome.cum_fim_after, expected_fim);
    }

    #[test]
    fn simulate_response_low_true_theta_hard_item_is_almost_always_incorrect() {
        let true_theta = DVector::from_vec(vec![-6.0]);
        let a = DVector::from_vec(vec![1.0]);
        let mut rng = StdRng::seed_from_u64(7);

        let hits: u32 = (0..500)
            .map(|_| simulate_response(&true_theta, &a, 0.0, 0.0, &mut rng) as u32)
            .sum();

        assert!(
            hits < 20,
            "expected near-certain incorrect responses, got {hits}/500"
        );
    }
}
