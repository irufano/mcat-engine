#![allow(dead_code)]

//! Item-eligibility masking + a-vector projection for heterogeneous item
//! banks (see migrations 003/005/007).
//!
//! A bank's dimension *universe* (`master_item_bank_dimensions`) is the full
//! ordered set of dimensions any item in the bank may draw from. Each item
//! measures some non-empty subset of it (`master_item_dimensions`); a
//! `homogeneous` bank is the degenerate case where every item measures all of
//! it. A test-settings preset's `tested_dimensions` (`master_test_settings`)
//! is a separate ordered subset — possibly reordered relative to the
//! universe — that determines the actual θ/a-vector order for any session
//! created from it.
//!
//! `project_item` is the single place that reconciles these three orderings
//! for one item against one preset: it decides eligibility (does this item
//! measure only dimensions the preset actually tests?) and, if eligible,
//! re-indexes the item's a-vector into the preset's tested order.

use std::collections::HashSet;

/// Projects one item's bank-universe-order `a_params` into `tested_order`.
///
/// Returns `None` when the item measures a dimension that is *not* in
/// `tested_order` — the item is fully ineligible for that tested-dimension
/// set. It is never included with the extra dimension silently dropped:
/// doing so would use a response function / Fisher information different
/// from the item's real one (see Reckase, *Multidimensional Item Response
/// Theory*, 2009, on simple-structure / between-item MIRT).
///
/// Returns `Some(vec)` sized `tested_order.len()` when eligible: position `i`
/// holds the item's discrimination on `tested_order[i]` if the item measures
/// that dimension, else `0.0` — a legitimate "not applicable" zero, not an
/// estimated near-zero value.
pub fn project_item(
    universe_order: &[String],
    item_measured: &HashSet<String>,
    item_a_universe: &[f64],
    tested_order: &[String],
) -> Option<Vec<f64>> {
    if item_measured.iter().any(|d| !tested_order.contains(d)) {
        return None;
    }

    let projected = tested_order
        .iter()
        .map(|dim| {
            if !item_measured.contains(dim) {
                return 0.0;
            }
            universe_order
                .iter()
                .position(|u| u == dim)
                .and_then(|idx| item_a_universe.get(idx).copied())
                .unwrap_or(0.0)
        })
        .collect();
    Some(projected)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn set(codes: &[&str]) -> HashSet<String> {
        codes.iter().map(|c| c.to_string()).collect()
    }

    fn strs(codes: &[&str]) -> Vec<String> {
        codes.iter().map(|c| c.to_string()).collect()
    }

    // Universe = [verbal, numeric, spatial, reasoning], tested = [spatial, verbal, numeric] —
    // mirrors the worked example from the design discussion.
    const UNIVERSE: [&str; 4] = ["verbal", "numeric", "spatial", "reasoning"];
    const TESTED: [&str; 3] = ["spatial", "verbal", "numeric"];

    #[test]
    fn one_dimensional_item_projects_into_single_slot() {
        // item measures only "verbal", a_universe = [1.6, 0, 0, 0]
        let got = project_item(
            &strs(&UNIVERSE),
            &set(&["verbal"]),
            &[1.6, 0.0, 0.0, 0.0],
            &strs(&TESTED),
        );
        assert_eq!(got, Some(vec![0.0, 1.6, 0.0])); // [spatial, verbal, numeric]
    }

    #[test]
    fn item_outside_universe_order_still_projects_correctly() {
        // item measures only "spatial" (universe index 2)
        let got = project_item(
            &strs(&UNIVERSE),
            &set(&["spatial"]),
            &[0.0, 0.0, 1.5, 0.0],
            &strs(&TESTED),
        );
        assert_eq!(got, Some(vec![1.5, 0.0, 0.0]));
    }

    #[test]
    fn two_dimensional_item_measuring_only_tested_dims_is_eligible() {
        // item measures verbal + spatial, a_universe = [1.2, 0, 1.3, 0]
        let got = project_item(
            &strs(&UNIVERSE),
            &set(&["verbal", "spatial"]),
            &[1.2, 0.0, 1.3, 0.0],
            &strs(&TESTED),
        );
        assert_eq!(got, Some(vec![1.3, 1.2, 0.0])); // [spatial, verbal, numeric]
    }

    #[test]
    fn item_touching_untested_dimension_is_excluded_entirely() {
        // item measures verbal + reasoning — reasoning is not in TESTED.
        let got = project_item(
            &strs(&UNIVERSE),
            &set(&["verbal", "reasoning"]),
            &[1.1, 0.0, 0.0, 1.4],
            &strs(&TESTED),
        );
        assert_eq!(got, None);
    }

    #[test]
    fn three_dimensional_item_fully_within_tested_set_is_eligible() {
        let got = project_item(
            &strs(&UNIVERSE),
            &set(&["verbal", "numeric", "spatial"]),
            &[1.0, 1.1, 1.0, 0.0],
            &strs(&TESTED),
        );
        assert_eq!(got, Some(vec![1.0, 1.0, 1.1])); // [spatial, verbal, numeric]
    }

    #[test]
    fn homogeneous_bank_full_universe_equals_tested_set_is_identity_reorder() {
        // A homogeneous bank where tested_dimensions == universe but reordered.
        let universe = strs(&["verbal", "numeric", "reasoning"]);
        let tested = strs(&["reasoning", "verbal", "numeric"]);
        let measured = set(&["verbal", "numeric", "reasoning"]);
        let got = project_item(&universe, &measured, &[1.5, 0.3, 0.4], &tested);
        assert_eq!(got, Some(vec![0.4, 1.5, 0.3]));
    }
}
