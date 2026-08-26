use crate::mirt::{EPSILON, sigmoid};
use nalgebra::DVector;

/// Coverage-probability constant in the Chang & Ying (1996) shrinking interval
/// δ_m = INTERVAL_C / sqrt(m + 1). Chang & Ying leave the exact value of C as a design
/// choice tied to a target coverage probability rather than prescribing one fixed number;
/// C = 3.0 (~3 asymptotic standard errors) is the value commonly used in follow-up
/// literature. See kl_information_notes.md, section "Konstanta C (lebar interval)".
pub const INTERVAL_C: f64 = 3.0;

/// Number of Simpson's-rule panels used to approximate the integral in `score()`. Must be even.
const QUADRATURE_PANELS: usize = 40;

/// P(correct | z) for the compensatory M3PL composite score z = a·θ + d.
fn p_of_z(z: f64, c: f64) -> f64 {
    c + (1.0 - c) * sigmoid(z)
}

/// Pointwise Kullback-Leibler information between composite scores z and ẑ for one item.
///
/// Original formula (Chang & Ying, 1996; reproduced verbatim as Eq. 9 in Han, 2018, p.6,
/// and as Eq. 6 in Sorrel, Barrada, de la Torre & Abad, 2020, p.4):
///
///   K_i(z ‖ ẑ) = P_i(ẑ)·ln[P_i(ẑ)/P_i(z)] + [1 − P_i(ẑ)]·ln{[1 − P_i(ẑ)] / [1 − P_i(z)]}
///
/// This is the exact Kullback-Leibler divergence between the two Bernoulli response
/// distributions Bernoulli(P_i(ẑ)) and Bernoulli(P_i(z)) — NOT a Taylor/Fisher-information
/// approximation.
fn pointwise_kl(p_hat: f64, p: f64) -> f64 {
    let p = p.clamp(EPSILON, 1.0 - EPSILON);
    let p_hat = p_hat.clamp(EPSILON, 1.0 - EPSILON);
    p_hat * (p_hat / p).ln() + (1.0 - p_hat) * ((1.0 - p_hat) / (1.0 - p)).ln()
}

/// Global Kullback-Leibler information item-selection criterion.
///
/// Original formula (Chang & Ying, 1996; reproduced verbatim as Eq. 10 in Han, 2018, p.6):
///
///   K̄_i(θ̂) = ∫_{ẑ−δ}^{ẑ+δ} K_i(z ‖ ẑ) dz ,   δ = INTERVAL_C / sqrt(administered + 1)
///
/// θ enters the item response function only through the compensatory MIRT composite score
/// z = a·θ + d (see mirt::probability), so the k-dimensional integral over θ that Chang &
/// Ying define for the unidimensional case collapses, without further assumption, to this
/// 1-D integral over z. Derivation: kl_information_notes.md, section
/// "Reduksi Multidimensional → Skalar".
///
/// `administered` is the number of items already administered in the session (m in Eq. 10);
/// the interval shrinks as more items are given, reflecting the increasing precision of θ̂.
pub fn score(theta: &DVector<f64>, a: &DVector<f64>, d: f64, c: f64, administered: usize) -> f64 {
    let z_hat = a.dot(theta) + d;
    let p_hat = p_of_z(z_hat, c);

    let delta = INTERVAL_C / ((administered + 1) as f64).sqrt();
    let lower = z_hat - delta;
    let step = (2.0 * delta) / QUADRATURE_PANELS as f64;

    let f = |i: usize| pointwise_kl(p_hat, p_of_z(lower + step * i as f64, c));

    // Composite Simpson's rule: ∫f ≈ (step/3)·[f0 + fn + 4·Σ(odd) + 2·Σ(even, interior)]
    let mut sum = f(0) + f(QUADRATURE_PANELS);
    for i in 1..QUADRATURE_PANELS {
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(i);
    }
    (step / 3.0) * sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointwise_kl_is_zero_at_identical_probabilities() {
        assert!((pointwise_kl(0.6, 0.6)).abs() < 1e-12);
    }

    #[test]
    fn pointwise_kl_is_nonnegative() {
        // Gibbs' inequality: KL divergence is always >= 0.
        for p_hat in [0.1, 0.3, 0.5, 0.7, 0.9] {
            for p in [0.05, 0.2, 0.4, 0.6, 0.8, 0.95] {
                assert!(pointwise_kl(p_hat, p) >= -1e-12, "p_hat={p_hat} p={p}");
            }
        }
    }

    #[test]
    fn score_is_positive_for_a_discriminating_item() {
        let theta = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let a = DVector::from_vec(vec![0.5, 0.4, 2.0]);
        let s = score(&theta, &a, 0.30, 0.0, 0);
        assert!(s > 0.0, "score = {s}");
    }

    #[test]
    fn score_shrinks_as_more_items_are_administered() {
        // Same item/theta, only `administered` changes -> interval delta shrinks -> score shrinks.
        let theta = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let a = DVector::from_vec(vec![0.5, 0.4, 2.0]);
        let s_round1 = score(&theta, &a, 0.30, 0.0, 0);
        let s_round4 = score(&theta, &a, 0.30, 0.0, 3);
        assert!(
            s_round4 < s_round1,
            "expected shrinking interval to reduce score: round1={s_round1} round4={s_round4}"
        );
    }

    #[test]
    fn score_is_symmetric_in_theta_direction() {
        // z_hat = 0 either way (a·theta+d = 0), so the integration interval is centered
        // identically -> scores should match.
        let a = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let theta_pos = DVector::from_vec(vec![0.4, 0.0, 0.0]);
        let theta_neg = DVector::from_vec(vec![-0.4, 0.0, 0.0]);
        let s_pos = score(&theta_pos, &a, -0.4, 0.0, 1);
        let s_neg = score(&theta_neg, &a, 0.4, 0.0, 1);
        assert!((s_pos - s_neg).abs() < 1e-9, "s_pos={s_pos} s_neg={s_neg}");
    }
}
