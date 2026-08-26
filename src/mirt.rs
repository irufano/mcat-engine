#![allow(dead_code)]

use nalgebra::{DMatrix, DVector};

pub const EPSILON: f64 = 1e-10;

/// Compute P(X=1 | θ) under M3PL
///
/// P = c + (1-c) * sigmoid(a'θ + d)
pub fn probability(theta: &DVector<f64>, a: &DVector<f64>, d: f64, c: f64) -> f64 {
    let linear = a.dot(theta) + d;
    let p_star = sigmoid(linear);
    c + (1.0 - c) * p_star
}

/// Sigmoid (logistic) function
///
/// σ(x) = 1 / (1 + e^(-x))
pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Fisher Information Matrix for one item at θ
///
/// I_i(θ) = [P'_i]² / (P_i * Q_i) * a_i * a_i'
pub fn item_fim(theta: &DVector<f64>, a: &DVector<f64>, d: f64, c: f64) -> DMatrix<f64> {
    let linear = a.dot(theta) + d;
    let p_star = sigmoid(linear);
    let p = c + (1.0 - c) * p_star;
    let q = 1.0 - p;
    let p_prime = (1.0 - c) * p_star * (1.0 - p_star);
    // The + EPSILON guards against division by zero.
    // p * q can reach exactly 0.0 in two edge cases:
    // P → 1.0 (very high ability, very easy item): q = 1 - P → 0
    // P → 0.0 (very low ability, very hard item): p → 0
    // Both produce p * q = 0.0, which would give weight = infinity (or NaN in Rust's float arithmetic), corrupting the FIM matrix.
    let weight = (p_prime * p_prime) / (p * q + EPSILON);
    a * a.transpose() * weight
}

/// Log-likelihood contribution from one item
pub fn log_likelihood_item(x: u8, theta: &DVector<f64>, a: &DVector<f64>, d: f64, c: f64) -> f64 {
    let p = probability(theta, a, d, c).max(EPSILON).min(1.0 - EPSILON);
    if x == 1 { p.ln() } else { (1.0 - p).ln() }
}

/// Serialize DMatrix<f64> to flat row-major Vec<f64>
pub fn fim_to_flat(fim: &DMatrix<f64>) -> Vec<f64> {
    let k = fim.nrows();
    (0..k)
        .flat_map(|r| (0..k).map(move |c| fim[(r, c)]))
        .collect()
}

/// Deserialize flat row-major Vec<f64> to DMatrix<f64>
pub fn flat_to_fim(flat: &[f64], k: usize) -> DMatrix<f64> {
    DMatrix::from_row_slice(k, k, flat)
}

/// Per-dimension standard errors: sqrt of the diagonal of the inverse FIM.
/// `None` for a dimension when the FIM isn't invertible or the variance is non-positive.
pub fn se_vector(fim: &DMatrix<f64>, k: usize) -> Vec<Option<f64>> {
    match fim.clone().try_inverse() {
        Some(inv) => (0..k)
            .map(|i| {
                let v = inv[(i, i)];
                if v > 0.0 { Some(v.sqrt()) } else { None }
            })
            .collect(),
        None => vec![None; k],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    // ─── sigmoid ─────────────────────────────────────────────────────────────

    #[test]
    fn sigmoid_zero_is_half() {
        let result = sigmoid(0.0);
        assert!((result - 0.5).abs() < 1e-10, "sigmoid(0) = {result}");
    }

    #[test]
    fn sigmoid_large_positive_approaches_one() {
        assert!(sigmoid(100.0) > 0.9999);
    }

    #[test]
    fn sigmoid_large_negative_approaches_zero() {
        assert!(sigmoid(-100.0) < 0.0001);
    }

    #[test]
    fn sigmoid_known_value() {
        // sigmoid(1.0) ≈ 0.7310585786
        let result = sigmoid(1.0);
        assert!(
            (result - 0.7310585786).abs() < 1e-6,
            "sigmoid(1) = {result}"
        );
    }

    // ─── probability (M3PL) ──────────────────────────────────────────────────

    #[test]
    fn probability_no_guessing_at_zero_theta() {
        // a=[1], d=0, c=0 → P = sigmoid(0) = 0.5
        let theta = DVector::from_vec(vec![0.0]);
        let a = DVector::from_vec(vec![1.0]);
        let p = probability(&theta, &a, 0.0, 0.0);
        assert!((p - 0.5).abs() < 1e-10, "p = {p}");
    }

    #[test]
    fn probability_with_guessing_floor() {
        // c=0.25, very low ability → P approaches c
        let theta = DVector::from_vec(vec![-100.0]);
        let a = DVector::from_vec(vec![1.0]);
        let p = probability(&theta, &a, 0.0, 0.25);
        assert!((p - 0.25).abs() < 1e-6, "p = {p}, expected ≈ 0.25");
    }

    #[test]
    fn probability_approaches_one_at_high_theta() {
        let theta = DVector::from_vec(vec![100.0]);
        let a = DVector::from_vec(vec![1.0]);
        let p = probability(&theta, &a, 0.0, 0.0);
        assert!(p > 0.9999, "p = {p}");
    }

    #[test]
    fn probability_multidim_real_example() {
        // θ=[1.0, 0.5], a=[1.2, 0.8], d=-0.5, c=0.2
        // linear = 1.2*1.0 + 0.8*0.5 - 0.5 = 1.1
        // p_star = sigmoid(1.1) ≈ 0.75026
        // P = 0.2 + 0.8 * 0.75026 ≈ 0.80021
        let theta = DVector::from_vec(vec![1.0, 0.5]);
        let a = DVector::from_vec(vec![1.2, 0.8]);
        let p = probability(&theta, &a, -0.5, 0.2);
        assert!((p - 0.80021).abs() < 1e-4, "p = {p}, expected ≈ 0.80021");
    }

    // ─── item_fim ────────────────────────────────────────────────────────────

    #[test]
    fn item_fim_is_symmetric() {
        let theta = DVector::from_vec(vec![0.5, -0.3]);
        let a = DVector::from_vec(vec![1.0, 0.7]);
        let fim = item_fim(&theta, &a, -0.2, 0.15);
        assert_eq!(fim.nrows(), 2);
        assert_eq!(fim.ncols(), 2);
        let diff = (fim[(0, 1)] - fim[(1, 0)]).abs();
        assert!(diff < 1e-10, "FIM not symmetric: {diff}");
    }

    #[test]
    fn item_fim_1d_known_value() {
        // θ=[0], a=[1], d=0, c=0
        // p_star=0.5, p'=0.25, weight=0.25²/0.25=0.25 → FIM=[[0.25]]
        let theta = DVector::from_vec(vec![0.0]);
        let a = DVector::from_vec(vec![1.0]);
        let fim = item_fim(&theta, &a, 0.0, 0.0);
        assert!(
            (fim[(0, 0)] - 0.25).abs() < 1e-6,
            "FIM[0,0] = {}",
            fim[(0, 0)]
        );
    }

    #[test]
    fn item_fim_diagonal_is_nonnegative() {
        let theta = DVector::from_vec(vec![1.0, -1.0]);
        let a = DVector::from_vec(vec![0.9, 1.1]);
        let fim = item_fim(&theta, &a, 0.0, 0.2);
        assert!(fim[(0, 0)] >= 0.0);
        assert!(fim[(1, 1)] >= 0.0);
    }

    // ─── log_likelihood_item ─────────────────────────────────────────────────

    #[test]
    fn log_likelihood_correct_response_high_ability() {
        // High θ → P≈1, correct answer → ll near 0
        let theta = DVector::from_vec(vec![3.0]);
        let a = DVector::from_vec(vec![1.0]);
        let ll = log_likelihood_item(1, &theta, &a, 0.0, 0.0);
        assert!(ll > -0.05, "ll = {ll}, expected near 0");
    }

    #[test]
    fn log_likelihood_wrong_response_high_ability() {
        // High θ → P≈1, wrong answer → ll very negative
        let theta = DVector::from_vec(vec![3.0]);
        let a = DVector::from_vec(vec![1.0]);
        let ll = log_likelihood_item(0, &theta, &a, 0.0, 0.0);
        assert!(ll < -2.0, "ll = {ll}, expected very negative");
    }

    #[test]
    fn log_likelihood_at_chance() {
        // P=0.5 → ln(0.5) ≈ -0.6931 for either response
        let theta = DVector::from_vec(vec![0.0]);
        let a = DVector::from_vec(vec![1.0]);
        let ll1 = log_likelihood_item(1, &theta, &a, 0.0, 0.0);
        let ll0 = log_likelihood_item(0, &theta, &a, 0.0, 0.0);
        assert!((ll1 - (-0.6931471)).abs() < 1e-4, "ll(x=1) = {ll1}");
        assert!((ll0 - (-0.6931471)).abs() < 1e-4, "ll(x=0) = {ll0}");
    }

    // ─── fim_to_flat / flat_to_fim roundtrip ─────────────────────────────────

    #[test]
    fn fim_flat_roundtrip_2x2() {
        let theta = DVector::from_vec(vec![0.5, -0.5]);
        let a = DVector::from_vec(vec![1.2, 0.9]);
        let original = item_fim(&theta, &a, -0.3, 0.1);

        let flat = fim_to_flat(&original);
        assert_eq!(flat.len(), 4);

        let recovered = flat_to_fim(&flat, 2);
        for r in 0..2 {
            for c in 0..2 {
                let diff = (original[(r, c)] - recovered[(r, c)]).abs();
                assert!(diff < 1e-12, "mismatch at [{r},{c}]: {diff}");
            }
        }
    }

    #[test]
    fn flat_to_fim_correct_row_major_layout() {
        // [[1,2],[3,4]] stored row-major as [1,2,3,4]
        let flat = vec![1.0, 2.0, 3.0, 4.0];
        let m = flat_to_fim(&flat, 2);
        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m[(1, 1)], 4.0);
    }
}
