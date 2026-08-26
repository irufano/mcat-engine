use nalgebra::DMatrix;

const DET_THRESHOLD: f64 = 1e-10;

/// Minimize tr(I_n^{-1}), negated so higher score = better.
/// Returns NEG_INFINITY when updated_fim is singular (rank-deficient).
pub fn score(updated_fim: &DMatrix<f64>) -> f64 {
    if updated_fim.determinant().abs() < DET_THRESHOLD {
        return f64::NEG_INFINITY;
    }
    match updated_fim.clone().try_inverse() {
        Some(inv) => -inv.trace(),
        None => f64::NEG_INFINITY,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn identity_fim_scores_negative_k() {
        let fim = DMatrix::<f64>::identity(3, 3);
        assert!((score(&fim) - (-3.0)).abs() < 1e-9);
    }

    #[test]
    fn near_singular_fim_scores_negative_infinity() {
        let fim = DMatrix::<f64>::zeros(2, 2);
        assert_eq!(score(&fim), f64::NEG_INFINITY);
    }

    #[test]
    fn more_information_scores_higher() {
        // Larger information (bigger diagonal) -> smaller trace(inverse) -> higher (less
        // negative) score.
        let small_info = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let large_info = DMatrix::from_diagonal(&DVector::from_vec(vec![4.0, 4.0]));
        assert!(score(&large_info) > score(&small_info));
    }
}
