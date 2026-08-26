use nalgebra::DMatrix;

/// Maximize det(I_(n-1) + I_i)
pub fn score(updated_fim: &DMatrix<f64>) -> f64 {
    updated_fim.determinant()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn diagonal_fim_score_is_product_of_diagonal() {
        let fim = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 3.0]));
        assert!((score(&fim) - 6.0).abs() < 1e-9);
    }

    #[test]
    fn singular_fim_scores_zero() {
        let fim = DMatrix::<f64>::zeros(2, 2);
        assert_eq!(score(&fim), 0.0);
    }

    #[test]
    fn more_information_scores_higher() {
        let small = DMatrix::from_diagonal(&DVector::from_vec(vec![1.0, 1.0]));
        let large = DMatrix::from_diagonal(&DVector::from_vec(vec![2.0, 2.0]));
        assert!(score(&large) > score(&small));
    }
}
