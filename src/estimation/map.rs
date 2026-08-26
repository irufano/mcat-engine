use super::EstimationInput;
use crate::debug::{EstimationDebug, EstimationIteration};
use crate::mirt::{EPSILON, sigmoid};
use nalgebra::{DMatrix, DVector};

pub fn estimate(input: &EstimationInput) -> (DVector<f64>, EstimationDebug) {
    let mut theta = input.prior_mean.clone();
    let mut iterations = Vec::new();
    let mut converged = false;
    for iter in 0..100 {
        let mut grad = DVector::zeros(input.k);
        let mut hess = input.prior_cov_inv.clone();
        for i in 0..input.responses.len() {
            let a = &input.item_a_vecs[i];
            let d = input.item_d[i];
            let c = input.item_c[i];
            let x = input.responses[i];
            let linear = a.dot(&theta) + d;
            let p_star = sigmoid(linear);
            let p = c + (1.0 - c) * p_star;
            let q = 1.0 - p;
            let p_prime = (1.0 - c) * p_star * (1.0 - p_star);
            let residual = (x as f64 - p) * p_prime / (p * q + EPSILON);
            grad += a * residual;
            let w = (p_prime * p_prime) / (p * q + EPSILON);
            hess += a * a.transpose() * w;
        }
        grad -= input.prior_cov_inv * (&theta - input.prior_mean);
        let delta = hess
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(input.k, input.k))
            * &grad;
        theta += &delta;
        let delta_norm = delta.norm();
        iterations.push(EstimationIteration {
            iter,
            theta: theta.as_slice().to_vec(),
            delta_norm,
        });
        if delta_norm < 1e-6 {
            converged = true;
            break;
        }
    }
    let debug = EstimationDebug {
        method: "map".to_string(),
        items_used: input.responses.len(),
        iterations,
        converged: Some(converged),
        final_theta: theta.as_slice().to_vec(),
        // MAP's SE is derived afterward from cum_fim + prior_cov_inv
        // (McatEngine::compute_se), not produced here.
        posterior_se: None,
        extra: serde_json::Value::Null,
    };
    (theta, debug)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn input<'a>(
        a: &'a [DVector<f64>],
        d: &'a [f64],
        c: &'a [f64],
        r: &'a [u8],
        prior_mean: &'a DVector<f64>,
        prior_cov_inv: &'a DMatrix<f64>,
    ) -> EstimationInput<'a> {
        EstimationInput {
            item_a_vecs: a,
            item_d: d,
            item_c: c,
            responses: r,
            prior_mean,
            prior_cov_inv,
            eap_quad_pts: 21,
            k: prior_mean.len(),
        }
    }

    #[test]
    fn with_no_responses_theta_stays_exactly_at_the_prior_mean() {
        let prior_mean = DVector::from_vec(vec![0.7, -0.3]);
        let prior_cov_inv = DMatrix::<f64>::identity(2, 2);
        let (theta, debug) = estimate(&input(&[], &[], &[], &[], &prior_mean, &prior_cov_inv));

        assert_eq!(theta, prior_mean);
        assert_eq!(debug.method, "map");
        assert_eq!(debug.items_used, 0);
        // Zero-gradient at the starting point converges on the very first iteration.
        assert_eq!(debug.converged, Some(true));
        assert_eq!(debug.iterations.len(), 1);
    }

    #[test]
    fn regularization_keeps_a_perfect_response_pattern_bounded_and_converged() {
        // Unlike raw MLE (see estimation::mle::tests), the prior's curvature bounds an
        // all-correct pattern to a finite, converged estimate.
        let a = vec![DVector::from_vec(vec![1.0])];
        let d = [0.0];
        let c = [0.0];
        let r = [1u8];
        let prior_mean = DVector::from_vec(vec![0.0]);
        let prior_cov_inv = DMatrix::<f64>::identity(1, 1);
        let (theta, debug) = estimate(&input(&a, &d, &c, &r, &prior_mean, &prior_cov_inv));

        assert_eq!(debug.converged, Some(true));
        assert!(theta[0].is_finite());
        assert!(theta[0] > 0.0);
        assert!(
            theta[0] < 10.0,
            "expected a bounded theta, got {}",
            theta[0]
        );
    }
}
