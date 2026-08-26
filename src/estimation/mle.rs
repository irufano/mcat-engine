use super::EstimationInput;
use crate::debug::{EstimationDebug, EstimationIteration};
use crate::mirt::{EPSILON, sigmoid};
use nalgebra::{DMatrix, DVector};

pub fn estimate(input: &EstimationInput) -> (DVector<f64>, EstimationDebug) {
    let mut theta = DVector::zeros(input.k);
    let mut iterations = Vec::new();
    let mut converged = false;
    for iter in 0..100 {
        let (grad, hess) = gradient_hessian(&theta, input);
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
        method: "mle".to_string(),
        items_used: input.responses.len(),
        iterations,
        converged: Some(converged),
        final_theta: theta.as_slice().to_vec(),
        // MLE's SE is derived afterward from the cumulative FIM (McatEngine::compute_se),
        // not produced here.
        posterior_se: None,
        extra: serde_json::Value::Null,
    };
    (theta, debug)
}

fn gradient_hessian(theta: &DVector<f64>, input: &EstimationInput) -> (DVector<f64>, DMatrix<f64>) {
    let mut grad = DVector::zeros(input.k);
    let mut hess = DMatrix::zeros(input.k, input.k);
    for i in 0..input.responses.len() {
        let a = &input.item_a_vecs[i];
        let d = input.item_d[i];
        let c = input.item_c[i];
        let x = input.responses[i];
        let linear = a.dot(theta) + d;
        let p_star = sigmoid(linear);
        let p = c + (1.0 - c) * p_star;
        let q = 1.0 - p;
        let p_prime = (1.0 - c) * p_star * (1.0 - p_star);
        let residual = (x as f64 - p) * p_prime / (p * q + EPSILON);
        grad += a * residual;
        let w = (p_prime * p_prime) / (p * q + EPSILON);
        hess += a * a.transpose() * w;
    }
    (grad, hess)
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
            k: a.first().map(|v| v.len()).unwrap_or(1),
        }
    }

    #[test]
    fn recovers_a_mixed_response_pattern_at_an_interior_maximum() {
        // Three items of increasing difficulty; a mixed correct/incorrect pattern gives
        // the likelihood a genuine interior maximum (unlike an all-correct/all-wrong
        // pattern, which drives theta to +/-infinity — see the diverges_... test below).
        let a = vec![
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![1.0]),
            DVector::from_vec(vec![1.0]),
        ];
        let d = [1.0, 0.0, -1.0];
        let c = [0.0, 0.0, 0.0];
        let r = [1u8, 1, 0];
        let prior_mean = DVector::from_vec(vec![0.0]);
        let prior_cov_inv = DMatrix::<f64>::identity(1, 1);
        let (theta, debug) = estimate(&input(&a, &d, &c, &r, &prior_mean, &prior_cov_inv));

        assert_eq!(debug.method, "mle");
        assert_eq!(debug.items_used, 3);
        assert_eq!(debug.converged, Some(true));
        assert!(theta[0].is_finite());
        // Correct on the two easier items, wrong on the hardest -> a middling ability,
        // not driven to an extreme.
        assert!(
            theta[0].abs() < 5.0,
            "expected a moderate theta, got {}",
            theta[0]
        );
    }

    #[test]
    fn all_correct_responses_drive_theta_to_an_extreme_boundary() {
        // Exactly examples/estimation/mle/estimation_mle.rs's DEMO 3 (sample_bank(), all
        // 3 items answered correctly) — Mulder & van der Linden (2009), p.276: an
        // all-correct pattern has no finite MLE, since P(θ)→1 for every item as θ→+∞
        // along a positive-discrimination direction.
        //
        // Note: `converged` still ends up `Some(true)` here — once θ is large enough,
        // the gradient and Hessian both underflow together and the Newton step
        // artifactually drops below the 1e-6 threshold. That's *not* a genuine interior
        // maximum (see the doc comment above and `estimation_mle.rs`'s own printed
        // output) — the giveaway is theta landing far outside any plausible ability
        // range, which is what this test actually asserts on.
        let a = vec![
            DVector::from_vec(vec![1.9, 0.2, 0.3]),
            DVector::from_vec(vec![0.3, 1.9, 0.4]),
            DVector::from_vec(vec![0.5, 0.4, 2.0]),
        ];
        let d = [0.40, 0.80, 0.30];
        let c = [0.0, 0.0, 0.0];
        let r = [1u8, 1, 1];
        let prior_mean = DVector::from_vec(vec![0.0, 0.0, 0.0]);
        let prior_cov_inv = DMatrix::<f64>::identity(3, 3);
        let (theta, _debug) = estimate(&input(&a, &d, &c, &r, &prior_mean, &prior_cov_inv));

        assert!(
            theta.iter().all(|&t| t > 5.0),
            "expected every dimension pushed to an extreme, got {theta:?}"
        );
    }
}
