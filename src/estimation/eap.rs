use super::EstimationInput;
use crate::debug::EstimationDebug;
use crate::mirt::probability;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

/// Ceiling on the `pts^k` Gauss-Hermite grid `estimate()` below builds. The grid grows
/// exponentially in the number of trait dimensions `k` (e.g. the default `pts=21` gives
/// 9,261 points at k=3, but 4,084,101 at k=5, and ~85.7M at k=6) — unlike `map`/`mle`
/// (Newton-Raphson, cost grows with k^3 via a k×k matrix inverse), EAP has no way to scale
/// to many dimensions. This cap keeps a single estimation call from silently turning into
/// a multi-second (or worse) hang inside a live session. 500,000 is chosen so the default
/// pts=21 still works through k=4 (194,481 points) while blocking k=5+ at the default,
/// which is where response latency stops being practical for a real-time API.
pub const MAX_GRID_POINTS: u64 = 500_000;

/// Reject an EAP configuration whose `pts^k` quadrature grid would exceed
/// [`MAX_GRID_POINTS`]. Called both at settings-write time (`modules::settings::service`, so bad
/// configs are rejected before they're saved) and at estimation time
/// (`McatEngine::estimate_theta`, as a defensive check covering presets that predate this
/// validation or came from playground overrides).
pub fn check_feasible(k: usize, pts: usize) -> Result<(), String> {
    let total = (pts as u64).checked_pow(k as u32).unwrap_or(u64::MAX);
    if total <= MAX_GRID_POINTS {
        return Ok(());
    }
    let max_pts_for_k = (MAX_GRID_POINTS as f64).powf(1.0 / k.max(1) as f64).floor() as u64;
    Err(format!(
        "eap_quadrature_points^dimensions = {pts}^{k} = {total} quadrature points, which exceeds \
         the safety limit of {MAX_GRID_POINTS}. EAP's grid grows exponentially with the number of \
         trait dimensions ({k} here). Reduce eap_quadrature_points to at most {max_pts_for_k} for a \
         {k}-dimension bank, or use estimation_method \"map\" or \"mle\" instead — both scale with \
         k^3 (a k×k matrix inverse) rather than pts^k."
    ))
}

/// Physicists' Gauss-Hermite nodes/weights (roots of the Hermite polynomial
/// `H_n`, orthogonal under the weight `e^{-x^2}`), computed via the
/// Golub-Welsch algorithm: the nodes are the eigenvalues of the symmetric
/// tridiagonal Jacobi matrix with zero diagonal and off-diagonal
/// `sqrt(i/2)`, and each weight is `sqrt(pi)` times the squared first
/// component of the corresponding normalized eigenvector.
fn gauss_hermite(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut jacobi = DMatrix::<f64>::zeros(n, n);
    for i in 0..n - 1 {
        let off = ((i + 1) as f64 / 2.0).sqrt();
        jacobi[(i, i + 1)] = off;
        jacobi[(i + 1, i)] = off;
    }
    let eigen = SymmetricEigen::new(jacobi);
    let mut pairs: Vec<(f64, f64)> = (0..n)
        .map(|i| {
            let x = eigen.eigenvalues[i];
            let v0 = eigen.eigenvectors[(0, i)];
            let w = std::f64::consts::PI.sqrt() * v0 * v0;
            (x, w)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    (
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1).collect(),
    )
}

pub fn estimate(input: &EstimationInput) -> (DVector<f64>, EstimationDebug) {
    let pts = input.eap_quad_pts;
    let k = input.k;

    // Change of variable θ = μ + √2·σ·x turns ∫f(θ)N(θ;μ,σ²)dθ into
    // (1/√π)∫f(μ+√2σx)e^{-x²}dx, which Gauss-Hermite quadrature integrates
    // exactly for polynomial f up to degree 2·pts-1. μ/σ are taken from the
    // SAME prior (prior_mean/prior_cov_inv) MAP uses, per dimension, so
    // switching estimation_method between map/eap doesn't silently swap the
    // prior along with the estimator.
    let (nodes, weights) = gauss_hermite(pts);
    let quad_w: Vec<f64> = weights
        .iter()
        .map(|&w| w / std::f64::consts::PI.sqrt())
        .collect();
    let sd: Vec<f64> = (0..k)
        .map(|d| (1.0 / input.prior_cov_inv[(d, d)]).sqrt())
        .collect();
    let mu: Vec<f64> = (0..k).map(|d| input.prior_mean[d]).collect();
    let grid: Vec<Vec<f64>> = (0..k)
        .map(|d| {
            nodes
                .iter()
                .map(|&x| mu[d] + (2.0_f64).sqrt() * sd[d] * x)
                .collect()
        })
        .collect();

    let mut num = DVector::zeros(k);
    // Second raw moment E[θ²] per dimension, accumulated alongside `num` (E[θ]) so the
    // posterior variance — Var[θ] = E[θ²] - E[θ]² — falls out at the end for free, from
    // the SAME quadrature sum, instead of a second pass over the grid.
    let mut num2 = DVector::zeros(k);
    let mut den = 0.0_f64;

    // Cartesian product of the `pts` nodes across all `k` dimensions (pts^k
    // points total), enumerated via a mixed-radix counter `idx` (like an
    // odometer) instead of `k` hand-written nested loops, so this works for
    // any number of dimensions rather than only k=3.
    let mut idx = vec![0usize; k];
    for _ in 0..pts.pow(k as u32) {
        let theta_q = DVector::from_iterator(k, (0..k).map(|d| grid[d][idx[d]]));
        let prior_w: f64 = (0..k).map(|d| quad_w[idx[d]]).product();
        let likelihood: f64 = (0..input.responses.len())
            .map(|i| {
                let p = probability(
                    &theta_q,
                    &input.item_a_vecs[i],
                    input.item_d[i],
                    input.item_c[i],
                )
                .max(1e-10)
                .min(1.0 - 1e-10);
                if input.responses[i] == 1 { p } else { 1.0 - p }
            })
            .product();
        let w = likelihood * prior_w;
        num += &theta_q * w;
        num2 += theta_q.component_mul(&theta_q) * w;
        den += w;

        for d in 0..k {
            idx[d] += 1;
            if idx[d] < pts {
                break;
            }
            idx[d] = 0;
        }
    }

    let theta = if den > 1e-300 {
        num / den
    } else {
        input.prior_mean.clone()
    };

    // Posterior SD per dimension: sqrt(E[θ²] - E[θ]²) under the SAME normalized weights
    // used for the mean above. `None` for a dimension when the posterior mass is too
    // small to normalize (mirrors the `theta` fallback above) or rounding makes the
    // variance non-positive (near-zero true variance).
    let posterior_se: Option<Vec<Option<f64>>> = if den > 1e-300 {
        Some(
            (0..k)
                .map(|d| {
                    let mean = theta[d];
                    let variance = num2[d] / den - mean * mean;
                    if variance > 0.0 {
                        Some(variance.sqrt())
                    } else {
                        None
                    }
                })
                .collect(),
        )
    } else {
        None
    };

    let debug = EstimationDebug {
        method: "eap".to_string(),
        items_used: input.responses.len(),
        iterations: Vec::new(),
        converged: None,
        final_theta: theta.as_slice().to_vec(),
        posterior_se: posterior_se.clone(),
        extra: serde_json::json!({
            "quad_pts": pts, "prior_mean": mu, "prior_sd": sd, "posterior_mass": den,
            "posterior_se": posterior_se,
        }),
    };
    (theta, debug)
}

#[cfg(test)]
mod feasibility_tests {
    use super::*;

    #[test]
    fn default_pts_ok_through_k4() {
        // 21^3 = 9,261; 21^4 = 194,481 — both under the 500,000 cap.
        assert!(check_feasible(3, 21).is_ok());
        assert!(check_feasible(4, 21).is_ok());
    }

    #[test]
    fn default_pts_rejected_at_k5() {
        // 21^5 = 4,084,101 — well over the cap.
        assert!(check_feasible(5, 21).is_err());
    }

    #[test]
    fn error_message_suggests_a_workable_pts_value() {
        let err = check_feasible(5, 21).unwrap_err();
        // Whatever max_pts_for_k the message suggests should itself pass.
        let suggested: usize = err
            .split("at most ")
            .nth(1)
            .and_then(|s| s.split(' ').next())
            .and_then(|s| s.parse().ok())
            .expect("message should contain a parseable suggested pts value");
        assert!(
            check_feasible(5, suggested).is_ok(),
            "suggested pts {suggested} should itself fit"
        );
    }

    #[test]
    fn does_not_overflow_on_large_k() {
        // checked_pow must saturate rather than panic when pts/k are large.
        assert!(check_feasible(64, 100).is_err());
    }

    #[test]
    fn k_zero_is_trivially_feasible() {
        // pts^0 = 1 point regardless of pts.
        assert!(check_feasible(0, 21).is_ok());
    }
}
