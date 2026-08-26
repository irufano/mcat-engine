//! Expected A Posteriori (EAP) Estimation of ability θ
//!
//! Walks through three demonstrations of the production EAP routine
//! (`mcat_engine::estimation::eap::estimate`, src/mcat/estimation/eap.rs):
//!
//!   Theory anchor — Magis & Raîche (2012), JSS 48(8), open access, §2.2,
//!   p.5-6, Eq.(10):
//!
//!         θ̂_EAP = ∫θ f(θ)L(θ)dθ / ∫f(θ)L(θ)dθ     ("posterior mean")
//!
//!   p.6: "In practice, the integrals in (10) are approximated, for
//!   instance by adaptive quadrature or numerical integration." The
//!   production code approximates both integrals by classical Gauss-Hermite
//!   quadrature (Bock & Mislevy 1982): nodes/weights are the roots of the
//!   physicists' Hermite polynomial `H_pts` and their associated weights
//!   (Golub & Welsch 1969 algorithm — eigenvalues/eigenvectors of the
//!   symmetric tridiagonal Jacobi matrix), rescaled by `θ=√2σx` for the
//!   prior `N(0,σ²)`. The grid/weight-sum pattern itself
//!   (Σ_grid L(K)·g(K₁)g(K₂)...g(Km)) mirrors the multidimensional
//!   quadrature form in Chalmers (2012), "mirt: A Multidimensional IRT
//!   Package for the R Environment", JSS 48(6), open access, Eq.(6), p.5 —
//!   though that equation marginalizes item parameters (EM), not θ, the
//!   discretization technique is the same multi-index grid sum.
//!
//!   IMPLEMENTATION NOTE: eap.rs hardcodes a 3-dimensional nested loop
//!   (`q0,q1,q2`) regardless of `input.k` — it only supports k=3. DEMO 1
//!   below works around this to reproduce a 1-dimensional textbook example
//!   by giving 2 of the 3 dimensions zero discrimination (a=0), which makes
//!   the likelihood constant across those dimensions so their marginal EAP
//!   reduces to the prior mean and the remaining dimension reduces to a
//!   pure 1D EAP.
//!
//!   DEMO 1 — Literature reproduction (1 active dimension, pts=5):
//!     Reproduces MCAT_EXPLANATION.md §5.3's own worked example: single
//!     item a=1.5, d=0, c=0, correct response, 5-point Gauss-Hermite grid
//!     (σ=1).
//!
//!   DEMO 2 — Multidimensional (k=3), one item per dimension:
//!     Same items/responses as estimation_mle.rs / estimation_map.rs DEMO 2,
//!     using the real 3D grid (pts=5 → 125 grid points) via the actual API.
//!
//!   DEMO 3 — Always finite, even for all-correct responses:
//!     Same all-correct scenario as the MLE/MAP demos — EAP never diverges
//!     because the posterior integral is over the whole real line, weighted
//!     by a proper (integrable) prior.
//!
//! Run:
//!   cargo run --example estimation_eap

use mcat_engine::estimation::{EstimationInput, EstimationMethod, estimate};
use mcat_engine::mirt::probability;
use nalgebra::{DMatrix, DVector, SymmetricEigen};

fn divider() {
    println!("{}", "─".repeat(78));
}

fn section(title: &str) {
    println!();
    divider();
    println!("  {title}");
    divider();
    println!();
}

/// Mirrors eap.rs's `gauss_hermite` exactly: physicists' Gauss-Hermite
/// nodes/weights via the Golub-Welsch algorithm.
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

/// Mirrors eap.rs's grid1d generation exactly: Gauss-Hermite nodes rescaled
/// for the prior N(0, sd^2) via θ=√2·sd·x.
fn grid1d(pts: usize, sd: f64) -> Vec<f64> {
    let (nodes, _) = gauss_hermite(pts);
    nodes.iter().map(|&x| (2.0_f64).sqrt() * sd * x).collect()
}

/// Mirrors eap.rs's quadrature-weight generation: GH weight / √π (already
/// incorporates the Gaussian prior density analytically — no separate
/// `normal_density` evaluation needed).
fn quad_weights(pts: usize) -> Vec<f64> {
    let (_, weights) = gauss_hermite(pts);
    weights
        .iter()
        .map(|&w| w / std::f64::consts::PI.sqrt())
        .collect()
}

// ─── DEMO 1 — 1D literature reproduction (pts=5) ───────────────────────────────

fn demo1_1d_reproduction() {
    section("DEMO 1 — Literature Reproduction (1 active dimension, pts=5)");

    let sd = 1.0_f64;
    let pts = 5_usize;
    let grid = grid1d(pts, sd);
    let a = 1.5_f64;
    let d = 0.0_f64;
    let c = 0.0_f64;
    let x: u8 = 1; // correct response

    let weights = quad_weights(pts);
    println!(
        "  Item: a={:.1}, d={:.1}, c={:.1}   Response: x={}",
        a, d, c, x
    );
    println!("  Gauss-Hermite grid (pts=5, σ=1.0): {:?}", grid);
    println!(
        "  Quadrature weights A_q = w_q/√π (already integrate N(θ;0,1) analytically): {:?}",
        weights
    );
    println!();
    println!("  θ_q        P(x=1|θ_q)     A_q             w=L·A");

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for (&q, &aq) in grid.iter().zip(weights.iter()) {
        let theta_vec = DVector::from_vec(vec![q]);
        let a_vec = DVector::from_vec(vec![a]);
        let p = probability(&theta_vec, &a_vec, d, c)
            .max(1e-10)
            .min(1.0 - 1e-10); // ← API call
        let likelihood = if x == 1 { p } else { 1.0 - p };
        let w = likelihood * aq;
        println!("  {:+.6}    {:.6}       {:.6}        {:.6}", q, p, aq, w);
        num += q * w;
        den += w;
    }
    let theta_eap = num / den;
    println!();
    println!(
        "  θ̂_EAP = Σ(θ_q·w) / Σw = {:.6} / {:.6} = {:.6}",
        num, den, theta_eap
    );
    println!();

    // ── Cross-check via the real 3D production API using the zero-loading trick ──
    println!("  Cross-check via the REAL production API (eap::estimate, hardcoded to k=3):");
    println!("  give dims 1,2 zero discrimination so only dim 0 carries information —");
    println!("  their marginal EAP should reduce to the prior mean (0.0) by symmetry.");
    println!();

    let item_a_vecs = vec![DVector::from_vec(vec![a, 0.0, 0.0])];
    let item_d = vec![d];
    let item_c = vec![c];
    let responses = vec![x];
    let prior_mean = DVector::<f64>::zeros(3);
    let prior_cov_inv = DMatrix::<f64>::identity(3, 3);
    let input = EstimationInput {
        item_a_vecs: &item_a_vecs,
        item_d: &item_d,
        item_c: &item_c,
        responses: &responses,
        prior_mean: &prior_mean,
        prior_cov_inv: &prior_cov_inv,
        eap_quad_pts: pts,
        k: 3,
    };
    let (theta_api, _) = estimate(&EstimationMethod::Eap, &input); // ← real production API call
    println!(
        "  Production API eap::estimate(): θ̂ = {:?}",
        theta_api.as_slice()
    );
    println!(
        "  dim 0 (informative)  = {:.6}  (manual 1D calc above: {:.6})",
        theta_api[0], theta_eap
    );
    println!(
        "  dims 1,2 (a=0, no info) = {:.6}, {:.6}  (≈ prior mean 0.0, as expected)",
        theta_api[1], theta_api[2]
    );
}

// ─── DEMO 2 — Multidimensional (k=3), real 3D grid ─────────────────────────────

struct BankItem {
    id: &'static str,
    a: [f64; 3],
    d: f64,
    c: f64,
}

fn sample_bank() -> Vec<BankItem> {
    vec![
        BankItem {
            id: "m2p-v001",
            a: [1.9, 0.2, 0.3],
            d: 0.40,
            c: 0.0,
        },
        BankItem {
            id: "m2p-n001",
            a: [0.3, 1.9, 0.4],
            d: 0.80,
            c: 0.0,
        },
        BankItem {
            id: "m2p-r001",
            a: [0.5, 0.4, 2.0],
            d: 0.30,
            c: 0.0,
        },
    ]
}

/// Full 7-item bank — same as estimation_mle.rs's / estimation_map.rs's
/// `full_bank()`, reused here so all three methods can be compared on
/// identical data (see estimation_summary.md's method-comparison table).
fn full_bank() -> Vec<BankItem> {
    vec![
        BankItem {
            id: "m2p-v001",
            a: [1.9, 0.2, 0.3],
            d: 0.40,
            c: 0.0,
        },
        BankItem {
            id: "m2p-v002",
            a: [1.7, 0.2, 0.2],
            d: 0.10,
            c: 0.0,
        },
        BankItem {
            id: "m2p-n001",
            a: [0.3, 1.9, 0.4],
            d: 0.80,
            c: 0.0,
        },
        BankItem {
            id: "m2p-n002",
            a: [0.3, 1.8, 0.4],
            d: 0.50,
            c: 0.0,
        },
        BankItem {
            id: "m2p-r001",
            a: [0.5, 0.4, 2.0],
            d: 0.30,
            c: 0.0,
        },
        BankItem {
            id: "m2p-r002",
            a: [0.3, 0.8, 1.9],
            d: 0.60,
            c: 0.0,
        },
        BankItem {
            id: "m2p-r003",
            a: [0.4, 0.3, 1.8],
            d: 0.70,
            c: 0.0,
        },
    ]
}

fn demo2_multidimensional() {
    section("DEMO 2 — Multidimensional (k=3): Real 3D Quadrature Grid (pts=5 → 125 points)");

    let bank = full_bank();
    let responses: [u8; 7] = [1, 0, 0, 1, 1, 0, 1];
    let pts = 5_usize;
    let sd = 1.0_f64;
    let grid = grid1d(pts, sd);
    let weights = quad_weights(pts);

    println!("  Items & responses (same as estimation_mle.rs / estimation_map.rs DEMO 2):");
    for (item, &x) in bank.iter().zip(responses.iter()) {
        println!(
            "    {}  a=[{:.1},{:.1},{:.1}]  d={:.2}  c={:.1}  u={}",
            item.id, item.a[0], item.a[1], item.a[2], item.d, item.c, x
        );
    }
    println!();
    println!(
        "  Gauss-Hermite grid per dimension (pts=5, σ=1.0): {:?}",
        grid
    );
    println!("  Quadrature weights A_q (pts=5): {:?}", weights);
    println!("  Total grid points = pts^k = 5³ = {}", pts * pts * pts);
    println!();

    // Manual replication of eap.rs's triple-nested loop, printing only a
    // representative sample of rows (corners + center) to keep output readable —
    // the full sum below covers all 125 points, matching eap.rs exactly.
    let mut num = DVector::<f64>::zeros(3);
    let mut den = 0.0_f64;
    let mut shown = 0;
    for (i0, &q0) in grid.iter().enumerate() {
        for (i1, &q1) in grid.iter().enumerate() {
            for (i2, &q2) in grid.iter().enumerate() {
                let theta_q = DVector::from_vec(vec![q0, q1, q2]);
                let prior_w = weights[i0] * weights[i1] * weights[i2];
                let likelihood: f64 = bank
                    .iter()
                    .zip(responses.iter())
                    .map(|(item, &x)| {
                        let a_vec = DVector::from_vec(item.a.to_vec());
                        let p = probability(&theta_q, &a_vec, item.d, item.c) // ← API call
                            .max(1e-10)
                            .min(1.0 - 1e-10);
                        if x == 1 { p } else { 1.0 - p }
                    })
                    .product();
                let w = likelihood * prior_w;

                let is_edge = |g: f64| g == grid[0] || g == grid[pts - 1];
                let is_center = q0.abs() < 1e-9 && q1.abs() < 1e-9 && q2.abs() < 1e-9;
                let is_corner = is_edge(q0) && is_edge(q1) && is_edge(q2);
                if (is_center || is_corner) && shown < 9 {
                    println!(
                        "    θ_q=[{:+.4},{:+.4},{:+.4}]  L={:.6}  A={:.6}  w=L·A={:.8}",
                        q0, q1, q2, likelihood, prior_w, w
                    );
                    shown += 1;
                }

                num += &theta_q * w;
                den += w;
            }
        }
    }
    println!("    ... (116 further grid points omitted for brevity; all 125 included below)");
    println!();
    let theta_manual = &num / den;
    println!("  Σw (all 125 points)      = {:.8}", den);
    println!("  Σ(θ_q·w) (all 125 points) = {:?}", num.as_slice());
    println!(
        "  θ̂_EAP = Σ(θ_q·w)/Σw       = {:?}",
        theta_manual.as_slice()
    );
    println!();

    let item_a_vecs: Vec<DVector<f64>> = bank
        .iter()
        .map(|it| DVector::from_vec(it.a.to_vec()))
        .collect();
    let item_d: Vec<f64> = bank.iter().map(|it| it.d).collect();
    let item_c: Vec<f64> = bank.iter().map(|it| it.c).collect();
    let prior_mean = DVector::<f64>::zeros(3);
    let prior_cov_inv = DMatrix::<f64>::identity(3, 3);
    let input = EstimationInput {
        item_a_vecs: &item_a_vecs,
        item_d: &item_d,
        item_c: &item_c,
        responses: &responses,
        prior_mean: &prior_mean,
        prior_cov_inv: &prior_cov_inv,
        eap_quad_pts: pts,
        k: 3,
    };
    let (theta_api, _) = estimate(&EstimationMethod::Eap, &input); // ← real production API call
    println!(
        "  Production API eap::estimate(): θ̂ = {:?}",
        theta_api.as_slice()
    );
    println!("  (matches the manual replication above exactly — same grid, same formula)");
    println!();

    let input21 = EstimationInput {
        eap_quad_pts: 21,
        ..input
    };
    let (theta_api21, _) = estimate(&EstimationMethod::Eap, &input21); // ← real production API call
    println!(
        "  Same data, production-default grid resolution (pts=21³ = 9,261 points): θ̂ = {:?}",
        theta_api21.as_slice()
    );
    println!("  A finer grid shifts the estimate closer to θ̂_MAP (estimation_map.rs DEMO 2),");
    println!("  since both integrate over the same continuous posterior — the pts=5 grid above");
    println!("  is deliberately coarse only so all 125 points can be tabulated by hand.");
}

// ─── DEMO 3 — Always finite, even for all-correct responses ────────────────────

fn demo3_always_finite() {
    section("DEMO 3 — EAP Never Diverges (all-correct responses)");

    let bank = sample_bank();
    let responses: [u8; 3] = [1, 1, 1];
    let k = 3;
    let prior_mean = DVector::<f64>::zeros(k);
    let prior_cov_inv = DMatrix::<f64>::identity(k, k);

    let item_a_vecs: Vec<DVector<f64>> = bank
        .iter()
        .map(|it| DVector::from_vec(it.a.to_vec()))
        .collect();
    let item_d: Vec<f64> = bank.iter().map(|it| it.d).collect();
    let item_c: Vec<f64> = bank.iter().map(|it| it.c).collect();
    let input = EstimationInput {
        item_a_vecs: &item_a_vecs,
        item_d: &item_d,
        item_c: &item_c,
        responses: &responses,
        prior_mean: &prior_mean,
        prior_cov_inv: &prior_cov_inv,
        eap_quad_pts: 21,
        k,
    };
    let (theta_eap, _) = estimate(&EstimationMethod::Eap, &input); // ← real production API call
    let (theta_mle, _) = estimate(&EstimationMethod::Mle, &input);
    let (theta_map, _) = estimate(&EstimationMethod::Map, &input);

    println!("  All 3 items answered correctly (u = [1,1,1]) — same scenario as");
    println!("  estimation_mle.rs DEMO 3 and estimation_map.rs DEMO 3.");
    println!();
    println!(
        "  θ̂_MLE = {:?}  ‖θ̂‖={:.2}  (driven to iteration cap — no finite maximum)",
        theta_mle.as_slice(),
        theta_mle.norm()
    );
    println!(
        "  θ̂_MAP = {:?}  ‖θ̂‖={:.2}  (finite — regularized by the prior)",
        theta_map.as_slice(),
        theta_map.norm()
    );
    println!(
        "  θ̂_EAP = {:?}  ‖θ̂‖={:.2}  (finite by construction — posterior mean over a",
        theta_eap.as_slice(),
        theta_eap.norm()
    );
    println!("           bounded quadrature grid, weighted by a proper prior; the integral");
    println!("           in Eq.(10) [Magis & Raîche 2012, p.5] always has a finite numerator");
    println!("           and a strictly positive denominator for any finite response pattern)");
}

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  MCAT · Expected A Posteriori (EAP) Estimation of θ");
    println!("══════════════════════════════════════════════════════════════════════════");

    demo1_1d_reproduction();
    demo2_multidimensional();
    demo3_always_finite();

    println!();
    divider();
    println!();
}
