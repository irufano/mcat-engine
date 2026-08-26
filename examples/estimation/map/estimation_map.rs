//! Maximum A Posteriori (MAP / Bayes Modal) Estimation of ability θ
//!
//! Walks through three demonstrations of the production MAP routine
//! (`mcat_engine::estimation::map::estimate`, src/mcat/estimation/map.rs):
//!
//!   Theory anchor — Magis & Raîche (2012), "Random Generation of Response
//!   Patterns under Computerized Adaptive Testing with the R Package catR",
//!   Journal of Statistical Software 48(8), open access, §2.2 "Ability
//!   estimation", p.4-5:
//!
//!     Eq.(5), p.5:  log g(θ) = log f(θ) + log L(θ)     (posterior ∝ prior × likelihood)
//!     θ̂_BM = argmax_θ g(θ)                              (the "Bayes modal" estimator)
//!     Eq.(6), p.5:  se(θ̂_BM) = 1 / sqrt( 1/σ² + Σᵢ Iᵢ(θ̂_BM) )   (normal prior N(μ,σ²))
//!
//!   The production code performs Newton-Raphson directly on log g(θ) (rather
//!   than computing se(θ̂_BM) in closed form). Differentiating Eq.(5) for a
//!   multivariate normal prior N(μ, Σ) gives (derived below, not copied from
//!   any single source — standard multivariate-normal calculus applied to
//!   Eq.(5)):
//!
//!     ∇log g(θ)  = ∇log L(θ) - Σ⁻¹(θ-μ)
//!     H(log g)   ≈ -H_LL - Σ⁻¹    (using the Fisher/expected form of H_LL,
//!                                  i.e. the FIM of Mulder & van der Linden
//!                                  2009, Eq.4, p.276 — same substitution
//!                                  used for MLE, see estimation_mle.rs)
//!
//!   which is exactly `map.rs`'s `hess = prior_cov_inv + Σ FIM_i` and
//!   `grad -= prior_cov_inv * (theta - prior_mean)`.
//!
//!   DEMO 1 — Derivation check (1 dimension), prior N(0,1):
//!     Same 3-item test as estimation_mle.rs DEMO 1 (Baker 2001, p.86-88),
//!     now with a standard normal prior added — shows term-by-term how the
//!     MAP gradient/Hessian differ from the MLE ones already verified there.
//!
//!   DEMO 2 — Multidimensional (k=3), one item per dimension:
//!     Same items/responses as estimation_mle.rs DEMO 2.
//!
//!   DEMO 3 — Regularizes the MLE divergence case:
//!     Same all-correct scenario as estimation_mle.rs DEMO 3 — MAP stays
//!     finite because the prior's -Σ⁻¹(θ-μ) term pulls the gradient back to
//!     zero at a finite θ, unlike the unregularized likelihood.
//!
//! Run:
//!   cargo run --example estimation_map

use mcat_engine::estimation::{EstimationInput, EstimationMethod, estimate};
use mcat_engine::mirt::{EPSILON, probability, sigmoid};
use nalgebra::{DMatrix, DVector};

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

// ─── DEMO 1 — 1-dimensional MAP vs MLE (same items as estimation_mle.rs) ──────

fn demo1_map_vs_mle_1d() {
    section("DEMO 1 — MAP vs MLE (k=1): Prior N(0,1) added to Baker (2001) example");

    let a = [1.0_f64, 1.2, 0.8];
    let d = [1.0_f64, 0.0, -0.8];
    let c = [0.0_f64, 0.0, 0.0];
    let u = [1_u8, 0, 1];
    let prior_mean = 0.0_f64;
    let prior_var = 1.0_f64;
    let prior_cov_inv = 1.0 / prior_var;

    println!("  Same items as estimation_mle.rs DEMO 1: a=[1.0,1.2,0.8], d=[1.0,0.0,-0.8],");
    println!(
        "  u=[1,0,1]. Prior: N(μ=0, σ²=1) → prior_cov_inv = 1/σ² = {:.1}",
        prior_cov_inv
    );
    println!(
        "  Starting θ̂_0 = prior_mean = {:.1}  (map.rs:6: `input.prior_mean.clone()`,",
        prior_mean
    );
    println!("  vs. MLE which starts at 0.0 unconditionally — here they coincide since μ=0)");
    println!();

    let mut theta = prior_mean;
    for iter in 1..=6 {
        println!("  Iteration {iter} — θ̂ = {:.4}", theta);
        let mut grad_ll = 0.0_f64;
        let mut hess_ll = 0.0_f64;
        for i in 0..3 {
            let theta_vec = DVector::from_vec(vec![theta]);
            let a_vec = DVector::from_vec(vec![a[i]]);
            let p = probability(&theta_vec, &a_vec, d[i], c[i]); // ← API call
            let q = 1.0 - p;
            grad_ll += a[i] * (u[i] as f64 - p); // p_prime=PQ for 2PL ⇒ residual=(u-P)
            hess_ll += a[i] * a[i] * p * q;
            println!("    item {}: P={:.6}  Q={:.6}", i + 1, p, q);
        }
        let prior_term = prior_cov_inv * (theta - prior_mean);
        let grad_map = grad_ll - prior_term;
        let hess_map = hess_ll + prior_cov_inv;
        let delta = grad_map / hess_map;
        println!(
            "    ∇LL={:+.4}  prior_term=Σ⁻¹(θ-μ)={:+.4}  ∇MAP=∇LL-prior_term={:+.4}",
            grad_ll, prior_term, grad_map
        );
        println!(
            "    H_LL={:.4}  +Σ⁻¹={:.1}  H_MAP={:.4}  Δθ̂=∇MAP/H_MAP={:+.4}",
            hess_ll, prior_cov_inv, hess_map, delta
        );
        theta += delta;
        println!("    θ̂_new = {:.4}", theta);
        println!();
        if delta.abs() < 1e-6 {
            println!("    ‖Δθ̂‖ < 1e-6 → converged.");
            break;
        }
    }
    println!("  Final θ̂_MAP ≈ {:.6}", theta);
    println!("  (compare to θ̂_MLE ≈ 0.324846 on the same items with no prior, see");
    println!("   estimation_mle.rs DEMO 1 — MAP is pulled");
    println!("   slightly toward the prior mean 0, exactly the regularization behavior");
    println!("   described by Magis & Raîche 2012, p.4: \"the BM estimator ... obtained by a");
    println!("   combination of the prior distribution f(θ) and the likelihood function L(θ)\")");
}

// ─── DEMO 2 — Multidimensional (k=3) ───────────────────────────────────────────

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

/// Full 7-item bank — same as estimation_mle.rs's `full_bank()`, reused here so
/// the MAP-vs-MLE comparison in DEMO 2 is on identical data.
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
    section("DEMO 2 — Multidimensional (k=3): MAP Gradient Vector & Hessian Matrix");

    println!("  ∇MAP = ∇LL - Σ⁻¹(θ-μ)     H_MAP = H_LL(=FIM) + Σ⁻¹");
    println!("  Prior: μ=[0,0,0], Σ=I₃ (prior_cov_inv=I₃) — TestSettings default.");
    println!("  Same 7-item bank & mixed response pattern as estimation_mle.rs DEMO 2.");
    println!();

    let bank = full_bank();
    let responses: [u8; 7] = [1, 0, 0, 1, 1, 0, 1];
    let k = 3;
    let prior_mean = DVector::<f64>::zeros(k);
    let prior_cov_inv = DMatrix::<f64>::identity(k, k);
    let mut theta = prior_mean.clone();

    for iter in 1..=3 {
        println!("  Iteration {iter} — θ̂ = {:?}", theta.as_slice());
        let mut grad = DVector::<f64>::zeros(k);
        let mut hess = prior_cov_inv.clone();

        for (item, &x) in bank.iter().zip(responses.iter()) {
            let a_vec = DVector::from_vec(item.a.to_vec());
            let linear = a_vec.dot(&theta) + item.d;
            let p_star = sigmoid(linear); // ← API call
            let p = item.c + (1.0 - item.c) * p_star;
            let q = 1.0 - p;
            let p_prime = (1.0 - item.c) * p_star * (1.0 - p_star);
            let residual = (x as f64 - p) * p_prime / (p * q + EPSILON);
            let w = (p_prime * p_prime) / (p * q + EPSILON);
            grad += &a_vec * residual;
            hess += &a_vec * a_vec.transpose() * w;
            println!(
                "    {}: P={:.6}  residual={:+.6}  w={:.6}",
                item.id, p, residual, w
            );
        }

        let prior_term = &prior_cov_inv * (&theta - &prior_mean);
        println!("    Σaᵢ residual (∇LL)      = {:?}", grad.as_slice());
        println!("    Σ⁻¹(θ-μ) (prior term)  = {:?}", prior_term.as_slice());
        grad -= &prior_term;
        println!("    ∇MAP = ∇LL - Σ⁻¹(θ-μ)   = {:?}", grad.as_slice());
        println!(
            "    H_MAP diag              = [{:.4}, {:.4}, {:.4}]  (= FIM diag + 1.0 prior precision)",
            hess[(0, 0)],
            hess[(1, 1)],
            hess[(2, 2)]
        );

        let delta = hess
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(k, k))
            * &grad;
        theta += &delta;
        println!("    Δθ̂ = H_MAP⁻¹ · ∇MAP = {:?}", delta.as_slice());
        println!("    θ̂_new = {:?}", theta.as_slice());
        println!();
        if delta.norm() < 1e-6 {
            break;
        }
    }

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
    let (theta_api, _) = estimate(&EstimationMethod::Map, &input); // ← real production API call
    let (theta_mle_api, _) = estimate(&EstimationMethod::Mle, &input);

    println!(
        "  Production API map::estimate(): θ̂_MAP = {:?}",
        theta_api.as_slice()
    );
    println!(
        "  Production API mle::estimate(): θ̂_MLE = {:?}  (no prior, for comparison)",
        theta_mle_api.as_slice()
    );
    println!("  θ̂_MAP is shrunk toward μ=[0,0,0] relative to θ̂_MLE on every dimension —");
    println!("  the prior precision Σ⁻¹=I adds exactly 1.0 to each diagonal of the Hessian,");
    println!("  which is why MAP's information is always ≥ MLE's (Σ⁻¹ ⪰ 0 added to FIM ⪰ 0).");
}

// ─── DEMO 3 — MAP regularizes the MLE divergence case ─────────────────────────

fn demo3_regularized_divergence() {
    section("DEMO 3 — MAP Regularizes the All-Correct Divergence Case");

    let bank = sample_bank();
    let responses: [u8; 3] = [1, 1, 1]; // all correct — MLE diverges (see estimation_mle.rs DEMO 3)
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
    let (theta_map, _) = estimate(&EstimationMethod::Map, &input); // ← real production API call
    let (theta_mle, _) = estimate(&EstimationMethod::Mle, &input);

    println!("  All 3 items answered correctly (u = [1,1,1]).");
    println!(
        "  Production API mle::estimate(): θ̂_MLE = {:?}  ‖θ̂‖={:.2}  (driven to iteration cap)",
        theta_mle.as_slice(),
        theta_mle.norm()
    );
    println!(
        "  Production API map::estimate(): θ̂_MAP = {:?}  ‖θ̂‖={:.2}  (finite, regularized)",
        theta_map.as_slice(),
        theta_map.norm()
    );
    println!();
    println!("  Why MAP stays finite: at large θ, ∇LL → 0⁺ (every item's residual (1-P)→0) but");
    println!("  never crosses zero — with the prior, ∇MAP = ∇LL - Σ⁻¹(θ-μ) DOES cross zero at");
    println!("  a finite θ, because -Σ⁻¹(θ-μ) grows more negative as θ grows, eventually");
    println!("  cancelling the small positive ∇LL. This is exactly Magis & Raîche (2012)'s");
    println!("  description of the BM estimator (p.4-5): the mode of prior × likelihood always");
    println!("  exists for a proper (integrable) prior, even when the likelihood alone has none.");
}

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  MCAT · Maximum A Posteriori (MAP / Bayes Modal) Estimation of θ");
    println!("══════════════════════════════════════════════════════════════════════════");

    demo1_map_vs_mle_1d();
    demo2_multidimensional();
    demo3_regularized_divergence();

    println!();
    divider();
    println!();
}
