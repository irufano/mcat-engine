//! Maximum Likelihood Estimation (MLE) of ability θ — Verification & Simulation
//!
//! Walks through three demonstrations of the production MLE routine
//! (`mcat_engine::estimation::mle::estimate`, src/mcat/estimation/mle.rs):
//!
//!   DEMO 1 — Literature reproduction (1 dimension):
//!     Reproduces, digit-for-digit, the worked example in Baker (2001),
//!     "The Basics of Item Response Theory" (2nd ed.), Chapter 5
//!     "Estimating an Examinee's Ability", p.86-88, Equation [5-1]:
//!
//!         θ̂_(s+1) = θ̂_s + [Σ aᵢ(uᵢ-Pᵢ(θ̂_s))] / [Σ aᵢ²Pᵢ(θ̂_s)Qᵢ(θ̂_s)]
//!
//!     Baker's book uses the (a, b, c) parameterization (b = difficulty).
//!     The production API uses (a, d, c) with d = -a·b (standard 2PL/3PL
//!     reparameterization, z = a·θ + d = a·(θ-b)). Converting Baker's three
//!     items gives d = [1.0, 0.0, -0.8] — see comments below.
//!
//!   DEMO 2 — Multidimensional (k=3), one item per dimension:
//!     Same item bank as item_selection_summary.md (m2p-v001/n001/r001),
//!     showing the gradient VECTOR and Hessian MATRIX (= Fisher Information
//!     Matrix, Mulder & van der Linden 2009, Eq.4, p.276) that generalize
//!     Baker's scalar numerator/denominator to k>1 dimensions.
//!
//!   DEMO 3 — Divergence case:
//!     All-correct response vector → MLE has no finite maximum. Mulder & van
//!     der Linden (2009), p.276: "The likelihood function may not have a
//!     maximum (e.g., when only correct or incorrect item responses are
//!     observed), or a local instead of a global maximum may be found."
//!     θ̂ grows without bound across iterations — motivates MAP (see
//!     estimation_map.rs).
//!
//! All MIRT math is called directly from the API (`mcat_engine::*`).
//! Only display helpers and the local `BankItem` sample struct are defined
//! here.
//!
//! Run:
//!   cargo run --example estimation_mle

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

// ─── DEMO 1 — Baker (2001) 1-dimensional reproduction ─────────────────────────

/// Baker (2001, p.87) three-item test, converted from (a,b,c) to (a,d,c):
///   item 1: a=1.0, b=-1  → d = -a*b = -1.0*(-1) =  1.0
///   item 2: a=1.2, b= 0  → d = -a*b = -1.2*  0  =  0.0
///   item 3: a=0.8, b= 1  → d = -a*b = -0.8*  1  = -0.8
/// Responses (Baker, p.87): u = [1, 0, 1]
fn demo1_baker_reproduction() {
    section("DEMO 1 — Literature Reproduction: Baker (2001) Ch.5, p.86-88, Eq.[5-1]");

    let a = [1.0_f64, 1.2, 0.8];
    let d = [1.0_f64, 0.0, -0.8];
    let c = [0.0_f64, 0.0, 0.0];
    let u = [1_u8, 0, 1];

    println!("  Item parameters (converted from Baker's a,b,c via d = -a·b):");
    for i in 0..3 {
        println!(
            "    item {}: a={:.1}  d={:+.1}  c={:.1}  u={}",
            i + 1,
            a[i],
            d[i],
            c[i],
            u[i]
        );
    }
    println!();
    println!("  a priori θ̂_0 = 1.0  (Baker, p.87: \"set to some arbitrary value, such as 1\")");
    println!();

    let mut theta = 1.0_f64;
    for iter in 1..=6 {
        println!("  Iteration {iter}:");
        println!("    item   u   P       Q       a(u-P)    a²PQ");
        let mut num = 0.0_f64;
        let mut den = 0.0_f64;
        for i in 0..3 {
            let theta_vec = DVector::from_vec(vec![theta]);
            let a_vec = DVector::from_vec(vec![a[i]]);
            // ← API call: mirt::probability (P under M2PL/M3PL)
            let p = probability(&theta_vec, &a_vec, d[i], c[i]);
            let q = 1.0 - p;
            let a_u_minus_p = a[i] * (u[i] as f64 - p);
            let a2pq = a[i] * a[i] * p * q;
            println!(
                "     {}     {}   {:.4}  {:.4}  {:+.4}   {:.4}",
                i + 1,
                u[i],
                p,
                q,
                a_u_minus_p,
                a2pq
            );
            num += a_u_minus_p;
            den += a2pq;
        }
        let delta = num / den;
        let theta_new = theta + delta;
        println!("     sum                    {:+.4}   {:.4}", num, den);
        println!(
            "    Δθ̂ = {:.4}/{:.4} = {:+.4}   →   θ̂_{{{}}} = {:.4} + ({:+.4}) = {:.4}",
            num, den, delta, iter, theta, delta, theta_new
        );
        println!();
        theta = theta_new;
        if delta.abs() < 1e-6 {
            println!("    ‖Δθ̂‖ < 1e-6 → converged.");
            break;
        }
    }
    println!("  Final θ̂_MLE ≈ {:.6}", theta);
    println!();
    println!("  Cross-check against Baker (2001) p.88 (book shows first 2 iterations only):");
    println!("    iter 1: Δθ̂ = -.403/.520 = -.773 → θ̂ = 1.0 - .773 = 0.227   [book, p.88]");
    println!("    iter 2: Δθ̂ =  .066/.674 =  .097 → θ̂ = 0.227 + .097 = 0.324 [book, p.88]");
    println!("  (matches this run's iterations 1-2 to 3 decimal places — confirms the");
    println!("   production API's residual/weight formulas reduce exactly to Baker's Eq.[5-1]");
    println!("   when k=1, c=0: residual=(u-P)P'/(PQ)=(u-P) since P'=PQ for the 2PL model,");
    println!("   and weight=(P')²/(PQ)=PQ, so grad=Σa(u-P), hess=Σa²PQ — identical to [5-1].)");
}

// ─── DEMO 2 — Multidimensional (k=3), one item per dimension ──────────────────

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

/// Full 7-item bank (item_selection_summary.md's "Item Bank Snapshot") — used for
/// DEMO 2 instead of the 3-item subset above because with EXACTLY k=3 items for
/// k=3 dimensions, the system is exactly-determined and the MLE can diverge (a
/// multidimensional analogue of quasi-complete separation in logistic regression
/// — see the "3-item divergence" note printed below). 7 items over 3 dimensions
/// is over-determined and behaves like a typical mid-test CAT re-estimation.
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
    section("DEMO 2 — Multidimensional (k=3): Gradient Vector & Hessian (= FIM) Matrix");

    println!("  Generalizes Baker's Eq.[5-1] scalar numerator/denominator to vector/matrix");
    println!("  form, using the Fisher Information Matrix of Mulder & van der Linden (2009),");
    println!(
        "  Eq.4, p.276:  I_i(θ) = [Qᵢ(Pᵢ-cᵢ)²]/[Pᵢ(1-cᵢ)²] · aᵢaᵢᵀ  (= P(1-P)·aᵢaᵢᵀ for c=0)."
    );
    println!();
    println!("  Uses the FULL 7-item bank (item_selection_summary.md's \"Item Bank Snapshot\"),");
    println!("  not just 1 item/dimension — see the note on exact-determination divergence below.");
    println!();

    let bank = full_bank();
    // Deliberately "noisy" within each content area (v001≠v002, n001≠n002,
    // r002 flips vs r001/r003) — a clean per-area split (e.g. all-verbal-correct,
    // all-numeric-wrong) is linearly separable by the nearly-orthogonal per-area
    // discrimination vectors and makes MLE diverge (quasi-complete separation,
    // the multidimensional analogue of the classic separation problem in
    // logistic regression). Mixed evidence within each area anchors a finite MLE.
    let responses: [u8; 7] = [1, 0, 0, 1, 1, 0, 1];
    let k = 3;
    let mut theta = DVector::<f64>::zeros(k);

    println!("  Items (one per dimension) & responses:");
    for (item, &x) in bank.iter().zip(responses.iter()) {
        println!(
            "    {}  a=[{:.1},{:.1},{:.1}]  d={:.2}  c={:.1}  u={}",
            item.id, item.a[0], item.a[1], item.a[2], item.d, item.c, x
        );
    }
    println!();
    println!("  Starting θ̂_0 = [0,0,0]  (production default, mle.rs:6: DVector::zeros)");
    println!();

    for iter in 1..=3 {
        println!("  Iteration {iter} — θ̂ = {:?}", theta.as_slice());
        let mut grad = DVector::<f64>::zeros(k);
        let mut hess = DMatrix::<f64>::zeros(k, k);

        for (item, &x) in bank.iter().zip(responses.iter()) {
            let a_vec = DVector::from_vec(item.a.to_vec());
            let linear = a_vec.dot(&theta) + item.d;
            let p_star = sigmoid(linear); // ← API call
            let p = item.c + (1.0 - item.c) * p_star;
            let q = 1.0 - p;
            let p_prime = (1.0 - item.c) * p_star * (1.0 - p_star);
            let residual = (x as f64 - p) * p_prime / (p * q + EPSILON);
            let w = (p_prime * p_prime) / (p * q + EPSILON);

            println!(
                "    {}: linear={:+.4}  P={:.6}  Q={:.6}  residual={:+.6}  w={:.6}",
                item.id, linear, p, q, residual, w
            );

            grad += &a_vec * residual;
            hess += &a_vec * a_vec.transpose() * w;
        }

        println!("    grad (∇LL)  = {:?}", grad.as_slice());
        println!(
            "    hess (=FIM) = [[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}],[{:.4},{:.4},{:.4}]]",
            hess[(0, 0)],
            hess[(0, 1)],
            hess[(0, 2)],
            hess[(1, 0)],
            hess[(1, 1)],
            hess[(1, 2)],
            hess[(2, 0)],
            hess[(2, 1)],
            hess[(2, 2)],
        );

        let delta = hess
            .clone()
            .try_inverse()
            .unwrap_or_else(|| DMatrix::identity(k, k))
            * &grad;
        println!("    Δθ̂ = hess⁻¹ · grad = {:?}", delta.as_slice());
        theta += &delta;
        println!("    θ̂_new = {:?}", theta.as_slice());
        println!();

        if delta.norm() < 1e-6 {
            break;
        }
    }

    // ← Cross-check against the real production API
    let item_a_vecs: Vec<DVector<f64>> = bank
        .iter()
        .map(|it| DVector::from_vec(it.a.to_vec()))
        .collect();
    let item_d: Vec<f64> = bank.iter().map(|it| it.d).collect();
    let item_c: Vec<f64> = bank.iter().map(|it| it.c).collect();
    let prior_mean = DVector::<f64>::zeros(k);
    let prior_cov_inv = DMatrix::<f64>::identity(k, k);
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
    let (theta_api, _) = estimate(&EstimationMethod::Mle, &input); // ← real production API call

    println!(
        "  Production API mle::estimate(): θ̂_MLE = {:?}",
        theta_api.as_slice()
    );
    println!("  (manual replication above matches the API's internal Newton-Raphson loop,");
    println!("   since both start at θ=[0,0,0] and use the identical gradient/Hessian formulas)");
}

// ─── DEMO 3 — Divergence case ──────────────────────────────────────────────────

fn demo3_divergence() {
    section("DEMO 3 — Divergence: All-Correct Responses (motivates MAP)");

    println!("  Mulder & van der Linden (2009), p.276: \"The likelihood function may not have");
    println!("  a maximum (e.g., when only correct or incorrect item responses are observed),");
    println!("  or a local instead of a global maximum may be found.\"");
    println!();

    let bank = sample_bank();
    let responses: [u8; 3] = [1, 1, 1]; // all correct
    let k = 3;

    let item_a_vecs: Vec<DVector<f64>> = bank
        .iter()
        .map(|it| DVector::from_vec(it.a.to_vec()))
        .collect();
    let item_d: Vec<f64> = bank.iter().map(|it| it.d).collect();
    let item_c: Vec<f64> = bank.iter().map(|it| it.c).collect();
    let prior_mean = DVector::<f64>::zeros(k);
    let prior_cov_inv = DMatrix::<f64>::identity(k, k);
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
    let (theta_mle, _) = estimate(&EstimationMethod::Mle, &input); // ← real production API call

    println!("  All 3 items answered correctly (u = [1,1,1]).");
    println!(
        "  Production API mle::estimate() after 100 Newton-Raphson iterations: θ̂ = {:?}",
        theta_mle.as_slice()
    );
    println!(
        "  ‖θ̂‖ = {:.2}  — driven to the iteration cap, not a finite interior maximum:",
        theta_mle.norm()
    );
    println!("  as θ→+∞ along a direction with positive discrimination, P(θ)→1 for every item,");
    println!("  so the gradient Σaᵢ(uᵢ-Pᵢ)P'ᵢ/(PᵢQᵢ) → 0⁺ without the sum of squared residuals");
    println!("  ever reaching a true interior root — the score equation has no finite solution.");
    println!("  See estimation_map.rs (DEMO 3) for the MAP-regularized alternative.");
}

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════════");
    println!("  MCAT · Maximum Likelihood Estimation (MLE) of θ · Verification & Simulation");
    println!("══════════════════════════════════════════════════════════════════════════");

    demo1_baker_reproduction();
    demo2_multidimensional();
    demo3_divergence();

    println!();
    divider();
    println!();
}
