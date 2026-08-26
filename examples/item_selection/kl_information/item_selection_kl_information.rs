//! KL Information (Kullback-Leibler Information) Item Selection — Round 1 Simulation
//!
//! Demonstrates the KL-information item-selection pipeline for the 1st item
//! (before any items have been administered).
//!
//! Original formula (Chang & Ying, 1996 — "A Global Information Approach to
//! Computerized Adaptive Testing", Applied Psychological Measurement, 20(3), 213-229;
//! reproduced verbatim as Eq. 9-10 in Han, K.T. (2018), J Educ Eval Health Prof, 15:7,
//! https://doi.org/10.3352/jeehp.2018.15.7, p.6, open access):
//!
//!   K_i(z ‖ ẑ)  = P_i(ẑ)·ln[P_i(ẑ)/P_i(z)] + [1-P_i(ẑ)]·ln{[1-P_i(ẑ)]/[1-P_i(z)]}
//!   K̄_i(θ̂)     = ∫_{ẑ-δ}^{ẑ+δ} K_i(z ‖ ẑ) dz ,   δ = 3 / sqrt(administered + 1)
//!
//! where z = a·θ + d is the compensatory MIRT composite score. This is the EXACT
//! Kullback-Leibler divergence between the two Bernoulli response distributions at ẑ
//! and z (Gibbs' inequality: always ≥ 0) — NOT the 0.5×tr(item_FIM) Taylor/Fisher-information
//! approximation used in earlier revisions of this module. See kl_information_notes.md for
//! the full theoretical derivation, numeric verification and citations.
//!
//! Key properties:
//!   - Depends on θ only through the item's own composite score z = a·θ + d — the
//!     multidimensional integral over θ collapses exactly to this 1-D integral (proof in notes).
//!   - The integration interval δ shrinks as more items are administered (m), reflecting
//!     the increasing precision of θ̂ — unlike D-optimal/A-optimal which need cum_FIM.
//!   - No degenerate case: the integral is always finite and non-negative.
//!
//! All MIRT math and selection/exposure functions are called directly from
//! the API (`mcat_engine::*`).  Only display helpers and the local
//! `BankItem` sample struct are defined here.
//!
//! Run:
//!   cargo run --example item_selection_kl_information
//!
//! Tweak `INITIAL_THETA` or the item bank in `sample_bank()` to experiment.

use mcat_engine::{
    exposure::{ExposureMethod, is_eligible},
    mirt::{EPSILON, item_fim, probability, sigmoid},
    selection::{SelectionContext, SelectionMethod, score_item},
};
use nalgebra::{DMatrix, DVector};

// --- Tunable parameters ---

/// Theta untuk item yang sudah di-administer di round sebelumnya.
/// Digunakan saat menghitung cum_FIM secara programmatik.
///
/// Round 1 → [0.0, 0.0, 0.0]  (prior, belum ada re-estimasi)
/// Round 4 → [1.0, -0.5, 0.5] (hasil MAP/MLE setelah 3 item)
const HIST_THETA: [f64; 3] = [0.0, 0.0, 0.0]; // Round 1
// const HIST_THETA: [f64; 3] = [1.0, -0.5, 0.5]; // Round 4

/// Theta untuk round saat ini (digunakan saat scoring kandidat).
///
/// Round 1 → [0.0, 0.0, 0.0]  (prior)
/// Round 4 → [1.0, -0.5, 0.5] (re-estimasi terbaru)
const INITIAL_THETA: [f64; 3] = [0.0, 0.0, 0.0]; // Round 1
// const INITIAL_THETA: [f64; 3] = [1.0, -0.5, 0.5]; // Round 4

/// Items already administered before this round.
/// cum_FIM dihitung otomatis dari daftar ini via item_fim().
///
/// Round 1 → []                         (no items administered yet)
/// Round 4 → [r001, v001, n001]         (three items, full-rank cum_FIM)
fn initial_administered() -> Vec<&'static str> {
    // Round 1: nothing administered yet
    vec![]

    // Round 4: three items administered
    // vec!["m2p-r001", "m2p-v001", "m2p-n001"]
}

// ─── Sample item bank ─────────────────────────────────────────────────────────

struct BankItem {
    id: &'static str,
    content_area: &'static str,
    /// Discrimination vector [verbal, numeric, reasoning].
    a: [f64; 3],
    /// Intercept (positive d → item easier than average examinee).
    d: f64,
    /// Guessing parameter; 0.0 for M2PL.
    c: f64,
    is_active: bool,
    /// Fraction of past sessions where this item was administered (0 = fresh).
    exposure_rate: f64,
    /// Sympson-Hetter r_i: floor probability used by the exposure gate.
    sh_r_param: f64,
}

fn sample_bank() -> Vec<BankItem> {
    vec![
        BankItem {
            id: "m2p-v001",
            content_area: "verbal",
            a: [1.9, 0.2, 0.3],
            d: 0.40,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-v002",
            content_area: "verbal",
            a: [1.7, 0.2, 0.2],
            d: 0.10,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-n001",
            content_area: "numeric",
            a: [0.3, 1.9, 0.4],
            d: 0.80,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-n002",
            content_area: "numeric",
            a: [0.3, 1.8, 0.4],
            d: 0.50,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-r001",
            content_area: "reasoning",
            a: [0.5, 0.4, 2.0],
            d: 0.30,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-r002",
            content_area: "reasoning",
            a: [0.3, 0.8, 1.9],
            d: 0.60,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
        BankItem {
            id: "m2p-r003",
            content_area: "reasoning",
            a: [0.4, 0.3, 1.8],
            d: 0.70,
            c: 0.0,
            is_active: true,
            exposure_rate: 0.00,
            sh_r_param: 1.0,
        },
    ]
}

// ─── Local-only helpers ───────────────────────────────────────────────────────

fn divider() {
    println!("{}", "─".repeat(70));
}

fn section(title: &str) {
    println!();
    divider();
    println!("  {title}");
    divider();
    println!();
}

// ─── Scored candidate (populated in Stage 2) ──────────────────────────────────

struct Scored<'a> {
    item: &'a BankItem,
    /// z_hat = a·θ̂ + d — the item's composite score at the current ability estimate.
    z_hat: f64,
    p_hat: f64,
    /// δ = 3 / sqrt(administered + 1) — half-width of the integration interval.
    delta: f64,
    lower: f64,
    upper: f64,
    /// K_i(z ‖ ẑ) evaluated at the lower/upper bounds, shown for illustration.
    kl_at_lower: f64,
    kl_at_upper: f64,
    /// API score: K̄_i(θ̂) = ∫_{ẑ-δ}^{ẑ+δ} K_i(z‖ẑ) dz.  Higher = better.
    kl_score: f64,
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  MCAT · KL Information Item Selection · Round 1 Simulation");
    println!("══════════════════════════════════════════════════════════════════════");

    let k: usize = 3;
    let theta = DVector::from_vec(INITIAL_THETA.to_vec());
    let hist_theta = DVector::from_vec(HIST_THETA.to_vec());
    let administered = initial_administered();
    let bank = sample_bank();

    // cum_FIM dihitung dari item_fim() langsung dari administered items.
    // Catatan: cum_FIM tidak digunakan dalam KL score, tapi tetap dihitung
    // karena SelectionContext memerlukannya (untuk metode lain dalam dispatcher).
    let cum_fim = {
        let mut fim = DMatrix::zeros(k, k);
        for id in &administered {
            if let Some(item) = bank.iter().find(|b| b.id == *id) {
                let a = DVector::from_vec(item.a.to_vec());
                fim += item_fim(&hist_theta, &a, item.d, item.c);
            }
        }
        fim
    };

    let cum_fim_is_zero = cum_fim.iter().all(|&v| v.abs() < 1e-12);

    println!();
    println!("  Session state at the start of this round:");
    println!("    θ̂           = {:?}", theta.as_slice());
    println!("    dimensions  = {k}   [verbal, numeric, reasoning]");
    if cum_fim_is_zero {
        println!("    cum_FIM     = {k}×{k} zero matrix  (round 1 — KL score not affected)");
    } else {
        println!(
            "    cum_FIM     = {k}×{k} matrix  (accumulated from {} administered items):",
            administered.len()
        );
        for r in 0..k {
            print!("                  [");
            for c in 0..k {
                if c > 0 {
                    print!("  ");
                }
                print!("{:8.4}", cum_fim[(r, c)]);
            }
            println!("  ]");
        }
    }
    println!("    bank size   = {} items", bank.len());
    println!("    selection   = kl_information → argmax_i ∫_{{ẑ-δ}}^{{ẑ+δ}} K_i(z‖ẑ) dz");
    println!("    exposure    = sympson_hetter (sh_r_param used directly, per item)");
    if administered.is_empty() {
        println!("    administered= []  (none yet)");
    } else {
        println!("    administered= {:?}", administered);
    }

    println!();
    println!("  ★ KL-information score does NOT depend on cum_FIM.");
    println!("    K_i(z‖ẑ)  = P_i(ẑ)·ln[P_i(ẑ)/P_i(z)] + [1-P_i(ẑ)]·ln{{[1-P_i(ẑ)]/[1-P_i(z)]}}");
    println!("    score_i   = ∫_{{ẑ-δ}}^{{ẑ+δ}} K_i(z‖ẑ) dz,  δ = 3/√(administered+1)");
    println!("    All scores are always finite and ≥ 0 (Gibbs' inequality) → no degenerate case.");

    // ---
    // STAGE 1 — Exposure gate
    // ---
    section("STAGE 1 — Exposure Gate (Sympson-Hetter)");

    println!("  API: mcat_engine::exposure::is_eligible(method, sh_r_param)");
    println!();
    println!("  Gate formula (src/mcat/exposure/sympson_hetter.rs) — matches Sympson &");
    println!("  Hetter (1985) / mirtCAT's exposure_type == \"SH\" directly:");
    println!("    eligible = gen_bool(sh_r_param)   [sh_r_param = K_i = P(A|S), pre-calibrated]");
    println!();

    let mut candidates: Vec<&BankItem> = vec![];

    for item in &bank {
        print!("  {:12}  [{:9}]  ", item.id, item.content_area);

        if !item.is_active {
            println!("SKIP  (inactive)");
            continue;
        }
        if administered.contains(&item.id) {
            println!("SKIP  (already administered)");
            continue;
        }

        // ← actual API call (src/mcat/exposure/mod.rs → sympson_hetter.rs)
        let eligible = is_eligible(&ExposureMethod::SympsonHetter, item.sh_r_param);

        if eligible {
            println!(
                "PASS   exposure={:.2}  sh_r_param={:.2}",
                item.exposure_rate, item.sh_r_param
            );
            candidates.push(item);
        } else {
            println!(
                "BLOCK  exposure={:.2}  sh_r_param={:.2}  (blocked)",
                item.exposure_rate, item.sh_r_param
            );
        }
    }

    println!();
    println!(
        "  → {} / {} items passed the gate.",
        candidates.len(),
        bank.len()
    );

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 2 — Compute the composite score z, the integration interval, and the
    // global KL-information score.
    // API calls:
    //   mcat_engine::mirt::sigmoid
    //   mcat_engine::mirt::probability
    //   mcat_engine::mirt::item_fim          (still needed by SelectionContext/score_item
    //                                             dispatcher, but NOT used by KL itself)
    //   mcat_engine::selection::score_item   (with SelectionMethod::KlInformation)
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 2 — Compute the Global KL-Information Score");

    println!("  APIs used per item:");
    println!("    mirt::sigmoid(z)                              → P_i(z)");
    println!("    mirt::probability(θ,a,d,c)                    → P_i(ẑ)  (M2PL/M3PL)");
    println!("    selection::score_item(KlInformation,ctx,...)   → K̄_i(θ̂)");
    println!();
    println!("  Original formula (Chang & Ying, 1996; Han, 2018, Eq. 9-10, p.6, open access):");
    println!("    K_i(z‖ẑ) = P_i(ẑ)·ln[P_i(ẑ)/P_i(z)] + [1-P_i(ẑ)]·ln{{[1-P_i(ẑ)]/[1-P_i(z)]}}");
    println!("    K̄_i(θ̂)  = ∫_{{ẑ-δ}}^{{ẑ+δ}} K_i(z‖ẑ) dz ,   δ = 3/√(administered+1)");
    println!();
    println!("  z = a·θ̂ + d is the compensatory composite score; P_i depends on θ only");
    println!("  through z, so the multidimensional integral over θ collapses to this 1-D");
    println!("  integral over z (proof: kl_information_notes.md, 'Reduksi Multidimensional').");
    println!();
    println!("  score_item() computes updated = cum_FIM + item_FIM for the OTHER selection");
    println!("  methods, but KL information ignores it entirely — no cum_FIM dependency.");
    println!();

    let ctx = SelectionContext {
        theta: &theta,
        cum_fim: &cum_fim,
        k,
        administered: administered.len(),
    };
    let delta = mcat_engine::selection::kl_information::INTERVAL_C
        / ((administered.len() + 1) as f64).sqrt();

    let mut scored: Vec<Scored> = vec![];

    for item in &candidates {
        divider();
        println!("  Item: {}  [{}]", item.id, item.content_area);
        println!(
            "    a = [{:.1}, {:.1}, {:.1}]   d = {:.2}   c = {:.1}",
            item.a[0], item.a[1], item.a[2], item.d, item.c
        );
        println!();

        let a_vec = DVector::from_vec(item.a.to_vec());

        // — z_hat (step-by-step) ──────────────────────────────────────────────
        let dot: f64 = item
            .a
            .iter()
            .zip(INITIAL_THETA.iter())
            .map(|(a, t)| a * t)
            .sum();
        print!("    z_hat = ");
        for (i, (&ai, &ti)) in item.a.iter().zip(INITIAL_THETA.iter()).enumerate() {
            if i > 0 {
                print!(" + ");
            }
            print!("({:.1}×{:.1})", ai, ti);
        }
        let z_hat = a_vec.dot(&theta) + item.d;
        println!(" + {:.2}", item.d);
        println!("          = {:.4} + {:.2} = {:.4}", dot, item.d, z_hat);
        println!();

        // — P* via API: mirt::sigmoid ─────────────────────────────────────────
        let p_star = sigmoid(z_hat);
        println!("    P* = mirt::sigmoid({:.4})", z_hat);
        println!("       = 1 / (1 + e^({:.4}))", -z_hat);
        println!("       = 1 / (1 + {:.6})", (-z_hat).exp());
        println!("       = {:.6}", p_star);
        println!();

        // — P_i(ẑ) via API: mirt::probability ─────────────────────────────────
        let p_hat = probability(&theta, &a_vec, item.d, item.c);

        if item.c == 0.0 {
            println!("    P_i(ẑ) = mirt::probability(θ,a,d,c=0) = P*  (M2PL)");
            println!("           = {:.6}", p_hat);
        } else {
            println!("    P_i(ẑ) = mirt::probability(θ,a,d,c)");
            println!(
                "           = {:.2} + (1−{:.2})×{:.6} = {:.6}",
                item.c, item.c, p_star, p_hat
            );
        }
        println!();

        // — integration interval [ẑ-δ, ẑ+δ] ────────────────────────────────────
        let lower = z_hat - delta;
        let upper = z_hat + delta;
        println!(
            "    δ = {:.4} / √({}+1) = {:.4}",
            mcat_engine::selection::kl_information::INTERVAL_C,
            administered.len(),
            delta
        );
        println!(
            "    [ẑ-δ, ẑ+δ] = [{:.4}, {:.4}]  (integration bounds in z-space)",
            lower, upper
        );
        println!();

        // — pointwise KL at the bounds, for illustration ───────────────────────
        let kl_at = |z: f64| -> f64 {
            let p = item.c + (1.0 - item.c) * sigmoid(z);
            let p = p.clamp(EPSILON, 1.0 - EPSILON);
            let ph = p_hat.clamp(EPSILON, 1.0 - EPSILON);
            ph * (ph / p).ln() + (1.0 - ph) * ((1.0 - ph) / (1.0 - p)).ln()
        };
        let kl_at_lower = kl_at(lower);
        let kl_at_upper = kl_at(upper);
        println!(
            "    K_i(ẑ-δ ‖ ẑ) = {:.6}      K_i(ẑ+δ ‖ ẑ) = {:.6}   (K_i(ẑ‖ẑ)=0 at the center)",
            kl_at_lower, kl_at_upper
        );
        println!();

        // — KL-information score via API: selection::score_item ────────────────
        // score_item internally calls kl_information::score(), which numerically
        // integrates K_i(z‖ẑ) over [ẑ-δ, ẑ+δ] via composite Simpson's rule.
        // item_fim/cum_FIM are passed only because score_item()'s signature is
        // shared across all 4 selection methods — KL itself ignores both.
        let fim = item_fim(&theta, &a_vec, item.d, item.c);
        let kl_score = score_item(
            &SelectionMethod::KlInformation,
            &ctx,
            &fim,
            &a_vec,
            item.d,
            item.c,
        );

        println!("    kl_score = selection::score_item(KlInformation, ctx, &fim, &a, d, c)");
        println!("             = ∫_{{ẑ-δ}}^{{ẑ+δ}} K_i(z‖ẑ) dz   (composite Simpson's rule)");
        println!("             = {:.6}   ← higher = better", kl_score);

        scored.push(Scored {
            item,
            z_hat,
            p_hat,
            delta,
            lower,
            upper,
            kl_at_lower,
            kl_at_upper,
            kl_score,
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 3 — Rank and select
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 3 — Ranking & argmax (Item Selection)");

    println!("  Ranking by kl_score = ∫_{{ẑ-δ}}^{{ẑ+δ}} K_i(z‖ẑ) dz  [higher = better]");
    println!();
    println!("  Item menang jika K_i(z‖ẑ) tumbuh cepat begitu z bergerak menjauhi ẑ dalam");
    println!("  interval [ẑ-δ, ẑ+δ] — yaitu item yang sangat diskriminatif (a besar) di");
    println!("  sekitar θ̂ saat ini.");
    println!();
    println!("  KL-information tidak mempertimbangkan cum_FIM →");
    println!("  tidak adaptif terhadap informasi yang sudah terkumpul.");
    println!();

    scored.sort_by(|a, b| {
        b.kl_score
            .partial_cmp(&a.kl_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("  Rank  ID           Area          kl_score        z_hat       δ");
    println!("  ──────────────────────────────────────────────────────────────────");
    for (rank, s) in scored.iter().enumerate() {
        let marker = if rank == 0 { " ← SELECTED" } else { "" };
        println!(
            "    {:2}    {:12} {:12}  {:10.6}    {:8.4}  {:6.4}{}",
            rank + 1,
            s.item.id,
            s.item.content_area,
            s.kl_score,
            s.z_hat,
            s.delta,
            marker
        );
    }
    println!();

    // ── Result ────────────────────────────────────────────────────────────────
    divider();
    if let Some(winner) = scored.first() {
        let w = winner.item;
        println!();
        println!("  SELECTED ITEM: {}  [{}]", w.id, w.content_area);
        println!();
        println!(
            "    a            = [{:.1}, {:.1}, {:.1}]  (verbal, numeric, reasoning)",
            w.a[0], w.a[1], w.a[2]
        );
        println!("    d            = {:.2}", w.d);
        println!("    z_hat        = {:.4}  (= a·θ̂ + d)", winner.z_hat);
        println!(
            "    P_i(ẑ)       = {:.6}  [mirt::probability]",
            winner.p_hat
        );
        println!(
            "    δ            = {:.4}  ([ẑ-δ, ẑ+δ] = [{:.4}, {:.4}])",
            winner.delta, winner.lower, winner.upper
        );
        println!(
            "    K_i at bounds = {:.6} (lower), {:.6} (upper)",
            winner.kl_at_lower, winner.kl_at_upper
        );
        println!(
            "    kl_score     = {:.6}  [selection::score_item(KlInformation)]",
            winner.kl_score
        );
        println!();
        println!("  Why {} wins:", w.id);
        println!("    K_i(z‖ẑ) grows fastest across [ẑ-δ, ẑ+δ] for the item whose response curve");
        println!(
            "    is steepest (largest discrimination) near θ̂ — here {}.",
            w.id
        );
        println!();
        println!("  Next step: present item, record response x ∈ {{0,1}}, then:");
        println!("    cum_FIM ← cum_FIM + mirt::item_fim(θ̂)   (used by other selection methods)");
        println!("    θ̂      ← MAP/MLE re-estimate (all responses so far)");
        println!("    check stopping rule (n < n_min → CONTINUE)");
        println!();
    } else {
        println!();
        println!("  No item selected — all candidates blocked by exposure gate.");
        println!();
    }
    divider();
    println!();
}
