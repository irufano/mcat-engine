//! A-Optimal Item Selection — Round 4 Simulation
//!
//! Demonstrates the A-optimal item-selection pipeline for the 4th item
//! (after 3 items already administered: m2p-r001, m2p-v001, m2p-n001).
//!
//! A-optimal criterion:  argmin_i  tr( (cum_FIM + item_FIM_i)^{-1} )
//! API score (negated):  score_i   = -tr(inv(updated_FIM_i))   [higher = better]
//! When updated_FIM is singular → score = f64::NEG_INFINITY    [degenerate]
//!
//! Degeneracy behaviour:
//!   Round 1 (cum_FIM = 0):       updated = rank-1  → singular → all -∞
//!   Round 2 (cum_FIM = rank-1):  updated ≤ rank-2  → singular → all -∞
//!   Round 3+ (cum_FIM ≥ rank-2): updated can reach rank-3 → invertible → valid
//!
//! Contrast with D-optimal which returns det=0 (tiebreak via trace) instead of -∞.
//! A-optimal has no built-in fallback for the degenerate case.
//!
//! All MIRT math and selection/exposure functions are called directly from
//! the API (`mcat_engine::*`).  Only display helpers and the local
//! `BankItem` sample struct are defined here.
//!
//! Run:
//!   cargo run --example item_selection_a_optimal
//!
//! Tweak `INITIAL_THETA` or the item bank in `sample_bank()` to experiment.

use mcat_engine::{
    exposure::{ExposureMethod, is_eligible},
    mirt::{item_fim, probability, sigmoid},
    selection::{SelectionContext, SelectionMethod, score_item},
};
use nalgebra::{DMatrix, DVector};

// --- Tunable parameters ---

/// Theta untuk item yang sudah di-administer di round sebelumnya.
/// Digunakan saat menghitung cum_FIM secara programmatik.
///
/// Round 1/2/3 → [0.0, 0.0, 0.0]  (prior, belum ada re-estimasi)
/// Round 4     → [1.0, -0.5, 0.5] (hasil MAP/MLE setelah 3 item)
// const HIST_THETA: [f64; 3] = [0.0, 0.0, 0.0];   // Round 1 / 2 / 3
const HIST_THETA: [f64; 3] = [1.0, -0.5, 0.5]; // Round 4

/// Theta untuk round saat ini (digunakan saat scoring kandidat).
///
/// Round 1/2/3 → [0.0, 0.0, 0.0]  (prior)
/// Round 4     → [1.0, -0.5, 0.5] (re-estimasi terbaru)
// const INITIAL_THETA: [f64; 3] = [0.0, 0.0, 0.0];   // Round 1 / 2 / 3
const INITIAL_THETA: [f64; 3] = [1.0, -0.5, 0.5]; // Round 4

/// Items already administered before this round.
/// cum_FIM dihitung otomatis dari daftar ini via item_fim() — tidak hardcode.
///
/// Round 1 → []                              (degenerate: cum_FIM = 0, semua -∞)
/// Round 2 → [r001]                          (degenerate: rank-1, semua -∞)
/// Round 3 → [r001, v001]                    (rank-2, mixed: numeric candidates finite)
/// Round 4 → [r001, v001, n001]              (rank-3, semua finite)
fn initial_administered() -> Vec<&'static str> {
    // Round 1: nothing administered yet
    // vec![]

    // Round 2: one item administered
    // vec!["m2p-r001"]

    // Round 3: two items administered
    // vec!["m2p-r001", "m2p-r001"]

    // Round 4: three items administered
    vec!["m2p-r001", "m2p-v001", "m2p-n001"]
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

fn print_matrix(m: &DMatrix<f64>, label: &str) {
    let k = m.nrows();
    println!("    {label}:");
    for r in 0..k {
        print!("      [");
        for c in 0..k {
            if c > 0 {
                print!("  ");
            }
            print!("{:8.4}", m[(r, c)]);
        }
        println!("  ]");
    }
}

// ─── Scored candidate (populated in Stage 2) ──────────────────────────────────

struct Scored<'a> {
    item: &'a BankItem,
    linear: f64,
    p_star: f64,
    p: f64,
    q: f64,
    w: f64,
    /// API score: -tr(inv(updated_FIM)).  Higher = better.
    /// f64::NEG_INFINITY when updated_FIM is singular (degenerate case).
    a_score: f64,
    /// Raw tr(inv(updated_FIM)).  Lower = better.
    /// f64::INFINITY when singular.
    trace_inv: f64,
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  MCAT · A-Optimal Item Selection · Round 4 Simulation");
    println!("══════════════════════════════════════════════════════════════════════");

    let k: usize = 3;
    let theta = DVector::from_vec(INITIAL_THETA.to_vec());
    let hist_theta = DVector::from_vec(HIST_THETA.to_vec());
    let administered = initial_administered();
    let bank = sample_bank();

    // cum_FIM dihitung dari item_fim() langsung — hindari rounding error dari hardcode.
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
        println!("    cum_FIM     = {k}×{k} zero matrix  (round 1 — A-optimal degenerate)");
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
    println!("    selection   = a_optimal → argmin_i tr((cum_FIM + item_FIM_i)⁻¹)");
    println!("    exposure    = sympson_hetter (sh_r_param used directly, per item)");
    if administered.is_empty() {
        println!("    administered= []  (none yet — A-optimal will be degenerate)");
    } else {
        println!("    administered= {:?}", administered);
    }

    // ---
    // STAGE 1 — Exposure gate
    // API call: mcat_engine::exposure::is_eligible
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
    // STAGE 2 — Compute item_FIM and A-Optimal score
    // API calls:
    //   mcat_engine::mirt::sigmoid
    //   mcat_engine::mirt::probability
    //   mcat_engine::mirt::item_fim
    //   mcat_engine::selection::score_item  (with SelectionMethod::AOptimal)
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 2 — Compute item_FIM and A-Optimal Score");

    println!("  APIs used per item:");
    println!("    mirt::sigmoid(linear)                    → P*");
    println!("    mirt::probability(θ,a,d,c)               → P  (derive Q, w)");
    println!("    mirt::item_fim(θ,a,d,c)                  → item_FIM");
    println!("    selection::score_item(AOptimal,ctx,&fim) → -tr(inv(cum_FIM + item_FIM))");
    println!();
    println!("  A-optimal criterion (src/mcat/selection/a_optimal.rs):");
    println!("    updated_FIM = cum_FIM + item_FIM");
    println!("    a_score     = -tr(updated_FIM⁻¹)   [higher = better]");
    println!("    ↕  equivalent: argmin_i tr(updated_FIM_i⁻¹)  [lower trace_inv = better]");
    println!();
    println!("  Interpretation of tr(FIM⁻¹):");
    println!("    FIM⁻¹ ≈ covariance matrix of θ̂  (Cramér-Rao lower bound)");
    println!("    diagonal of FIM⁻¹ = [Var(θ̂₁), Var(θ̂₂), Var(θ̂₃)]");
    println!("    tr(FIM⁻¹) = Var(θ̂₁) + Var(θ̂₂) + Var(θ̂₃) = total uncertainty");
    println!("    → A-optimal minimizes total uncertainty across all dimensions.");
    println!();
    println!("  When updated_FIM is singular (rank < k):");
    println!("    try_inverse() fails → a_score = f64::NEG_INFINITY");
    println!("    All candidates tie at -∞ → no meaningful discrimination.");
    println!("    This occurs at rounds 1–2 (and possibly round 3 depending on item directions).");
    println!();

    let ctx = SelectionContext {
        theta: &theta,
        cum_fim: &cum_fim,
        k,
        administered: administered.len(),
    };

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

        // — linear (step-by-step) ─────────────────────────────────────────────
        let dot: f64 = item
            .a
            .iter()
            .zip(INITIAL_THETA.iter())
            .map(|(a, t)| a * t)
            .sum();
        print!("    linear = ");
        for (i, (&ai, &ti)) in item.a.iter().zip(INITIAL_THETA.iter()).enumerate() {
            if i > 0 {
                print!(" + ");
            }
            print!("({:.1}×{:.1})", ai, ti);
        }
        let linear = a_vec.dot(&theta) + item.d;
        println!(" + {:.2}", item.d);
        println!("           = {:.4} + {:.2} = {:.4}", dot, item.d, linear);
        println!();

        // — P* via API: mirt::sigmoid ─────────────────────────────────────────
        let p_star = sigmoid(linear);
        println!("    P* = mirt::sigmoid({:.4})", linear);
        println!("       = 1 / (1 + e^({:.4}))", -linear);
        println!("       = 1 / (1 + {:.6})", (-linear).exp());
        println!("       = {:.6}", p_star);
        println!();

        // — P via API: mirt::probability ──────────────────────────────────────
        let p = probability(&theta, &a_vec, item.d, item.c);
        let q = 1.0 - p;
        let w = p * q;

        if item.c == 0.0 {
            println!("    P  = mirt::probability(θ,a,d,c=0) = P*  (M2PL)");
            println!("       = {:.6}", p);
        } else {
            println!("    P  = mirt::probability(θ,a,d,c)");
            println!(
                "       = {:.2} + (1−{:.2})×{:.6} = {:.6}",
                item.c, item.c, p_star, p
            );
        }
        println!("    Q  = 1 − P = 1 − {:.6} = {:.6}", p, q);
        println!("    w  = P×Q   = {:.6} × {:.6} = {:.6}", p, q, w);
        println!();

        // — item_FIM via API: mirt::item_fim ──────────────────────────────────
        let fim = item_fim(&theta, &a_vec, item.d, item.c);
        println!("    FIM = mirt::item_fim(θ,a,d,c)  =  w × (a × aᵀ)");
        println!(
            "        = {:.6} × outer([{:.1}, {:.1}, {:.1}])",
            w, item.a[0], item.a[1], item.a[2]
        );
        println!();
        print_matrix(&fim, "item_FIM");
        println!();

        // — updated_FIM = cum_FIM + item_FIM ──────────────────────────────────
        let updated = &cum_fim + &fim;
        print_matrix(&updated, "updated_FIM  = cum_FIM + item_FIM");
        println!();

        // — A-optimal score via API: selection::score_item ────────────────────
        // score_item internally computes -tr(inv(cum_fim + item_fim))
        let a_score = score_item(
            &SelectionMethod::AOptimal,
            &ctx,
            &fim,
            &a_vec,
            item.d,
            item.c,
        );
        let trace_inv = -a_score;

        println!("    a_score = selection::score_item(AOptimal, ctx, &fim)");
        println!("            = -tr(updated_FIM⁻¹)");
        if a_score.is_finite() {
            println!("            = -({:.6})", trace_inv);
            println!("            = {:.6}", a_score);
            println!();
            println!(
                "    tr(inv) = {:.6}  ← minimize (lower = less total variance)",
                trace_inv
            );
            println!(
                "    a_score = {:.6}  ← API score (higher = better)",
                a_score
            );
        } else {
            println!("            = −∞  (updated_FIM is singular — degenerate case)");
            println!("            Reason: cum_FIM rank < k; updated_FIM not full-rank.");
        }

        scored.push(Scored {
            item,
            linear,
            p_star,
            p,
            q,
            w,
            a_score,
            trace_inv: if a_score.is_finite() {
                trace_inv
            } else {
                f64::INFINITY
            },
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 3 — Rank and select
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 3 — Ranking & argmax (Item Selection)");

    let all_degenerate = scored.iter().all(|s| !s.a_score.is_finite());

    if all_degenerate {
        println!("  All A-scores = −∞  (updated_FIM singular for all candidates).");
        println!("  Reason: cum_FIM has rank < k; no candidate can bring it to full rank.");
        println!();
        println!("  A-optimal cannot discriminate between items in this round.");
        println!("  The API returns -∞ for all → selection is effectively iterator order.");
        println!("  Workaround: use D-optimal for early rounds (falls back to trace tiebreak).");
    } else {
        println!("  Ranking by a_score = -tr(inv(updated_FIM))  [higher = better]");
        println!("  Equivalently: rank by tr(inv) ascending       [lower trace_inv = better]");
        println!();
        println!("  A-optimal selects the item that, when added to the current information,");
        println!("  minimises the sum of variances across all θ dimensions.");
        println!("  It targets the weakest direction in the current cum_FIM.");
    }
    println!();

    scored.sort_by(|a, b| {
        b.a_score
            .partial_cmp(&a.a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    println!("  Rank  ID           Area          a_score           tr(inv)");
    println!("  ──────────────────────────────────────────────────────────────────");
    for (rank, s) in scored.iter().enumerate() {
        let marker = if rank == 0 { " ← SELECTED" } else { "" };
        let a_str = if s.a_score.is_finite() {
            format!("{:12.6}", s.a_score)
        } else {
            format!("{:>12}", "−∞")
        };
        let t_str = if s.trace_inv.is_finite() {
            format!("{:10.6}", s.trace_inv)
        } else {
            format!("{:>10}", "+∞")
        };
        println!(
            "    {:2}    {:12} {:12}  {}    {}{}",
            rank + 1,
            s.item.id,
            s.item.content_area,
            a_str,
            t_str,
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
        println!("    linear       = {:.4}  (= a·θ̂ + d)", winner.linear);
        println!("    P*           = {:.6}  [mirt::sigmoid]", winner.p_star);
        println!("    P            = {:.6}  [mirt::probability]", winner.p);
        println!("    Q            = {:.6}", winner.q);
        println!("    w = P×Q      = {:.6}", winner.w);
        if winner.a_score.is_finite() {
            println!(
                "    a_score      = {:.6}  [selection::score_item(AOptimal)]",
                winner.a_score
            );
            println!(
                "    tr(inv)      = {:.6}  ← lowest among all candidates",
                winner.trace_inv
            );
            println!();
            println!("  Why {} wins:", w.id);
            println!("    Adding this item to cum_FIM produces the updated_FIM with the");
            println!("    smallest trace(inv) = smallest total variance across all dimensions.");
            println!("    A-optimal targets the dimension where cum_FIM is currently weakest.");
        } else {
            println!("    a_score      = −∞  (degenerate — see notes above)");
        }
        println!();
        println!("  Next step: present item, record response x ∈ {{0,1}}, then:");
        println!("    cum_FIM ← cum_FIM + mirt::item_fim(θ̂)");
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
