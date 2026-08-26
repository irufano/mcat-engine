//! D-Optimal Item Selection — Round 1 / Round 2 / Round 3 Simulation
//!
//! Walks through the complete item-selection pipeline described in
//! MCAT_EXPLANATION.md §9 "Putting It All Together — M2PL Full Example"
//! from session initialisation through Stage 3 (item selected, before
//! response & θ update).
//!
//! D-optimal criterion:  argmax_i  det(cum_FIM + item_FIM_i)
//! API score:             score_i   = det(updated_FIM)              [higher = better]
//! Degenerate case:       det = 0 for ALL candidates while updated_FIM is rank < k
//!                        (no -∞ like A-optimal — score_item() just returns 0.0).
//!
//! Degeneracy behaviour (see d_optimal_notes.md, "Catatan: Degenerasi di Round 1 dan Round 2"):
//!   Round 1 (cum_FIM = 0):       updated = rank-1  → det = 0 for all candidates
//!   Round 2 (cum_FIM = rank-1):  updated ≤ rank-2  → det = 0 for all candidates
//!   Round 3+ (cum_FIM = rank-2): updated can reach rank-3 → det > 0 for candidates whose
//!                        `a` is linearly independent from the two already-administered items
//!
//! No tiebreak when all det_scores are 0 — this mirrors the real production API
//! (`McatEngine::select_next_item`, src/mcat/engine.rs), which keeps whichever candidate it
//! encountered first and only replaces it on a *strictly greater* score (`score > best_score`).
//! When every candidate scores 0.0, the winner is therefore simply the first eligible item in
//! iteration order (`sample_bank()` order here) — an arbitrary, non-principled outcome, not a
//! deliberate secondary criterion. This script sorts by `det_score` alone with a stable sort,
//! which reproduces that exact "first-max-wins" behaviour for tied scores.
//!
//! All MIRT math and selection/exposure functions are called directly from
//! the API (`mcat_engine::*`).  Only display helpers and the local
//! `BankItem` sample struct are defined here.
//!
//! Run:
//!   cargo run --example item_selection_d_optimal
//!
//! Tweak `HIST_THETA` / `INITIAL_THETA` / `initial_administered()` to switch
//! between Round 1, Round 2, and Round 3 (see comments below), or edit the
//! item bank in `sample_bank()` to experiment.

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
/// Round 1/2/3 → [0.0, 0.0, 0.0]  (masih prior — re-estimasi setelah 1-2 item dianggap
///                                 belum signifikan, sama seperti konvensi a_optimal)
const HIST_THETA: [f64; 3] = [0.0, 0.0, 0.0];

/// Theta untuk round saat ini (digunakan saat scoring kandidat).
///
/// Round 1/2/3 → [0.0, 0.0, 0.0]  (prior — lihat catatan HIST_THETA)
const INITIAL_THETA: [f64; 3] = [0.0, 0.0, 0.0];

/// Items already administered before this round.
/// cum_FIM dihitung otomatis dari daftar ini via item_fim() — tidak hardcode.
///
/// Round 1 → []                          (degenerate: cum_FIM = 0, semua det = 0,
///                                        winner = first item in bank order = m2p-v001)
/// Round 2 → [m2p-v001]                  (degenerate: cum_FIM rank-1, semua det masih 0,
///                                        winner = first remaining item in bank order = m2p-v002)
/// Round 3 → [m2p-v001, m2p-v002]        (cum_FIM rank-2 → det > 0 untuk kandidat yang arah
///                                        `a`-nya independen dari v001 & v002)
fn initial_administered() -> Vec<&'static str> {
    // Round 1: nothing administered yet
    // vec![]

    // Round 2: one item administered (m2p-v001 — winner of Round 1, see d_optimal_notes.md)
    // vec!["m2p-v001"]

    // Round 3: two items administered (m2p-v001, m2p-v002 — winners of Round 1 & 2)
    vec!["m2p-v001", "m2p-v002"]
}

// ─── Sample item bank ─────────────────────────────────────────────────────────

/// Lightweight local item descriptor — mirrors the fields of `src/models/item.rs`
/// that are relevant to item selection.  Add, remove, or edit rows freely.
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

/// Item bank — parameters from migrations/009_seed_m2pl_items.sql.
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

// ─── Local-only helpers (not in the API) ──────────────────────────────────────

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
    det_score: f64,
}

// ─── Main ─────────────────────────────────────────────────────────────────────

fn main() {
    println!();
    println!("══════════════════════════════════════════════════════════════════════");
    println!("  MCAT · D-Optimal Item Selection · Simulation");
    println!("══════════════════════════════════════════════════════════════════════");

    // --- Session state ---
    let k: usize = 3; // dimensions: verbal, numeric, reasoning
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
        println!("    cum_FIM     = {k}×{k} zero matrix  (round 1 — no history)");
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
    println!("    selection   = d_optimal → argmax_i det(cum_FIM + item_FIM_i)");
    println!("    exposure    = sympson_hetter (sh_r_param used directly, per item)");
    if administered.is_empty() {
        println!("    administered= []  (none yet)");
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
    // STAGE 2 — Compute item_FIM and D-Optimal score
    // API calls:
    //   mcat_engine::mirt::sigmoid
    //   mcat_engine::mirt::probability
    //   mcat_engine::mirt::item_fim
    //   mcat_engine::selection::score_item  (with SelectionMethod::DOptimal)
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 2 — Compute item_FIM and D-Optimal Score");

    println!("  APIs used per item:");
    println!("    mirt::sigmoid(linear)            → P*");
    println!("    mirt::probability(θ,a,d,c)       → P  (used to derive Q and w for display)");
    println!("    mirt::item_fim(θ,a,d,c)          → item_FIM matrix");
    println!("    selection::score_item(DOptimal,ctx,&fim) → det(cum_FIM + item_FIM)");
    println!();

    // SelectionContext is created once — same theta/cum_fim across all candidates.
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

        // — linear (shown step-by-step for clarity) ───────────────────────────
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
        let p_star = sigmoid(linear); // ← API call

        println!("    P* = mirt::sigmoid({:.4})", linear);
        println!("       = 1 / (1 + e^({:.4}))", -linear);
        println!("       = 1 / (1 + {:.6})", (-linear).exp());
        println!("       = {:.6}", p_star);
        println!();

        // — P via API: mirt::probability ──────────────────────────────────────
        let p = probability(&theta, &a_vec, item.d, item.c); // ← API call
        let q = 1.0 - p;
        let w = p * q; // M2PL shortcut: w = P·Q (since c=0)

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
        let fim = item_fim(&theta, &a_vec, item.d, item.c); // ← API call

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

        // — D-Optimal score via API: selection::score_item ────────────────────
        // score_item internally computes det(ctx.cum_fim + item_fim)
        let det = score_item(
            &SelectionMethod::DOptimal,
            &ctx,
            &fim,
            &a_vec,
            item.d,
            item.c,
        ); // ← API call

        println!("    det_score = selection::score_item(DOptimal, ctx, &fim)");
        println!("              = det(cum_FIM + item_FIM) = det(updated_FIM)");
        println!("              = {:.8}", det);
        if det.abs() < 1e-9 {
            println!("                ↑ updated_FIM is rank-deficient (rank < {k}) → det = 0");
        }
        println!();

        scored.push(Scored {
            item,
            linear,
            p_star,
            p,
            q,
            w,
            det_score: det,
        });
    }

    // ──────────────────────────────────────────────────────────────────────────
    // STAGE 3 — Rank and select
    // ──────────────────────────────────────────────────────────────────────────
    section("STAGE 3 — Ranking & argmax (Item Selection)");

    let all_det_zero = scored.iter().all(|s| s.det_score.abs() < 1e-9);

    if all_det_zero {
        println!("  All det_scores ≈ 0.0");
        println!("  Reason: updated_FIM = cum_FIM + item_FIM is rank-deficient (rank < {k}).");
        println!();
        println!("  D-optimal criterion is degenerate in this round.");
        println!("  No tiebreak exists in the real API (src/mcat/engine.rs): select_next_item()");
        println!("  keeps the first candidate and only replaces it on a STRICTLY greater score");
        println!("  (`score > best_score`), so on a full tie the winner is simply the first");
        println!("  eligible item in iteration order — an arbitrary outcome, not a criterion.");
        println!("  D-optimal becomes meaningful from round 3+ once cum_FIM");
        println!("  has accumulated rank ≥ 2 (two linearly independent FIM directions).");
    } else {
        println!("  Ranking by det_score (D-optimal primary criterion).");
    }
    println!();

    // Snap near-zero det scores to 0.0 before sorting to avoid ±0.0 artefacts.
    // Floating-point det of a rank-deficient matrix can be a tiny negative (e.g. -1e-17)
    // that partial_cmp would otherwise treat as strictly less than +0.0.
    //
    // Stable sort by det_score alone reproduces the real engine's "first-max-wins" behaviour:
    // `sort_by` preserves the original relative (bank/candidate) order among equal scores,
    // exactly like `select_next_item`'s strict `score > best_score` loop.
    let snap = |v: f64| if all_det_zero { 0.0 } else { v };
    scored.sort_by(|a, b| snap(b.det_score).partial_cmp(&snap(a.det_score)).unwrap());

    println!("  Rank  ID           Area          det_score");
    println!("  ──────────────────────────────────────────────");
    for (rank, s) in scored.iter().enumerate() {
        let marker = if rank == 0 { " ← SELECTED" } else { "" };
        println!(
            "    {:2}    {:12} {:12}  {:12.8}{}",
            rank + 1,
            s.item.id,
            s.item.content_area,
            s.det_score,
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
        println!(
            "    linear       = {:.4}  (= a·θ̂ + d = {:.4} + {:.2})",
            winner.linear,
            winner.linear - w.d,
            w.d
        );
        println!("    P*           = {:.6}  [mirt::sigmoid]", winner.p_star);
        println!("    P            = {:.6}  [mirt::probability]", winner.p);
        println!("    Q            = {:.6}", winner.q);
        println!("    w = P×Q      = {:.6}", winner.w);
        println!(
            "    det_score    = {:.8}  [selection::score_item(DOptimal)]",
            winner.det_score
        );
        println!();

        if all_det_zero {
            println!("  Why {} wins:", w.id);
            println!("    det_score is tied at 0.0 with every other candidate this round.");
            println!(
                "    {} wins only because it is first in iteration order — not a criterion.",
                w.id
            );
            println!();
            println!(
                "  Next step (not shown here): present item, record response x ∈ {{0,1}}, then:"
            );
            println!("    cum_FIM ← cum_FIM + mirt::item_fim(θ̂)");
            println!("    θ̂      ← MAP/MLE re-estimate (all responses so far)");
            println!("    check stopping rule (n < n_min=5 → CONTINUE)");
        } else {
            let a_norm_sq: f64 = w.a.iter().map(|x| x * x).sum();
            println!("  Why {} wins:", w.id);
            println!(
                "    ‖a‖² = {:.1}²+{:.1}²+{:.1}² = {:.4}",
                w.a[0], w.a[1], w.a[2], a_norm_sq
            );
            println!(
                "    det_score = {:.8}  — highest determinant among all candidates",
                winner.det_score
            );
        }
        println!();
    } else {
        println!();
        println!("  No item selected — all candidates blocked by exposure gate.");
        println!();
    }
    divider();
    println!();
}
