---
title: "Overview: Alur Implementasi MCAT"
date: "2026-08-06"
description: "Peta alur end-to-end implementasi Multidimensional CAT di project ini — dari REST endpoint sampai ke tiap komponen McatEngine — dengan diagram Mermaid."
author: "irufano"
tags:
  - AI
  - CAT
  - MCAT
  - Adaptive Test
---

> Dokumen ini adalah peta (bukan teori) — untuk pembuktian rumus per komponen lihat
> catatan di folder sebelah: [`estimation/`](./estimation/estimation_summary.md),
> [`item_selection/`](./item_selection), [`exposure/`](./exposure/exposure.md),
> [`stoping/`](./stoping/stoping.md). Kontrak client-facing resmi ada di
> [`API_CONTEXT.md`](../../API_CONTEXT.md). Kode produksi yang dirujuk di sini hidup di
> [`src/handlers/session_handler.rs`](../handlers/session_handler.rs) dan
> [`src/mcat/engine.rs`](../mcat/engine.rs).

## 0. Gambaran besar (big picture)

Sebelum masuk ke detail per komponen, ini inti dari seluruh algoritma MCAT: satu
**loop adaptif** yang berulang per item — pilih item paling informatif, examinee
menjawab, estimasi ulang kemampuan, cek apakah sudah cukup presisi untuk berhenti —
sampai stopping rule terpenuhi. Tiap kotak di bawah adalah satu "modul" yang dibahas
lebih dalam di section bernomor sama.

```mermaid
flowchart TD
    Start(("Start")) --> P1["Inisialisasi Sesi\ncreate_session\nθ̂ = 0, FIM = 0 (§1)"]
    P1 --> P2["Pilih Item Adaptif\nselection criterion\n+ exposure control (§1 & §2)"]
    P2 --> P3["Examinee Menjawab Item\nrespond → score_answer"]
    P3 --> P4["Estimasi Ulang Kemampuan\nMLE / MAP / EAP + SE (§3)"]
    P4 --> P5{"Cek Stopping Rule\nfixed_length / se_threshold /\nconvergence / hybrid (§4)"}
    P5 -- "Belum cukup presisi\n→ item berikutnya" --> P2
    P5 -- "Cukup presisi / n_max\n/ bank habis" --> P6["Sesi Completed\nθ_final + SE final (§1, §5)"]
    P6 --> End(("End"))
```

**Tiga poin kunci** yang membedakan MCAT dari CAT unidimensional biasa, semuanya
kelihatan di loop ini:

1. **Pilih Item** menilai informasi item secara **multidimensional** (matriks FIM
   k×k, bukan skalar) — lihat §2.
2. **Estimasi** menghasilkan **vektor θ̂** (Verbal, Numeric, Reasoning sekaligus),
   bukan satu skalar kemampuan — lihat §3.
3. **Stopping rule** mengevaluasi presisi di **semua dimensi sekaligus** (SE per
   dimensi harus memenuhi threshold, bukan cuma satu SE global) — lihat §4.

## 1. Siklus hidup satu sesi (level REST)

Satu sesi MCAT adalah loop `next-item → respond` yang berulang sampai stopping rule
terpenuhi atau bank soal habis, dibungkus oleh `create` di awal dan `result` di akhir.
Rute resmi ada di [`routes/sessions.rs`](../routes/sessions.rs):

```mermaid
flowchart TD
    Start(("Start")) --> A(["POST /sessions\ncreate_session"]) --> B["θ̂ = 0, FIM = 0\nSE dihitung dari prior\n(compute_se saat n=0)"]
    B --> C[("INSERT trans_test_sessions\nstatus = in_progress")]
    C --> D["GET /sessions/:id/next-item"]

    D --> E{"Ada trans_session_items pending\n(belum dijawab)?"}
    E -- "Ya (idempotent resume)" --> F["Kembalikan item pending\nyang sama"]
    E -- Tidak --> G["select_and_persist_next_item\n→ McatEngine::select_next_item"]
    G --> H{"Item terpilih?"}
    H -- Tidak --> I{"bank_exhausted_action"}
    I -- error --> I1["Tidak ada item eligible:\nGET next-item → 409 BankExhausted\nrespond (inline) → next_item=null,\nsesi TETAP in_progress"]
    I -- force_stop --> I2["status = completed\nstop_reason = bank_exhausted"]
    I -- extend --> I3["Ulangi seleksi tanpa\nfilter answered_ids"] --> H
    H -- Ya --> J[("INSERT trans_session_items\nrow pending + selection_detail")]
    F --> K["Item ditampilkan ke examinee"]
    J --> K

    K --> L["POST /sessions/:id/respond\npayload: item_id + response/answer"]
    L --> M["score_answer → 0/1"]
    M --> N["McatEngine::estimate_theta\natas SEMUA jawaban, bukan cuma yang baru"]
    N --> O["Update FIM kumulatif:\nI_n = I_n-1 + I_item(θ̂)"]
    O --> P["McatEngine::compute_se"]
    P --> Q["McatEngine::should_stop"]
    Q --> R{"Stop? (psikometrik:\nSE/convergence/n_max)"}
    R -- Ya --> S[("UPDATE trans_test_sessions\nstatus = completed + stop_reason\n— bank TIDAK dicek lagi, sudah\ntidak relevan")]
    R -- Tidak --> G2["select_and_persist_next_item\n(inline, sama seperti node G)"] --> H
    I2 -.-> S
    S --> T["GET /sessions/:id/result\nθ_final + SE + item_history"]
    T --> End(("End"))
    I1 --> End

    U["POST /sessions/:id/terminate\n(examinee keluar manual)"] -.-> S
```

**Catatan implementasi yang penting dari alur di atas:**

- `respond` selalu **re-estimasi θ̂ dari nol** memakai seluruh riwayat jawaban
  ([`session_handler.rs:340-368`](../handlers/session_handler.rs#L340-L368)), bukan
  update incremental — hanya FIM yang di-update secara aditif
  (`session_handler.rs:373-374`).
- `respond` menyisipkan seleksi item berikutnya **secara inline** di response yang
  sama ([`session_handler.rs:425-444`](../handlers/session_handler.rs#L425-L444)) —
  field `next_item`, supaya FE tidak perlu panggilan terpisah ke `GET next-item`
  di jalur normal. `GET next-item` sendiri tetap ada untuk resume/idempotency.
- `GET next-item` bersifat **idempotent**: kalau ada `trans_session_items` yang sudah
  di-insert tapi belum dijawab (`response IS NULL`), item yang sama dikembalikan
  lagi, bukan diseleksi ulang (`session_handler.rs:243-269`).

## 2. Di dalam `select_next_item` (item selection + exposure control)

Ini bagian yang paling banyak sub-komponennya. Dipanggil dari `next-item` maupun
inline dari `respond`, satu kali per kandidat item yang belum diadministrasikan dan
`is_active` ([`engine.rs:20-122`](../mcat/engine.rs#L20-L122)):

```mermaid
flowchart TD
    Start(("Start: dipanggil dari\nnext-item / respond")) --> A["Untuk setiap item aktif\nyang belum dijawab"]
    A --> B{"exposure::is_eligible?"}
    B -->|"ExposureMethod::None → true selalu"| D
    B -->|"SympsonHetter → hard gate\nBernoulli(sh_r_param)"| C{"Lolos gate?"}
    C -- Tidak --> Z["Skip item ini\n(catat eligible=false di CandidateDebug)"]
    C -- Ya --> D
    D["mirt::item_fim(θ̂, a, d, c)\n→ Fisher Information Matrix item"]
    D --> E{"selection_method?"}
    E -- d_optimal --> E1["d_optimal::score\ndet(FIM_baru)"]
    E -- a_optimal --> E2["a_optimal::score\ntrace(inv(FIM_baru))"]
    E -- kl_information --> E3["kl_information::score\nKL-divergence expected"]
    E1 --> F["score = raw_score"]
    E2 --> F
    E3 --> F
    F --> J["Bandingkan ke best score\n(argmax)"]
    Z -- "kandidat berikutnya" --> A
    J -- "kandidat berikutnya" --> A
    J -.->|"semua kandidat selesai\ndiproses"| End(("End: kembalikan\nitem dgn best score"))
```

Ringkasan filosofi per method (detail rumus & pembuktian di
[`item_selection/`](./item_selection) dan [`exposure/exposure.md`](./exposure/exposure.md)):

| Sub-komponen | Method | File produksi |
|---|---|---|
| Selection criterion | `d_optimal` / `a_optimal` / `kl_information` | [`selection/*.rs`](../mcat/selection) |
| Exposure control (hard gate) | `sympson_hetter` | [`exposure/sympson_hetter.rs`](../mcat/exposure/sympson_hetter.rs) |
| Exposure control (off) | `none` | tidak ada logika tambahan |

## 3. Di dalam `estimate_theta` + `compute_se` (ability estimation)

Dipanggil sekali per `respond`, atas **seluruh** jawaban yang sudah masuk
([`engine.rs:125-202`](../mcat/engine.rs#L125-L202)):

```mermaid
flowchart TD
    Start(("Start: dipanggil per\nrespond")) --> A["estimate_theta(settings, semua a/d/c/respons)"]
    A --> B{"estimation_method"}
    B -- mle --> C["mle::estimate\nNewton-Raphson Fisher scoring\nθ_s+1 = θ_s + I_S⁻¹ ∇log f\n(tanpa prior, bisa divergen)"]
    B -- map --> D["map::estimate\nNewton-Raphson + prior penalty\nH = I_S + prior_cov_inv\n(mode posterior, tahan divergen)"]
    B -- eap --> E["eap::estimate\nIntegrasi kuadratur numerik\nθ̂ = Ekspektasi(θ | u) = ∫ θ·posterior dθ\n(bukan iteratif, selalu finite)"]
    C --> F["θ̂ baru"]
    D --> F
    E --> F
    F --> G["FIM kumulatif += item_fim(θ̂ baru)"]
    G --> H{"compute_se: estimation_method?"}
    H -- mle --> H1["se_vector(cum_fim)\n√diag(inv(I_S))"]
    H -- map --> H2["se_vector(cum_fim + prior_cov_inv)\n(fold in prior curvature)"]
    H -- eap --> H3["posterior_se langsung dari kuadratur\n(fallback ke rumus MAP kalau tidak\nada estimasi EAP segar)"]
    H1 --> I["SE per dimensi"]
    H2 --> I
    H3 --> I
    I --> End(("End: θ̂ + SE\ndikembalikan ke handler"))
```

## 4. Di dalam `should_stop` (stopping rule)

Dipanggil setelah SE ter-update, per `respond`
([`engine.rs:205-246`](../mcat/engine.rs#L205-L246)):

```mermaid
flowchart TD
    Start(("Start: dipanggil setelah\nSE ter-update")) --> A["should_stop(n, se, θ_now, θ_prev)"]
    A --> B{"stopping_rule"}
    B -- fixed_length --> C["n >= n_max?"]
    B -- se_threshold --> D["semua SE <= se_threshold\n(dan n >= n_min)?"]
    B -- convergence --> E["‖θ_now − θ_prev‖ < delta\n(dan n >= n_min)?"]
    B -- hybrid --> F["kombinasi se_threshold\n+ fixed_length sebagai batas atas"]
    C --> G{"Berhenti?"}
    D --> G
    E --> G
    F --> G
    G -- Ya --> H["stop = true + reason\n→ sesi completed"]
    G -- Tidak --> I["stop = false\n→ lanjut seleksi item berikutnya"]
    H --> End(("End"))
    I --> End
```

## 5. Ringkasan pemetaan endpoint → engine → DB

```mermaid
sequenceDiagram
    participant FE as mcat-fe
    participant API as session_handler
    participant Eng as McatEngine
    participant DB as MySQL

    Note over FE,DB: Start
    FE->>API: POST /sessions
    API->>DB: INSERT trans_test_sessions (θ=0, FIM=0)
    API-->>FE: session_id, θ_hat, se_vector

    FE->>API: GET /next-item
    API->>DB: SELECT master_items WHERE bank_id & is_active
    API->>Eng: select_next_item(...)
    Eng-->>API: (item_idx, score) + SelectionDebug
    API->>DB: INSERT trans_session_items (pending)
    API-->>FE: NextItemData

    FE->>API: POST /respond {item_id, response}
    API->>DB: SELECT semua jawaban terdahulu
    API->>Eng: estimate_theta(...)
    Eng-->>API: θ_after + EstimationDebug
    API->>Eng: compute_se(...)
    API->>Eng: should_stop(...)
    Eng-->>API: (stop?, reason, StoppingDebug)
    Note over API,DB: trans_session_items, trans_session_item_debug, next-item pending,\ntest_sessions, master_item_banks, trans_item_exposure_stats — 1 transaction
    alt stop == false
        API->>Eng: select_next_item(...) (inline)
        API->>DB: INSERT trans_session_items (pending berikutnya)
    else stop == true
        API->>DB: UPDATE trans_test_sessions status=completed
        API->>DB: UPDATE master_item_banks completed_sessions_count+1
    end
    API->>DB: UPDATE trans_session_items (response, θ_after, se_after)
    API->>DB: UPSERT trans_item_exposure_stats (exposure_count/rate, scoped per bank)
    API-->>FE: SubmitResponseData (+ next_item inline)

    FE->>API: GET /result
    API->>DB: SELECT session + item_history
    API-->>FE: SessionResultData
    Note over FE,DB: End
```

## 6. Peta modul kode

```
src/mcat/
├── engine.rs         ← orkestrator: select_next_item, estimate_theta, compute_se, should_stop
├── mirt.rs           ← item_fim, se_vector (matematika IRT multidimensional bersama)
├── debug.rs          ← CandidateDebug/SelectionDebug/EstimationDebug/StoppingDebug (audit trail)
├── playground.rs     ← simulasi tanpa DB, dipakai fitur "Playground" di FE
├── estimation/       ← mle.rs, map.rs, eap.rs   (§3 di atas)
├── selection/        ← d_optimal.rs, a_optimal.rs, kl_information.rs   (§2 di atas)
├── exposure/         ← sympson_hetter.rs, BankExhaustedAction   (§2 di atas)
└── stopping/         ← fixed_length.rs, se_threshold.rs, convergence.rs, hybrid.rs   (§4 di atas)
```

Handler yang memanggil `McatEngine` dan mem-persist hasilnya ke MySQL:
[`src/handlers/session_handler.rs`](../handlers/session_handler.rs) — lihat §1 di atas
untuk alur lengkapnya per endpoint.
