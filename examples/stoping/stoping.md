# Stopping Rule pada MCAT — Catatan

> File ini adalah catatan riset (bukan dokumentasi API). Implementasi resmi ada di
> [`src/mcat/stopping/`](../../mcat/stopping), dipanggil dari
> [`McatEngine::should_stop`](../../mcat/engine.rs). Kontrak client-facing ada di
> [`API_CONTEXT.md`](../../../API_CONTEXT.md) (bagian `TestSettings`, `stopping_detail`).

## 1. Apa itu stopping rule dalam CAT

Computerized Adaptive Testing (CAT) menghemat jumlah item dengan memilih item
berikutnya berdasarkan estimasi kemampuan (θ) saat ini, lalu **berhenti** begitu
kondisi tertentu terpenuhi — bukan setelah jumlah item tetap seperti tes kertas.
Tiga pertanyaan yang harus dijawab oleh sebuah stopping rule (Weiss & Kingsbury,
dan diulang di hampir semua literatur CAT sejak itu):

1. **Precision** — apakah estimasi θ sudah cukup presisi (SE cukup kecil)?
2. **Efficiency** — bisakah tes berhenti lebih awal tanpa mengorbankan presisi?
3. **Bounds** — berapa panjang minimum/maksimum tes yang masih bisa diterima
   secara praktis (waktu, exposure, kesabaran peserta)?

Semua rule di project ini adalah variasi dari menjawab ketiga pertanyaan itu.

## 2. Rule yang diimplementasikan

`StoppingRule` ([`mod.rs`](../../mcat/stopping/mod.rs)) punya 4 varian. Semua
menerima `StoppingInput` yang sama: `n`, `n_min`, `n_max`, `se_threshold`,
`delta`, SE per-dimensi (`&[Option<f64>]`), serta `theta_now`/`theta_prev`.

### 2.1 `fixed_length` ([`fixed_length.rs`](../../mcat/stopping/fixed_length.rs))

```rust
if i.n >= i.n_max { (true, "fixed_length") } else { (false, "") }
```

Paling sederhana: berhenti setelah tepat `n_max` item. Ini adalah baseline klasik
di literatur (tidak ada precision check) — dipakai sebagai kontrol pembanding
saat mengevaluasi rule lain, bukan untuk produksi (semua peserta dapat jumlah
item sama, padahal informasi yang mereka berikan per item berbeda-beda).

### 2.2 `se_threshold` ([`se_threshold.rs`](../../mcat/stopping/se_threshold.rs))

```rust
if max_se(i.se) <= i.se_threshold { (true, "se_threshold") } else { (false, "") }
```

Ini rule presisi klasik ("minimum standard error" / "variable-length" rule):
berhenti begitu SE(θ) ≤ threshold. Nilai umum di literatur unidimensional
adalah **SE ≈ 0.20–0.30** logit (Aybek & Demirtaşlı, 2017 — lihat referensi).
Tidak ada batas bawah/atas jumlah item di sini — murni presisi, sehingga
berisiko berhenti terlalu cepat (kalau prior/estimasi awal kebetulan sudah
"cocok") atau tidak pernah berhenti (kalau θ ekstrem dan SE lambat turun).

**Catatan multidimensi**: `max_se()` ([`mod.rs:52`](../../mcat/stopping/mod.rs))
mengambil SE **terburuk** di antara ketiga dimensi (Verbal/Numeric/Reasoning),
dan `None` (SE belum terdefinisi) dihitung sebagai `+inf` supaya dimensi yang
belum punya observasi tidak bisa lolos threshold secara diam-diam. Ini persis
rule *non-compensatory* yang diformalkan Braeken & Paap (2020):

> ∀q · SE(θ̂_q) ≤ δ ⇒ Stop(MCAT)

— semua dimensi harus memenuhi threshold sendiri-sendiri, bukan rata-rata atau
gabungan (compensatory) antar dimensi. Ini kenapa memilih item yang meng-optimalkan
informasi *gabungan* (D-optimal/A-optimal di
[`experimental/item_selection`](../item_selection)) belum tentu efisien untuk stopping
rule yang marjinal per-dimensi — Braeken & Paap menyebut ini "asimetri antara
selection rule dan stopping rule", dan mengusulkan pemilihan item yang
diarahkan ke dimensi dengan SE terburuk begitu dimensi lain sudah "selesai".
Ide ini relevan untuk `experimental/item_selection/*` project ini juga.

### 2.3 `convergence` ([`convergence.rs`](../../mcat/stopping/convergence.rs))

```rust
let diff = (i.theta_now - i.theta_prev).norm();
if diff <= i.delta { (true, "convergence") } else { (false, "") }
```

Berhenti kalau estimasi θ (vektor 3 dimensi) sudah "stabil" — perubahan antar
iterasi (Euclidean norm) di bawah `delta` (default 0.01). ​Ini bukan ukuran
presisi absolut (SE), tapi ukuran *diminishing returns*: item berikutnya
diperkirakan tidak akan mengubah estimasi secara berarti. Wang, Weiss & Shang
(2019) secara eksplisit membahas rule "perubahan absolut θ" ini dan
memperingatkan bahwa dipakai sendirian ia bisa berhenti prematur (θ bisa
"kebetulan" stabil sebelum SE benar-benar kecil, terutama di awal tes saat
prior masih dominan) — mereka menyarankan dipakai sebagai sinyal sekunder,
bukan rule utama. Itulah kenapa di project ini `convergence` berdiri sendiri
sebagai satu opsi (untuk eksperimen/pembanding), sementara jalur produksi
default memakai `hybrid`.

### 2.4 `hybrid` ([`hybrid.rs`](../../mcat/stopping/hybrid.rs))

```rust
if i.n < i.n_min { return (false, ""); }
if i.n >= i.n_max { return (true, "max_length"); }
if max_se(i.se) <= i.se_threshold { return (true, "se_threshold"); }
(false, "")
```

Ini kombinasi **bounds + precision** — pola yang paling umum dipakai di CAT
operasional (mis. NIH Toolbox CAT, lihat Amagai et al., 2025):

- `n_min` mencegah berhenti terlalu cepat (item pertama selalu tinggi-informasi
  relatif ke prior yang masih longgar, jadi SE bisa turun cepat semu di awal).
- `n_max` adalah batas keras (biaya, waktu, exposure).
- Di antara keduanya, `se_threshold` yang memutuskan.

Ini default `stopping_rule` di [`config.rs`](../../config.rs) /
[`API_CONTEXT.md`](../../../API_CONTEXT.md) (`n_min=5, n_max=30,
se_threshold=0.30`), konsisten dengan rentang SE 0.20–0.30 yang umum di
literatur dan dengan pola "reduced maximum rule" pada Amagai et al. (2025) yang
justru menurunkan `n_max` (bukan menaikkan `n_min`) untuk mengurangi beban
peserta tanpa kehilangan presisi klinis yang dibutuhkan.

## 3. Yang **tidak** ditangani modul `stopping/` ini

`should_stop` hanya menjawab "berhenti karena alasan psikometrik?". Dua alasan
berhenti lain ditangani di layer session, bukan di sini:

- `bank_exhausted` — item pool habis sebelum rule psikometrik terpenuhi
  (lihat `session_handler.rs`, `bank_exhausted_action: force_stop | ...`).
- `user_terminated` — peserta/admin menghentikan sesi manual.

Ini pola yang wajar: stopping rule psikometrik (bab ini) menjawab "apakah
pengukuran sudah cukup baik", sedangkan alasan operasional lain menjawab
"apakah kita *bisa* melanjutkan sama sekali".

## 4. Ide eksperimental (alasan folder ini ada)

Beberapa arah yang didukung literatur dan belum ada di `mcat/stopping/`
produksi, cocok untuk dicoba di `experimental/`:

1. **SE-change rule** (Amagai et al., 2025): tambahan kondisi berhenti kalau
   penurunan SE antar item < ambang kecil (mis. < 0.01), sebagai pelengkap
   `se_threshold` absolut — analog ke `convergence.rs` tapi di ruang SE, bukan
   ruang θ.
2. **Dimension-targeted selection saat mendekati stop** (Braeken & Paap, 2020):
   begitu satu dimensi sudah di bawah `se_threshold`, arahkan pemilihan item
   berikutnya ke dimensi dengan SE terburuk saja (dynamic restriction/filtering)
   supaya tidak "membuang" item pada dimensi yang sudah cukup presisi.
3. **Sequential/likelihood-ratio stopping** untuk mode klasifikasi (SPRT/GLR,
   lihat referensi UMN di bawah) — relevan kalau MCAT ini pernah dipakai untuk
   keputusan lulus/tidak-lulus per kategori, bukan estimasi θ kontinu.

## 5. Referensi

Semua link berikut sudah dicek bisa diunduh langsung (open access / PDF publik)
per Agustus 2026, supaya bisa dikonfirmasi ulang.

1. Aybek, E. C., & Demirtaşlı, R. N. (2017). *Computerized Adaptive Test (CAT)
   Applications and Item Response Theory Models for Polytomous Items*.
   International Journal of Research in Education and Science, 3(2), 475–487.
   PDF: https://files.eric.ed.gov/fulltext/EJ1148445.pdf

2. Amagai, S., Kaat, A. J., Fox, R. S., et al. (2025). *Customizing
   Computerized Adaptive Test Stopping Rules for Clinical Settings Using the
   Negative Affect Subdomain of the NIH Toolbox Emotion Battery: Simulation
   Study*. JMIR Formative Research, 9, e60215.
   PDF: https://formative.jmir.org/2025/1/e60215/PDF
   (mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC11951945/)

3. Braeken, J., & Paap, M. C. S. (2020). *Making Fixed-Precision Between-Item
   Multidimensional Computerized Adaptive Tests Even Shorter by Reducing the
   Asymmetry Between Selection and Stopping Rules*. Applied Psychological
   Measurement, 44(7–8), 531–547.
   PDF (open access): https://pmc.ncbi.nlm.nih.gov/articles/PMC7495795/

4. Meneghetti, D. R., & Aquino Junior, P. T. (2017/2018). *Application and
   Simulation of Computerized Adaptive Tests Through the Package catsim*.
   arXiv:1707.03012.
   PDF: https://arxiv.org/pdf/1707.03012 (abstrak: https://arxiv.org/abs/1707.03012)
   — implementasi Python open-source dengan arsitektur stopping-rule yang bisa
   diplug-in, mirip pola `StoppingRule` enum + trait di project ini; bagus untuk
   pembanding desain.

5. Mulder, J., & van der Linden, W. J. (2009). *Multidimensional Adaptive
   Testing with Optimal Design Criteria for Item Selection*. Psychometrika,
   74(2), 273–296.
   PDF (open access via PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC2813188/
   — bukan tentang stopping secara langsung, tapi latar belakang D-/A-optimal
   yang dipakai di `experimental/item_selection/{d_optimal,a_optimal}`, dan
   relevan untuk poin asimetri selection↔stopping di §2.2 dan §4.2.

### Bacaan lanjutan (klasik, sering dikutip, tapi paywalled di semua sumber yang saya temukan — dicek via institusi/perpustakaan untuk konfirmasi rumus asli)

- Segall, D. O. (1996). *Multidimensional adaptive testing*. Psychometrika,
  61(2), 331–354. — paper fondasi MCAT (item selection + scoring Bayesian).
- Wang, C., Weiss, D. J., & Shang, Z. (2019). *Variable-Length Stopping Rules
  for Multidimensional Computerized Adaptive Testing*. Psychometrika, 84(3),
  749–771. https://doi.org/10.1007/s11336-018-9644-7 — pembahasan paling
  langsung soal rule presisi vs rule perubahan-θ di §2.3 di atas.
- Dodd, B. G., Koch, W. R., & De Ayala, R. J. (1993). *Computerized adaptive
  testing using the partial credit model: Effects of item pool characteristics
  and different stopping rules*. Educational and Psychological Measurement,
  53, 61–77.
