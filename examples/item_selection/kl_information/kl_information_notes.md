### Configuration

```
Dimensions (k=3): [verbal (0), numeric (1), reasoning (2)]
Selection:        KL Information (Kullback-Leibler Information)     [selection/kl_information.rs]
```


**Probability:**

```
General M3PL:  

    P = c + (1 - c) × σ(a·θ + d)

M2PL (c=0):   

    P = σ(a·θ + d) 
      = 1 / (1 + exp(-(a·θ + d)))
```

**Fisher Information Matrix (FIM):**

```
I_i(θ) = w × (a × aᵀ)

Where:
    P* = σ(a·θ + d)              — sigmoid probability (without guessing)
    P' = (1-c) × P* × (1 - P*)   — derivative of P w.r.t. linear predictor
    Q  = 1 - P                   — probability of incorrect
    w  = (P')² / (P × Q)         — information weight
```

```
M2PL (c=0): 

    P* = P
    P' = P × (1-P) = P × Q

    → w  = (P×Q)² / (P×Q) = P × Q


M2PL FIM:  

    I_i(θ) = P(1-P) × (a × aᵀ)
           = w × (a × aᵀ)
```

**Catatan penting:** `item_FIM` masih dihitung dan dilewatkan ke `selection::score_item()`
karena tanda tangan fungsi itu dipakai bersama oleh 3 metode seleksi (D-optimal, A-optimal,
KL information). Tetapi **KL information tidak menggunakan `item_FIM` sama
sekali** — skornya dihitung langsung dari `theta`, `a`, `d`, `c`, dan jumlah item yang sudah
di-administer (lihat bawah).

---

#### KL Information Criterion

##### Formula (Chang & Ying, 1996)

Diverifikasi kata-demi-kata dari dua sumber terbuka (lihat §Referensi [1] Eq. 9-10, p.6;
[2] Eq. 4 & 6, p.3-4 — keduanya mereproduksi rumus Chang & Ying secara identik):

**KL pointwise** (KL divergence eksak antara dua distribusi Bernoulli — bukan aproksimasi):

```
K_i(θ ‖ θ̂) = P_i(θ̂)·ln[P_i(θ̂)/P_i(θ)] + [1-P_i(θ̂)]·ln{[1-P_i(θ̂)] / [1-P_i(θ)]}
```

di mana `P_i(θ̂)` adalah peluang benar pada level kemampuan θ̂ (titik estimasi saat ini) dan
`P_i(θ)` peluang benar pada level kemampuan lain θ yang sedang diintegralkan. Formula ini
adalah KL divergence Bernoulli baku: `D(Bernoulli(P_i(θ̂)) ‖ Bernoulli(P_i(θ)))`, karena itu
selalu ≥ 0 (Gibbs' inequality) dan = 0 hanya ketika θ = θ̂.

**KL information global** (rata-rata bergerak / *moving average* Chang & Ying):

```
K̄_i(θ̂) = ∫_{θ̂-δ}^{θ̂+δ} K_i(θ ‖ θ̂) dθ ,     δ = C / √m
```

`m` = jumlah item yang sudah di-administer sejauh ini, `C` konstanta yang "dipilih menurut
target *coverage probability*" — Chang & Ying **tidak** mematok satu nilai C yang eksak ([1], p.6:
*"the exact determination of δ could be ambiguous, so Chang and Ying proposed c/√m as a
reasonable choice for δ"*). `C = 3` (~3 galat baku asimtotik) adalah nilai yang umum dipakai
dalam literatur turunannya dan yang dipakai sebagai default di implementasi ini
(`kl_information::INTERVAL_C`).

Item dipilih dengan `argmax_i K̄_i(θ̂)` — semakin besar KL information global, semakin
"informatif" item tersebut terhadap seluruh rentang kemampuan yang masih plausible di sekitar θ̂.

##### Reduksi Multidimensional → Skalar (pembuktian, bukan asumsi)

MCAT ini memakai model M2PL/M3PL kompensatorik (`src/mcat/mirt.rs:10-14`):

```
P_i(θ) = c + (1-c) × σ(a·θ + d) = f(z),   z = a·θ + d
```

`P_i` adalah fungsi θ **hanya** melalui skalar `z = a·θ + d` — dua vektor kemampuan θ₁, θ₂
yang menghasilkan `z` sama pasti menghasilkan `P_i` yang sama persis, tidak peduli
dimensi k-nya. Karena `K_i(θ‖θ̂)` (Eq. 9) dibangun murni dari `P_i(θ)` dan `P_i(θ̂)`, maka:

```
K_i(θ ‖ θ̂) = K_i(z ‖ ẑ)     — identik, bukan aproksimasi, untuk SEMUA θ, θ̂ dengan z, ẑ sama
```

Ini konsekuensi aljabar langsung dari definisi model compensatory MIRT, bukan asumsi yang
ditambahkan. Yang **merupakan keputusan desain** (bukan dikutip verbatim dari satu paper
multidimensional tertentu) adalah bagaimana interval integrasi didefinisikan untuk θ berdimensi-k:
implementasi ini mengintegralkan langsung dalam ruang z (`z ∈ [ẑ-δ, ẑ+δ]`), memperlakukan
komposit score z persis seperti θ unidimensional pada rumus asli Chang & Ying. Paper yang
mengembangkan generalisasi multidimensional secara formal — Veldkamp & van der Linden (2002,
*Psychometrika*, dikutip di [1] sebagai kandidat metode "paling mungkin" untuk item selection
berbasis informasi) dan Mulder & van der Linden (2010, *Elements of Adaptive Testing*, Springer,
Ch. 4) — tidak berhasil diakses full-text (SAGE/Springer/ResearchGate menolak akses tanpa
langganan saat penelusuran ini dilakukan), sehingga **tidak dapat dipastikan** apakah reduksi
di atas identik dengan pendekatan mereka. Yang bisa dibuktikan secara independen dari model
sendiri (dan karena itu tetap valid terlepas dari literatur mana yang dirujuk) adalah kesetaraan
`K_i(θ‖θ̂) = K_i(z‖ẑ)` di atas.

##### Catatan implementasi: `δ = C/√(m+1)`, bukan `C/√m`

Chang & Ying mendefinisikan `m` sebagai jumlah item yang sudah di-administer. Pada `m=0`
(sebelum item pertama), `C/√0` tidak terdefinisi. Implementasi ini memakai `C/√(m+1)` agar
interval selalu terdefinisi dan finite mulai dari ronde pertama, mengecil persis seperti pada
literatur begitu `m≥1`. Ini adalah keputusan implementasi untuk menutup kasus tepi yang tidak
dibahas eksplisit oleh Chang & Ying, bukan klaim yang diambil dari rumus mereka.

---

#### Trace Matriks (masih relevan untuk D-optimal / A-optimal)

```
trace dari matriks M = jumlah diagonal utama

M = [[m₀₀  m₀₁  m₀₂]
     [m₁₀  m₁₁  m₁₂]
     [m₂₀  m₂₁  m₂₂]]

tr(M) = m₀₀ + m₁₁ + m₂₂
```

`item_FIM` dan trace-nya **tidak dipakai oleh KL information** — hanya dipakai oleh D-optimal
dan A-optimal (lihat file notes masing-masing).

---

### Item Bank Snapshot (representative items)

```
ID         Content area  a = [verbal, numeric, reasoning]   d      c
─────────────────────────────────────────────────────────────────────
m2p-v001   verbal        [1.9, 0.2, 0.3]                   +0.40  0.0
m2p-v002   verbal        [1.7, 0.2, 0.2]                   +0.10  0.0
m2p-n001   numeric       [0.3, 1.9, 0.4]                   +0.80  0.0
m2p-n002   numeric       [0.3, 1.8, 0.4]                   +0.50  0.0
m2p-r001   reasoning     [0.5, 0.4, 2.0]                   +0.30  0.0
m2p-r002   reasoning     [0.3, 0.8, 1.9]                   +0.60  0.0
m2p-r003   reasoning     [0.4, 0.3, 1.8]                   +0.70  0.0
```

---

#### Item Selection — KL Information (Kullback-Leibler Information)

Seluruh angka pada bagian ini adalah **output asli** dari
`cargo run --example item_selection_kl_information` (bukan hasil hitung manual) — dijalankan
dua kali dengan konfigurasi `HIST_THETA` / `INITIAL_THETA` / `initial_administered()` yang
di-toggle ke skenario Round 1 dan Round 4, sesuai instruksi di dalam file script. Notes ini
karena itu dijamin identik dengan hasil script (`Jangan berasumsi, buktikan` — angka di sini
BUKAN didapat dari mental math).

---

##### Round 1 — 0 Item Administered

```
θ̂            = [0.0, 0.0, 0.0]   (initial estimate — belum ada informasi)
cum_FIM      = 3×3 zero matrix   (tidak dipakai oleh KL score)
Administered = []                (m = 0)
Candidates   = all 7 items
δ            = 3 / √(0+1) = 3.0000
```

**Perhitungan per item** (θ̂=[0,0,0], sehingga `z_hat = d`, dan `δ = 3/√(0+1) = 3.0000` untuk semua item):

---

**m2p-v001** — `a=[1.9, 0.2, 0.3]`, `d=0.40`

```
z_hat   =  1.9×0 + 0.2×0 + 0.3×0 + 0.40  =  0.4000

P_i(ẑ)  =  σ(0.4000)  =  1 / (1 + e^-0.4000)  =  1 / 1.670320  =  0.598688

[ẑ-δ, ẑ+δ]  =  [-2.6000, 3.4000]
P(lower)    =  σ(-2.6000)  =  0.069138
P(upper)    =  σ( 3.4000)  =  0.967705

K_i(lower‖ẑ) = 0.598688×ln(0.598688/0.069138) + 0.401312×ln(0.401312/0.930862)
             = 0.598688×2.1586 + 0.401312×(-0.8414)
             = 1.2923 − 0.3377  =  0.954692

K_i(upper‖ẑ) = 0.598688×ln(0.598688/0.967705) + 0.401312×ln(0.401312/0.032295)
             = 0.598688×(-0.4802) + 0.401312×2.5198
             = -0.2875 + 1.0112  =  0.723750

kl_score = ∫_{-2.6000}^{3.4000} K_i(z‖0.4000) dz   (Simpson 40-panel)  =  1.840805
```

---

**m2p-v002** — `a=[1.7, 0.2, 0.2]`, `d=0.10`

```
z_hat   =  1.7×0 + 0.2×0 + 0.2×0 + 0.10  =  0.1000

P_i(ẑ)  =  σ(0.1000)  =  1 / (1 + e^-0.1000)  =  1 / 1.904837  =  0.524979

[ẑ-δ, ẑ+δ]  =  [-2.9000, 3.1000]
P(lower)    =  σ(-2.9000)  =  0.052154
P(upper)    =  σ( 3.1000)  =  0.956893

K_i(lower‖ẑ) = 0.524979×ln(0.524979/0.052154) + 0.475021×ln(0.475021/0.947846)
             = 0.524979×2.3092 + 0.475021×(-0.6908)
             = 1.2123 − 0.3282  =  0.884104

K_i(upper‖ẑ) = 0.524979×ln(0.524979/0.956893) + 0.475021×ln(0.475021/0.043107)
             = 0.524979×(-0.6003) + 0.475021×2.3997
             = -0.3152 + 1.1399  =  0.824730

kl_score = ∫_{-2.9000}^{3.1000} K_i(z‖0.1000) dz   (Simpson 40-panel)  =  1.884718   ← highest
```

---

**m2p-n001** — `a=[0.3, 1.9, 0.4]`, `d=0.80`

```
z_hat   =  0.3×0 + 1.9×0 + 0.4×0 + 0.80  =  0.8000

P_i(ẑ)  =  σ(0.8000)  =  1 / 1.449329  =  0.689974

[ẑ-δ, ẑ+δ]  =  [-2.2000, 3.8000]
P(lower)    =  σ(-2.2000)  =  0.099750
P(upper)    =  σ( 3.8000)  =  0.978119

K_i(lower‖ẑ) = 0.689974×ln(0.689974/0.099750) + 0.310026×ln(0.310026/0.900250)
             = 0.689974×1.9340 + 0.310026×(-1.0660)
             = 1.3344 − 0.3305  =  1.003906

K_i(upper‖ẑ) = 0.689974×ln(0.689974/0.978119) + 0.310026×ln(0.310026/0.021881)
             = 0.689974×(-0.3490) + 0.310026×2.6510
             = -0.2408 + 0.8219  =  0.581100

kl_score = ∫_{-2.2000}^{3.8000} K_i(z‖0.8000) dz   (Simpson 40-panel)  =  1.708207   ← lowest
```

---

**m2p-n002** — `a=[0.3, 1.8, 0.4]`, `d=0.50`

```
z_hat   =  0.3×0 + 1.8×0 + 0.4×0 + 0.50  =  0.5000

P_i(ẑ)  =  σ(0.5000)  =  1 / 1.606531  =  0.622459

[ẑ-δ, ẑ+δ]  =  [-2.5000, 3.5000]
P(lower)    =  σ(-2.5000)  =  0.075858
P(upper)    =  σ( 3.5000)  =  0.970688

K_i(lower‖ẑ) = 0.622459×ln(0.622459/0.075858) + 0.377541×ln(0.377541/0.924142)
             = 0.622459×2.1048 + 0.377541×(-0.8952)
             = 1.3102 − 0.3380  =  0.972191

K_i(upper‖ẑ) = 0.622459×ln(0.622459/0.970688) + 0.377541×ln(0.377541/0.029312)
             = 0.622459×(-0.4443) + 0.377541×2.5557
             = -0.2766 + 0.9649  =  0.688295

kl_score = ∫_{-2.5000}^{3.5000} K_i(z‖0.5000) dz   (Simpson 40-panel)  =  1.815040
```

---

**m2p-r001** — `a=[0.5, 0.4, 2.0]`, `d=0.30`

```
z_hat   =  0.5×0 + 0.4×0 + 2.0×0 + 0.30  =  0.3000

P_i(ẑ)  =  σ(0.3000)  =  1 / 1.740818  =  0.574443

[ẑ-δ, ẑ+δ]  =  [-2.7000, 3.3000]
P(lower)    =  σ(-2.7000)  =  0.062973
P(upper)    =  σ( 3.3000)  =  0.964429

K_i(lower‖ẑ) = 0.574443×ln(0.574443/0.062973) + 0.425557×ln(0.425557/0.937027)
             = 0.574443×2.2107 + 0.425557×(-0.7893)
             = 1.2699 − 0.3359  =  0.934016

K_i(upper‖ẑ) = 0.574443×ln(0.574443/0.964429) + 0.425557×ln(0.425557/0.035571)
             = 0.574443×(-0.5181) + 0.425557×2.4819
             = -0.2976 + 1.0562  =  0.758536

kl_score = ∫_{-2.7000}^{3.3000} K_i(z‖0.3000) dz   (Simpson 40-panel)  =  1.861145
```

---

**m2p-r002** — `a=[0.3, 0.8, 1.9]`, `d=0.60`

```
z_hat   =  0.3×0 + 0.8×0 + 1.9×0 + 0.60  =  0.6000

P_i(ẑ)  =  σ(0.6000)  =  1 / 1.548812  =  0.645656

[ẑ-δ, ẑ+δ]  =  [-2.4000, 3.6000]
P(lower)    =  σ(-2.4000)  =  0.083173
P(upper)    =  σ( 3.6000)  =  0.973403

K_i(lower‖ẑ) = 0.645656×ln(0.645656/0.083173) + 0.354344×ln(0.354344/0.916827)
             = 0.645656×2.0493 + 0.354344×(-0.9507)
             = 1.3232 − 0.3369  =  0.986317

K_i(upper‖ẑ) = 0.645656×ln(0.645656/0.973403) + 0.354344×ln(0.354344/0.026597)
             = 0.645656×(-0.4105) + 0.354344×2.5895
             = -0.2651 + 0.9176  =  0.652500

kl_score = ∫_{-2.4000}^{3.6000} K_i(z‖0.6000) dz   (Simpson 40-panel)  =  1.784127
```

---

**m2p-r003** — `a=[0.4, 0.3, 1.8]`, `d=0.70`

```
z_hat   =  0.4×0 + 0.3×0 + 1.8×0 + 0.70  =  0.7000

P_i(ẑ)  =  σ(0.7000)  =  1 / 1.496585  =  0.668188

[ẑ-δ, ẑ+δ]  =  [-2.3000, 3.7000]
P(lower)    =  σ(-2.3000)  =  0.091123
P(upper)    =  σ( 3.7000)  =  0.975873

K_i(lower‖ẑ) = 0.668188×ln(0.668188/0.091123) + 0.331812×ln(0.331812/0.908877)
             = 0.668188×1.9924 + 0.331812×(-1.0076)
             = 1.3313 − 0.3343  =  0.996923

K_i(upper‖ẑ) = 0.668188×ln(0.668188/0.975873) + 0.331812×ln(0.331812/0.024127)
             = 0.668188×(-0.3788) + 0.331812×2.6212
             = -0.2531 + 0.8698  =  0.616673

kl_score = ∫_{-2.3000}^{3.7000} K_i(z‖0.7000) dz   (Simpson 40-panel)  =  1.748393
```

---

**Verifikasi manual integral (Simpson n=4, 5 titik: lower, q1, ẑ, q3, upper)**

Formula Simpson komposit dengan `h = 2δ/n`:

```
∫ ≈ (h/3) × [K(lower) + 4·K(q1) + 2·K(ẑ) + 4·K(q3) + K(upper)]
```

`K_i(ẑ‖ẑ) = 0` selalu (titik pusat, lihat pembuktian di §Formula). Dua contoh lengkap:

```
m2p-v002:  h = (3.1000-(-2.9000))/4 = 1.5000
  q1 = -1.4000 → P=0.197816 → K_q1=0.263490
  q3 =  1.6000 → P=0.832018 → K_q3=0.252035
  simpson(n=4) = (1.5/3)×[0.884104 + 4×0.263490 + 2×0 + 4×0.252035 + 0.824730]
               = 0.5×[0.884104+1.053960+0+1.008140+0.824730]
               = 0.5×3.770934 = 1.885467

m2p-r001:  h = (3.3000-(-2.7000))/4 = 1.5000
  q1 = -1.2000 → P=0.231475 → K_q1=0.270591
  q3 =  1.8000 → P=0.858149 → K_q3=0.236959
  simpson(n=4) = (1.5/3)×[0.934016 + 4×0.270591 + 2×0 + 4×0.236959 + 0.758536]
               = 0.5×[0.934016+1.082364+0+0.947836+0.758536]
               = 0.5×3.722752 = 1.861376
```

Tabel silang n=4 (manual) vs n=40 (kode, `kl_information.rs`) untuk semua item — selisih kecil
murni akibat resolusi kuadratur, membuktikan `kl_score` di kode memang hasil integral formula
di atas, bukan angka lain:

```
Item      simpson(n=4)   simpson(n=40, kode)   selisih
──────────────────────────────────────────────────────
m2p-v001    1.840604         1.840805           0.000201
m2p-v002    1.885467         1.884718          -0.000749
m2p-n001    1.705563         1.708207           0.002644
m2p-n002    1.814315         1.815040           0.000725
m2p-r001    1.861376         1.861145          -0.000231
m2p-r002    1.782804         1.784127           0.001323
m2p-r003    1.746422         1.748393           0.001971
```

##### Ranking Round 1

```
Rank  Item       Area          kl_score    z_hat    δ       Note
────────────────────────────────────────────────────────────────────────
  1   m2p-v002   verbal         1.884718   0.1000  3.0000   SELECTED
  2   m2p-r001   reasoning      1.861145   0.3000  3.0000
  3   m2p-v001   verbal         1.840805   0.4000  3.0000
  4   m2p-n002   numeric        1.815040   0.5000  3.0000
  5   m2p-r002   reasoning      1.784127   0.6000  3.0000
  6   m2p-r003   reasoning      1.748393   0.7000  3.0000
  7   m2p-n001   numeric        1.708207   0.8000  3.0000
```

**Selected: `m2p-v002`** — verbal item, `a=[1.7, 0.2, 0.2]`.

**Catatan:** item terpilih (`m2p-v002`) bukan item dengan `‖a‖²` terbesar di bank (itu
`m2p-r001`) — `m2p-v002` menang karena `z_hat` paling dekat ke pusat interval integrasi dan
kurva ICC-nya masih cukup curam di sekitar rentang [-2.9, 3.1]. Ini konsisten dengan temuan
Chang & Ying (1996): item selection berbasis KL information dapat berbeda dari item selection
berbasis Fisher information murni (MFI), khususnya pada tahap awal tes ketika θ̂ masih jauh dari
θ sebenarnya ([1], p.6: *"Chang and Ying (1996) found that replacing the MFI criterion with the
KLI criterion often reduced the biases and mean-squared errors of proficiency estimation when
the test length was short (m<30) or ... early stage"*).

**Semua score finite dan ≥ 0** (Gibbs' inequality, diverifikasi lewat unit test
`pointwise_kl_is_nonnegative` di `kl_information.rs`) — tidak ada degenerate case, karena KL
divergence Bernoulli selalu ≥ 0.

---

##### Round 4 — 3 Item Administered

Scenario: 3 item sudah di-administer (r001, v001, n001). Memilih item ke-4.

```
θ̂            = [1.0, -0.5, 0.5]   (verbal kuat, numeric lemah, reasoning sedang)
Administered = [m2p-r001, m2p-v001, m2p-n001]      (m = 3)
Candidates   = [m2p-v002, m2p-n002, m2p-r002, m2p-r003]
δ            = 3 / √(3+1) = 1.5000
```

**Perhitungan per item** (θ̂=[1.0, -0.5, 0.5], `δ = 3/√(3+1) = 1.5000` untuk semua item):

---

**m2p-v002** — `a=[1.7, 0.2, 0.2]`, `d=0.10`

```
z_hat   =  1.7×1.0 + 0.2×(-0.5) + 0.2×0.5 + 0.10
        =  1.7000 − 0.1000 + 0.1000 + 0.10  =  1.8000

P_i(ẑ)  =  σ(1.8000)  =  1 / (1 + e^-1.8000)  =  1 / 1.165299  =  0.858149

[ẑ-δ, ẑ+δ]  =  [0.3000, 3.3000]
P(lower)    =  σ(0.3000)  =  0.574443
P(upper)    =  σ(3.3000)  =  0.964429

K_i(lower‖ẑ) = 0.858149×ln(0.858149/0.574443) + 0.141851×ln(0.141851/0.425557)
             = 0.858149×0.4014 + 0.141851×(-1.0986)
             = 0.3444 − 0.1558  =  0.188601

K_i(upper‖ẑ) = 0.858149×ln(0.858149/0.964429) + 0.141851×ln(0.141851/0.035571)
             = 0.858149×(-0.1168) + 0.141851×1.3832
             = -0.1002 + 0.1962  =  0.096018

kl_score = ∫_{0.3000}^{3.3000} K_i(z‖1.8000) dz   (Simpson 40-panel)  =  0.140419
```

---

**m2p-n002** — `a=[0.3, 1.8, 0.4]`, `d=0.50`

```
z_hat   =  0.3×1.0 + 1.8×(-0.5) + 0.4×0.5 + 0.50
        =  0.3000 − 0.9000 + 0.2000 + 0.50  =  0.1000

P_i(ẑ)  =  σ(0.1000)  =  0.524979

[ẑ-δ, ẑ+δ]  =  [-1.4000, 1.6000]
P(lower)    =  σ(-1.4000)  =  0.197816
P(upper)    =  σ( 1.6000)  =  0.832018

K_i(lower‖ẑ) = 0.524979×ln(0.524979/0.197816) + 0.475021×ln(0.475021/0.802184)
             = 0.524979×0.9760 + 0.475021×(-0.5240)
             = 0.5124 − 0.2489  =  0.263490

K_i(upper‖ẑ) = 0.524979×ln(0.524979/0.832018) + 0.475021×ln(0.475021/0.167982)
             = 0.524979×(-0.4605) + 0.475021×1.0395
             = -0.2418 + 0.4938  =  0.252035

kl_score = ∫_{-1.4000}^{1.6000} K_i(z‖0.1000) dz   (Simpson 40-panel)  =  0.266355   ← highest
```

---

**m2p-r002** — `a=[0.3, 0.8, 1.9]`, `d=0.60`

```
z_hat   =  0.3×1.0 + 0.8×(-0.5) + 1.9×0.5 + 0.60
        =  0.3000 − 0.4000 + 0.9500 + 0.60  =  1.4500

P_i(ẑ)  =  σ(1.4500)  =  0.809998

[ẑ-δ, ẑ+δ]  =  [-0.0500, 2.9500]
P(lower)    =  σ(-0.0500)  =  0.487503
P(upper)    =  σ( 2.9500)  =  0.950263

K_i(lower‖ẑ) = 0.809998×ln(0.809998/0.487503) + 0.190002×ln(0.190002/0.512497)
             = 0.809998×0.5077 + 0.190002×(-0.9923)
             = 0.4113 − 0.1885  =  0.222734

K_i(upper‖ẑ) = 0.809998×ln(0.809998/0.950263) + 0.190002×ln(0.190002/0.049737)
             = 0.809998×(-0.1597) + 0.190002×1.3403
             = -0.1294 + 0.2547  =  0.125295

kl_score = ∫_{-0.0500}^{2.9500} K_i(z‖1.4500) dz   (Simpson 40-panel)  =  0.173915
```

---

**m2p-r003** — `a=[0.4, 0.3, 1.8]`, `d=0.70`

```
z_hat   =  0.4×1.0 + 0.3×(-0.5) + 1.8×0.5 + 0.70
        =  0.4000 − 0.1500 + 0.9000 + 0.70  =  1.8500

P_i(ẑ)  =  σ(1.8500)  =  0.864127

[ẑ-δ, ẑ+δ]  =  [0.3500, 3.3500]
P(lower)    =  σ(0.3500)  =  0.586618
P(upper)    =  σ(3.3500)  =  0.966105

K_i(lower‖ẑ) = 0.864127×ln(0.864127/0.586618) + 0.135873×ln(0.135873/0.413382)
             = 0.864127×0.3873 + 0.135873×(-1.1127)
             = 0.3347 − 0.1512  =  0.183537

K_i(upper‖ẑ) = 0.864127×ln(0.864127/0.966105) + 0.135873×ln(0.135873/0.033895)
             = 0.864127×(-0.1116) + 0.135873×1.3884
             = -0.0964 + 0.1887  =  0.092257

kl_score = ∫_{0.3500}^{3.3500} K_i(z‖1.8500) dz   (Simpson 40-panel)  =  0.135822   ← lowest
```

---

**Verifikasi manual integral (Simpson n=4)** — dua contoh lengkap (item pemenang dan salah satu pembanding):

```
m2p-n002:  h = (1.6000-(-1.4000))/4 = 0.7500
  q1 = -0.6500 → P=0.342990 → K_q1=0.069393
  q3 =  0.8500 → P=0.700567 → K_q3=0.067734
  simpson(n=4) = (0.75/3)×[0.263490 + 4×0.069393 + 2×0 + 4×0.067734 + 0.252035]
               = 0.25×[0.263490+0.277572+0+0.270936+0.252035]
               = 0.25×1.064033 = 0.266008

m2p-r002:  h = (2.9500-(-0.0500))/4 = 0.7500
  q1 = 0.7000 → P=0.668188 → K_q1=0.049962
  q3 = 2.2000 → P=0.900250 → K_q3=0.036862
  simpson(n=4) = (0.75/3)×[0.222734 + 4×0.049962 + 2×0 + 4×0.036862 + 0.125295]
               = 0.25×[0.222734+0.199848+0+0.147448+0.125295]
               = 0.25×0.695325 = 0.173831
```

Tabel silang n=4 (manual) vs n=40 (kode) untuk semua item Round 4:

```
Item      simpson(n=4)   simpson(n=40, kode)   selisih
──────────────────────────────────────────────────────
m2p-v002    0.140441         0.140419          -0.000022
m2p-n002    0.266008         0.266355           0.000347
m2p-r002    0.173831         0.173915           0.000084
m2p-r003    0.135858         0.135822          -0.000036
```

##### Ranking Round 4

```
Rank  Item       Area          kl_score    z_hat    δ       Note
────────────────────────────────────────────────────────────────────────
  1   m2p-n002   numeric        0.266355   0.1000  1.5000   SELECTED
  2   m2p-r002   reasoning      0.173915   1.4500  1.5000
  3   m2p-v002   verbal         0.140419   1.8000  1.5000
  4   m2p-r003   reasoning      0.135822   1.8500  1.5000
```

**Selected: `m2p-n002`** — numeric item, `a=[0.3, 1.8, 0.4]`.

`m2p-n002` menang karena `z_hat = 0.10` paling dekat dengan pusat kurva ICC (`P≈0.5`, titik
kemiringan maksimum), sehingga K_i(z‖ẑ) tumbuh paling cepat begitu z bergeser menjauhi ẑ di
sepanjang interval `[-1.4, 1.6]`. Item lain (`v002`, `r003`) sudah berada jauh di sisi "mudah"
kurva ICC (`P_i(ẑ) > 0.85`) untuk examinee ini, sehingga K_i tumbuh jauh lebih lambat meskipun
`‖a‖²`-nya besar.

**Catatan skala:** `kl_score` Round 4 (~0.14–0.27) jauh lebih kecil dari Round 1 (~1.7–1.9) —
ini konsisten dengan teori: interval integrasi mengecil dari `δ=3.0` menjadi `δ=1.5` seiring m
bertambah, sehingga area di bawah `K_i(z‖ẑ)` yang diintegralkan pun mengecil.

---

#### Perbedaan dengan D-optimal dan A-optimal

```
D-optimal:            score = det(cum_FIM + item_FIM)              ← bergantung pada cum_FIM
A-optimal:            score = -tr((cum_FIM + item_FIM)⁻¹)          ← bergantung pada cum_FIM
KL-information:       score = ∫_{ẑ-δ}^{ẑ+δ} K_i(z‖ẑ) dz            ← TIDAK bergantung pada cum_FIM
```

Karena score KL information hanya bergantung pada parameter item itu sendiri (`a`, `d`, `c`) dan
θ̂ saat ini (bukan `cum_FIM`):
- **Tidak ada degenerate case** di round berapa pun (integral selalu finite dan ≥ 0)
- Greedy: hanya mempertimbangkan seberapa informatif item ini di sekitar θ̂ saat ini, tidak
  mempertimbangkan informasi yang sudah terkumpul dari item-item sebelumnya
- Interval integrasi `δ` mengecil seiring bertambahnya item yang di-administer (`m`), membuat
  kriteria makin "lokal" — mendekati perilaku Fisher information murni saat m besar, dan lebih
  "global" saat m kecil. Ini mekanisme adaptif yang tidak dimiliki D-optimal/A-optimal/MI.

---

## Implementasi KL Information di API

### Layer Implementasi

```
Layer            File                                    Peran
────────────────────────────────────────────────────────────────────────────────────
Core math        src/mcat/selection/kl_information.rs     score(θ,a,d,c,administered)
                                                          = ∫_{ẑ-δ}^{ẑ+δ} K_i(z‖ẑ) dz
Dispatcher       src/mcat/selection/mod.rs                score_item() dispatch ke KL/D-optimal/A-optimal/MI
Engine           src/mcat/engine.rs                       McatEngine::select_next_item() — score semua item, pick argmax
API Handler      src/handlers/session_handler.rs           Dipanggil di endpoint next_item
Konfigurasi      TestSettings.selection_method (String)   Ditentukan per-test-settings, di-parse saat runtime
```

### Cara Aktifkan KL Information

Set `selection_method = "kl_information"` di `TestSettings`.

```
"d_optimal"          → det(cum_FIM + item_FIM)
"a_optimal"          → -tr(inv(cum_FIM + item_FIM))
"kl_information"      → ∫_{ẑ-δ}^{ẑ+δ} K_i(z‖ẑ) dz     (δ = 3/√(administered+1))
```

### Signature

`score_item()` sekarang menerima parameter mentah item (`a`, `d`, `c`) selain `item_fim`, karena
KL information — tidak seperti 3 metode lain — bukan fungsi dari `item_FIM` semata:

```rust
pub fn score_item(
    method: &SelectionMethod,
    ctx: &SelectionContext,   // + field `administered: usize` (baru)
    item_fim: &DMatrix<f64>,
    a: &DVector<f64>,        // baru — dipakai KL untuk hitung z_hat = a·θ + d
    d: f64,                  // baru
    c: f64,                  // baru
) -> f64
```

### Catatan: Tidak Ada Fallback Dibutuhkan

KL-information selalu menghasilkan score finite dan ≥ 0 (Gibbs' inequality) untuk semua item —
tidak ada kasus degenerate seperti D-optimal atau A-optimal di round awal.

Dalam `score_item()`, untuk KL information, `updated_FIM = cum_FIM + item_FIM` tetap dihitung
(karena baris kode itu dibagi oleh semua 4 metode) tapi tidak digunakan — score KL dihitung
sepenuhnya dari `theta`, `a`, `d`, `c`, dan `ctx.administered`.

---

## Referensi

Rumus dan turunan pada dokumen ini diverifikasi langsung dari isi PDF sumber [1] dan [2]
(bukan dari abstrak atau ringkasan pihak ketiga) — keduanya open access dan dapat diunduh bebas
tanpa langganan, sehingga dapat dibandingkan langsung dengan sumber aslinya.

**[1]** Han, K. T. (2018). Components of the item selection algorithm in computerized adaptive
testing. *Journal of Educational Evaluation for Health Professions*, 15, Article 7.
https://doi.org/10.3352/jeehp.2018.15.7 — Open access (CC-BY), Korea Health Personnel Licensing
Examination Institute. PDF: https://www.jeehp.org/upload/pdf/jeehp-15-7.pdf (mirror:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5968224/). **Rumus K_i(θ‖θ₀) = Eq. 9 dan
K̄_i(θ₀) = ∫_{θ₀-δ}^{θ₀+δ} K_i(θ‖θ₀)dθ = Eq. 10, section "Kullback-Leibler information
criterion", halaman 6 dari 13 (PDF).** Bagian ini juga mengutip dan menjelaskan δ = c/√m dan
sitasi langsung ke Chang & Ying (1996) sebagai penemu metode ini.

**[2]** Sorrel, M. A., Barrada, J. R., de la Torre, J., & Abad, F. J. (2020). Adapting cognitive
diagnosis computerized adaptive testing item selection rules to traditional item response
theory. *PLOS ONE*, 15(1), e0227196. https://doi.org/10.1371/journal.pone.0227196 — Open access
(CC-BY). PDF: https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0227196&type=printable.
**Rumus item selection j = argmax ∫ KL_i(θ‖θ̂)·W(θ)dθ = Eq. 4 (halaman 3) dan
KL_i(θ‖θ̂) = P_i(θ̂)ln[P_i(θ̂)/P_i(θ)] + [1-P_i(θ̂)]ln[(1-P_i(θ̂))/(1-P_i(θ))] = Eq. 6
(halaman 4 dari 17)**, section "Global measures as alternatives". Formula pada [2] identik
dengan [1] (verifikasi silang independen dari dua sumber terpisah).

**[3]** Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive
testing. *Applied Psychological Measurement*, 20(3), 213–229.
https://doi.org/10.1177/014662169602000303 — **Sumber primer/asli** dari rumus KL information
di atas (dikutip langsung oleh [1] dan [2]). Pada penelusuran ini, salinan PDF paper aslinya
tidak dapat diakses secara bebas-unduh (SAGE Journals berbayar; salinan di University of
Minnesota Digital Conservancy — https://conservancy.umn.edu/items/6adad3f3-b76f-4c16-96a8-cfe2c743242c
— dan ResearchGate menolak akses/403 saat dicoba). Karena itu rumus pada dokumen ini diverifikasi
lewat [1] dan [2], bukan dikutip langsung dari [3] — kedua sumber tersebut mereproduksi rumus
Chang & Ying secara verbatim dengan nomor persamaan eksplisit, dan saling cocok satu sama lain,
sehingga memberi verifikasi silang independen atas rumus yang sama.

**[4]** Chang, H.-H., & Ying, Z. (2009). Nonlinear sequential designs for logistic item response
theory models with applications to computerized adaptive tests. *The Annals of Statistics*,
37(3), 1466–1488. https://doi.org/10.1214/08-AOS614 — Open access via arXiv:
https://arxiv.org/pdf/0906.1859 (diunduh dan dibaca langsung untuk penelusuran ini). Ditulis oleh
penulis yang sama dengan [3]; dipakai di sini hanya sebagai rujukan pendukung untuk definisi
model logistik 2PL/3PL, P(θ), Q(θ), dan Fisher information (Eq. 1.1–1.4, hal. 2-3) yang menjadi
dasar bersama `src/mcat/mirt.rs` — **bukan** sumber rumus KL information itu sendiri.
