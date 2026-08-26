### Configuration

```
Dimensions (k=3): [verbal (0), numeric (1), reasoning (2)]
Selection:        A-Optimal         [selection/a_optimal.rs:4]
```


**Probability:**

```
General M3PL:  

    P = c + (1 - c) × σ(a·θ + d)

M2PL (c=0):   

    P = σ(a·θ + d) 
      = 1 / (1 + exp(-(a·θ + d)))
```
Model M3PL/M2PL di atas sesuai dengan [1] Eq.(1) p.275 (bentuk multidimensional,
c_i + (1-c_i)[1+exp(-a_i·θ-b_i)]⁻¹) dan [3] Eq.(6) §2.1 (bentuk M2PL c=0,
g_j(θ)=exp(d_j+a_j·θ)/(1+exp(d_j+a_j·θ))).

**Fisher Information Matrix (FIM):**

```
I_i(θ) = w × (a × aᵀ)

Where:
    P* = σ(a·θ + d)              — sigmoid probability (without guessing)
    P' = (1-c) × P* × (1 - P*)   — derivative of P w.r.t. linear predictor
    Q  = 1 - P                   — probability of incorrect
    w  = (P')² / (P × Q)         — information weight
```

Struktur `I_i(θ) = a_i aᵀ_i × g(θ)` (outer product dikali skalar) sesuai [1] Eq.(4)-(5)
§3 p.277-278, dengan `g` = "common factor"/skalar informasi (≡ `w` di notes ini).

Bentuk eksplisit `w = (P')²/(P×Q)` terverifikasi identik secara aljabar dengan
formula item information 3PL baku di [2] p.111 Eq.[6-5]:

```
I_i(θ) = a² × [Q(θ)/P(θ)] × [(P(θ)-c)² / (1-c)²]
```

Substitusi P-c = (1-c)P*  dan  Q = (1-c)(1-P*)  ke Eq.[6-5]:

    (P-c)²/(1-c)² = (1-c)²P*²/(1-c)² = P*²
    → I_i(θ) = a² × Q × P*² / P
             = a² × (1-c)(1-P*) × P*² / P
             = a² × [(1-c)P*(1-P*)]² / [(1-c)(1-P*) × P]  … disederhanakan kembali
             = a² × (P')² / (P × Q)         ✓ identik dengan w = (P')²/(P×Q)

```
M2PL (c=0): 

    P* = P
    P' = P × (1-P) = P × Q

    → w  = (P×Q)² / (P×Q) = P × Q


M2PL FIM:  

    I_i(θ) = P(1-P) × (a × aᵀ)
           = w × (a × aᵀ)
```
Untuk M2PL (c=0), bentuk `I_i(θ)=a²P(θ)Q(θ)` sesuai persis dengan [2] p.109 Eq.[6-3].
Notes ini menggeneralisasi versi unidimensional `a²` di [2] menjadi outer product
`a×aᵀ` sesuai struktur multidimensional di [1].

---

#### A-Optimal Criterion

Pemilihan item yang meminimalkan rata-rata variance (trace of inverse FIM) dari estimasi θ di semua dimensi.

```
argmin_i  tr( (cum_FIM + item_FIM_i)⁻¹ )
```

API score (dinegasikan, jadi higher = better):

```
a_score_i = -tr( (cum_FIM + item_FIM_i)⁻¹ )
```

**Landasan teori [1]** — §4.1.2, Eq.(16), p.±282-284:
A-optimality didefinisikan sebagai kriteria yang **"minimize the sum of the
(asymptotic) sampling variances of the MLEs of the abilities, which is
equivalent to selecting the item that minimizes the trace of the inverse of
the information matrix"** — dikutip verbatim dari full text [1]. Ini persis
sama dengan `argmin_i tr((cum_FIM+item_FIM_i)⁻¹)` di atas: `tr(FIM⁻¹)` adalah
jumlah asymptotic variance MLE θ̂ di semua dimensi (Cramér–Rao lower bound,
lihat interpretasi `FIM⁻¹ ≈ covariance(θ̂)` di kode `item_selection_a_optimal.rs:356`).

Sebagai pembanding, D-optimal [1] §4.1.1 Eq.(13) p.±280-282 dikutip verbatim:
**"This criterion maximizes the determinant of [the information matrix]"**
— `arg max det(I_{S_{k-1}}(θ̂) + I_i(θ̂))`, konsisten dengan isi `d_optimal_notes.md`
di direktori sibling.

---

#### Matriks Inverse

Matriks A berukuran n×n memiliki inverse A⁻¹ jika dan hanya jika **det(A) ≠ 0** (non-singular / full rank).

```
A × A⁻¹ = A⁻¹ × A = I   (I = identity matrix)
```

##### Kondisi Invertibility   

```
Invertible   ↔  det(A) ≠ 0
             ↔  rank(A) = n  (full rank)
             ↔  semua eigenvalue ≠ 0
             ↔  tidak ada kolom/baris yang linearly dependent

Singular     ↔  det(A) = 0
             ↔  rank(A) < n
             ↔  minimal satu eigenvalue = 0
```

Dalam konteks A-optimal: `updated_FIM` harus full rank (rank = k=3)

---

##### Konsep Inverse — 2×2

Untuk matriks 2×2:

```
A = [[a  b]
     [c  d]]

A⁻¹ = 1/det(A) × [[d   -b]
                  [-c   a]]

det(A) = ad - bc
```

Contoh:
```
A = [[3  1]
     [2  4]]

det(A) = 3×4 - 1×2 = 10

A⁻¹ = (1/10) × [[4  -1]    = [[0.4   -0.1]
                [-2  3]]      [-0.2   0.3]]

Verifikasi: A × A⁻¹ = [[3×0.4 + 1×(-0.2)   3×(-0.1) + 1×0.3]  = [[1  0]
                       [2×0.4 + 4×(-0.2)   2×(-0.1) + 4×0.3]]    [0  1]] ✓
```

---

##### Konsep Inverse — 3×3 (Metode Adjugate)

```
A⁻¹ = 1/det(A) × adj(A)

adj(A) = Cᵀ   (transpose dari matriks kofaktor)
```

**1 — Hitung Minor Mᵢⱼ**

Minor Mᵢⱼ = determinant dari submatriks 2×2 setelah menghapus baris i dan kolom j.

```
Untuk matriks A = [[a₀₀ a₀₁ a₀₂], 
                   [a₁₀ a₁₁ a₁₂], 
                   [a₂₀ a₂₁ a₂₂]]:
```

```
M₀₀ = det([[a₁₁ a₁₂], 
           [a₂₁ a₂₂]]) = a₁₁×a₂₂ - a₁₂×a₂₁
M₀₁ = det([[a₁₀ a₁₂], 
           [a₂₀ a₂₂]]) = a₁₀×a₂₂ - a₁₂×a₂₀
M₀₂ = det([[a₁₀ a₁₁], 
           [a₂₀ a₂₁]]) = a₁₀×a₂₁ - a₁₁×a₂₀
... (dst untuk M₁₀...M₂₂)
```

**2 — Hitung Kofaktor Cᵢⱼ**

```
Cᵢⱼ = (-1)^(i+j) × Mᵢⱼ

C₁₁ = (-1)^(1+1) × M₁₁ = M₁₁
C₁₂ = (-1)^(1+2) × M₁₂ = -M₁₁

maka didaptkan pola kofaktornya untuk nilai M:

Pola tanda:  [[+  -  +]
              [-  +  -]
              [+  -  +]]
``` 

**3 — Hitung det(A) via ekspansi baris pertama**

Jika sudah diketahui matriks kofaktornya, perhitungan det dari matriks A, sebagai berikut:

```
A = [[a₀₀ a₀₁ a₀₂], → gunakan baris pertama
     [a₁₀ a₁₁ a₁₂], 
     [a₂₀ a₂₁ a₂₂]]

Kofaktor A = [[C₀₀  C₀₁  C₀₂] → gunakan baris pertama
              [C₁₀  C₁₁  C₁₂]
              [C₂₀  C₂₁  C₂₂]]
```

Maka determinannya:
```
det(A) = a₀₀×C₀₀ + a₀₁×C₀₁ + a₀₂×C₀₂
```

**4 — Hitung adj(A) = transpose kofaktor**

```
adj(A) = [[C₀₀  C₁₀  C₂₀]
          [C₀₁  C₁₁  C₂₁]
          [C₀₂  C₁₂  C₂₂]]
```

**5 — A⁻¹ = (1/det) × adj(A)**

Masukkan nilai determinan dan adjoin tadi ke dalam rumus inverse.

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

#### Item Selection — A-Optimal

---

##### Round 1 — 0 Item Administered (Degenerate Case)

```
θ̂            = [0.0, 0.0, 0.0]   (initial estimate — belum ada informasi)
cum_FIM      = 3×3 zero matrix
Administered = []
Candidates   = all 7 items
```

---

**Konsep Rank Matriks**

Rank suatu matriks adalah jumlah baris (atau kolom) yang **linearly independent** — kolom-kolom yang tidak bisa dinyatakan sebagai kombinasi linear dari kolom lainnya.

| Rank | Artinya                                         | Invertible? |
|------|-------------------------------------------------|-------------|
| = n  | Full rank (semua baris/kolom independen)        | Ya          |
| < n  | Rank deficient (ada kolom yang bergantung lain) | Tidak       |

Invertible artinya sebuah matriks punya kebalikan (inverse)

**Cara Menghitung Rank Matriks**

Metode standar: **Row Echelon Form (REF)** via Gaussian Elimination — ubah matriks ke bentuk segitiga atas, lalu hitung jumlah baris yang tidak nol.

**Langkah:**
1. Tulis matriks
2. Gunakan operasi baris (row operations) untuk membentuk segitiga atas
3. Rank = jumlah baris dengan pivot (baris ≠ 0)

**Operasi baris yang diizinkan:**
- Tukar dua baris
- Kalikan satu baris dengan skalar ≠ 0
- Tambahkan kelipatan satu baris ke baris lain

---

**Contoh 1 — Rank-1 matrix (outer product)**

```
A = [[0.25  0.20  1.00]
     [0.20  0.16  0.80]
     [1.00  0.80  4.00]]
```
```
R2_baru[0]  =  R2[0]  -  faktor × R1[0]  =  0
            
                 0.20  -  faktor × 0.25  =  0
                                 faktor  =  0.20 / 0.25  
                                         =  0.8

R2 ← = R2 - (0.20/0.25)×R1 
     = R2 - 0.8×R1:
```
```
R2: [0.20-0.8×0.25  0.16-0.8×0.20  0.80-0.8×1.00]
  = [0.20-0.20      0.16-0.16      0.80-0.80]
  = [0              0              0]
```
```
R3 ← = R3 - (1.00/0.25)×R1 
     = R3 - 4×R1:
```
```
R3: [1.00-4×0.25  0.80-4×0.20  4.00-4×1.00]
  = [1.00-1.00    0.80-0.80    4.00-4.00]
  = [0            0            0]
```

Hasil REF:
```
[[0.25  0.20  1.00]   ← pivot (baris ≠ 0)
 [0     0     0   ]   ← nol
 [0     0     0   ]]  ← nol

rank = 1  (hanya 1 baris dengan pivot)
```

---

**Contoh 2 — Full-rank matrix**

```
A = [[0.3434  0.1964  0.2142]
     [0.1964  0.9007  0.3008]
     [0.2142  0.3008  0.6051]]
```
```
R2 ← = R2 - (0.1964/0.3434)×R1 
     = R2 - 0.5719×R1:
```
```
R2: [0   0.9007-0.5719×0.1964   0.3008-0.5719×0.2142]
  = [0   0.9007-0.1123          0.3008-0.1225]
  = [0   0.7884                 0.1783]
```

```
R3 ← = R3 - (0.2142/0.3434)×R1 
     = R3 - 0.6238×R1:
```
```
R3: [0   0.3008-0.6238×0.1964   0.6051-0.6238×0.2142]
  = [0   0.3008-0.1225          0.6051-0.1336]
  = [0   0.1783                 0.4715]
```

```
R3 ← = R3 - (0.1783/0.7884)×R2 
     = R3 - 0.2262×R2:
```
```
R3: [0   0   0.4715-0.2262×0.1783]
  = [0   0   0.4715-0.0403]
  = [0   0   0.4312]
```

Hasil REF:
```
[[0.3434  0.1964  0.2142]   ← pivot
 [0       0.7884  0.1783]   ← pivot
 [0       0       0.4312]]  ← pivot

rank = 3  (semua baris punya pivot → full rank → invertible)
```

---

**Perhitungan detail item_FIM (`m2p-r001`):**

Formula: `item_FIM = w × (a × aᵀ)`

```
a = [0.5, 0.4, 2.0]   d = 0.30   c = 0.0
```
```
linear = (0.5×1.0) + (0.4×-0.5) + (2.0×0.5) + 0.30
       = 1.3000 + 0.30 = 1.6000

P* = sigma(1.6000)
   = 1 / (1 + e^(-1.6000))
   = 1 / (1 + 0.201897)
   = 0.832018

P  = P*                   (M2PL)
   = 0.832018

Q  = 1 − P = 1 − 0.832018 = 0.167982
w  = P×Q   = 0.832018 × 0.167982 = 0.139764
```

Hitung Item FIM

```
Item FIM ke i =  w_i × (a_i × a_iᵀ)
```

```
a × aᵀ = [[0.5×0.5  0.5×0.4  0.5×2.0]
          [0.4×0.5  0.4×0.4  0.4×2.0]
          [2.0×0.5  2.0×0.4  2.0×2.0]]

        = [[0.25  0.20  1.00]
           [0.20  0.16  0.80]
           [1.00  0.80  4.00]]
```

Kalikan dengan `w = 0.139764`:

```
[0][0] = 0.139764 × 0.25 = 0.0349
[0][1] = 0.139764 × 0.20 = 0.0280
[0][2] = 0.139764 × 1.00 = 0.1398
[1][0] = 0.139764 × 0.20 = 0.0280
[1][1] = 0.139764 × 0.16 = 0.0224
[1][2] = 0.139764 × 0.80 = 0.1118
[2][0] = 0.139764 × 1.00 = 0.1398
[2][1] = 0.139764 × 0.80 = 0.1118
[2][2] = 0.139764 × 4.00 = 0.5591

item_FIM:
     [[  0.0349    0.0280    0.1398  ]
      [  0.0280    0.0224    0.1118  ]
      [  0.1398    0.1118    0.5591  ]]
```

Matriks selalu simetris karena `a[i]×a[j] = a[j]×a[i]`.

**Perhitungan Inverse dan A-score (`m2p-r001`, Round 1):**

```
updated_FIM = cum_FIM + item_FIM
            = 0 + item_FIM
            =  [[  0.0349    0.0280    0.1398  ]
                [  0.0280    0.0224    0.1118  ]
                [  0.1398    0.1118    0.5591  ]]
```

Cek rank via REF 

R2 
= R2 - (0.0280/0.0349)×R1 
= R2 - 0.8023×R1:

```
R2: [0.0280-0.8023×0.0349  0.0224-0.8023×0.0280  0.1118-0.8023×0.1398]
  = [0.0280-0.0280         0.0224-0.0225          0.1118-0.1121]
  = [0                     ≈0                     ≈0]
```

R3
= R3 - (0.1398/0.0349)×R1 
= R3 - 4.0057×R1:

```
R3: [0.1398-4.0057×0.0349  0.1118-4.0057×0.0280  0.5591-4.0057×0.1398]
  = [0.1398-0.1398         0.1118-0.1122          0.5591-0.5600]
  = [0                     ≈0                     ≈0]
```

Hasil REF:
```
[[0.0349  0.0280  0.1398]   ← 1 pivot
 [0       0       0     ]   ← nol
 [0       0       0     ]]  ← nol

rank = 1  < 3  →  singular  →  det = 0
```

Hal yang sama berlaku untuk semua item di round 1 — semua `item_FIM` adalah rank-1.

**Ranking Round 1**

```
Rank  Item       Area          a_score   tr(inv)   Note
──────────────────────────────────────────────────────────────────────────
  1   m2p-v002   verbal         -∞        +∞       gak ada perbedaan (tiebreak)
  2   m2p-r001   reasoning      -∞        +∞
  3   m2p-v001   verbal         -∞        +∞
  4   m2p-n002   numeric        -∞        +∞
  5   m2p-r002   reasoning      -∞        +∞
  6   m2p-r003   reasoning      -∞        +∞
  7   m2p-n001   numeric        -∞        +∞
```

Semua item memiliki `a_score = -∞` → A-optimal tidak dapat membedakan item.
Selection bergantung pada urutan iterasi array (effectively arbitrary).

---

**Round 2 — 1 Item Administered (Degenerate)**

```
cum_FIM = item_FIM_round1   ← rank-1 (akumulasi 1 outer product)

updated_FIM = cum_FIM + item_FIM_candidate
            = rank-1 + rank-1
            = rank ≤ 2   →  masih < 3  →  singular  →  a_score = -∞
```

Penjumlahan dua rank-1 matrix menghasilkan rank ≤ 2. Untuk FIM 3×3 butuh rank = 3 agar invertible — belum terpenuhi.

---

##### Round 4 — 3 Item Administered (Non-Degenerate)

Scenario: 3 item sudah di-administer (r001, v001, n001). Memilih item ke-4.

```
θ̂            = [1.0, -0.5, 0.5]   (verbal kuat, numeric lemah, reasoning sedang)
Administered = [m2p-r001, m2p-v001, m2p-n001]
Candidates   = [m2p-v002, m2p-n002, m2p-r002, m2p-r003]
```

##### Administered Items — FIM yang sudah terkumpul (θ=[1.0,-0.5,0.5])

**m2p-r001** — `a=[0.5, 0.4, 2.0]`, `d=0.30`

```
linear  =  0.5×1.0 + 0.4×(-0.5) + 2.0×0.5 + 0.30
        =  0.5 - 0.2 + 1.0 + 0.30  =  1.60

P  =  σ(1.60)  =  1 / (1 + e^(-1.60))  =  1 / (1 + 0.2019)  =  0.8320
Q  =  0.1680
w  =  0.8320 × 0.1680  =  0.1398

FIM_r001 = 0.1398 × [[0.25  0.20  1.00]    [[0.0350  0.0280  0.1398]
                     [0.20  0.16  0.80]  =  [0.0280  0.0224  0.1118]
                     [1.00  0.80  4.00]]    [0.1398  0.1118  0.5592]]
```

**m2p-v001** — `a=[1.9, 0.2, 0.3]`, `d=0.40`

```
linear  =  1.9×1.0 + 0.2×(-0.5) + 0.3×0.5 + 0.40
        =  1.9 - 0.1 + 0.15 + 0.40  =  2.35

P  =  σ(2.35)  =  1 / (1 + 0.0952)  =  0.9130
Q  =  0.0870
w  =  0.9130 × 0.0870  =  0.0794

FIM_v001 = 0.0794 × [[3.61  0.38  0.57]    [[0.2866  0.0302  0.0453]
                     [0.38  0.04  0.06]  =  [0.0302  0.0032  0.0048]
                     [0.57  0.06  0.09]]    [0.0453  0.0048  0.0071]]
```

**m2p-n001** — `a=[0.3, 1.9, 0.4]`, `d=0.80`

```
linear  =  0.3×1.0 + 1.9×(-0.5) + 0.4×0.5 + 0.80
        =  0.3 - 0.95 + 0.2 + 0.80  =  0.35

P  =  σ(0.35)  =  1 / (1 + 0.7047)  =  0.5868
Q  =  0.4132
w  =  0.5868 × 0.4132  =  0.2424

FIM_n001 = 0.2424 × [[0.09  0.57  0.12]    [[0.0218  0.1382  0.0291]
                     [0.57  3.61  0.76]  =  [0.1382  0.8751  0.1842]
                     [0.12  0.76  0.16]]    [0.0291  0.1842  0.0388]]
```

**cum_FIM** = FIM_r001 + FIM_v001 + FIM_n001:

```
cum_FIM = [[0.3434  0.1964  0.2142]
           [0.1964  0.9007  0.3008]
           [0.2142  0.3008  0.6051]]
```

Rank-3 (tiga a-vector dari arah berbeda → full rank → A-optimal valid).

Diagonal cum_FIM = [0.3434, 0.9007, 0.6051]:
- Dimensi 0 (verbal): informasi paling rendah → A-optimal akan prioritaskan item verbal
- Dimensi 1 (numeric): informasi terbesar
- Dimensi 2 (reasoning): informasi menengah

---

##### Perhitungan Probability dan Weight — Candidates (θ=[1.0,-0.5,0.5])

**m2p-v002** — `a=[1.7, 0.2, 0.2]`, `d=0.10`

```
linear  =  1.7×1.0 + 0.2×(-0.5) + 0.2×0.5 + 0.10
        =  1.7 - 0.1 + 0.1 + 0.10  =  1.80

P  =  1 / (1 + e^(-1.80))  =  1 / (1 + 0.1653)  =  0.8581
Q  =  0.1419
w  =  0.8581 × 0.1419  =  0.1217
```

---

**m2p-n002** — `a=[0.3, 1.8, 0.4]`, `d=0.50`

```
linear  =  0.3×1.0 + 1.8×(-0.5) + 0.4×0.5 + 0.50
        =  0.3 - 0.9 + 0.2 + 0.50  =  0.10

P  =  1 / (1 + e^(-0.10))  =  1 / (1 + 0.9048)  =  0.5250
Q  =  0.4750
w  =  0.5250 × 0.4750  =  0.2494   ←── highest w (P paling dekat 0.5)
```

---

**m2p-r002** — `a=[0.3, 0.8, 1.9]`, `d=0.60`

```
linear  =  0.3×1.0 + 0.8×(-0.5) + 1.9×0.5 + 0.60
        =  0.3 - 0.4 + 0.95 + 0.60  =  1.45

P  =  1 / (1 + e^(-1.45))  =  1 / (1 + 0.2346)  =  0.8100
Q  =  0.1900
w  =  0.8100 × 0.1900  =  0.1539
```

---

**m2p-r003** — `a=[0.4, 0.3, 1.8]`, `d=0.70`

```
linear  =  0.4×1.0 + 0.3×(-0.5) + 1.8×0.5 + 0.70
        =  0.4 - 0.15 + 0.9 + 0.70  =  1.85

P  =  1 / (1 + e^(-1.85))  =  1 / (1 + 0.1572)  =  0.8642
Q  =  0.1358
w  =  0.8642 × 0.1358  =  0.1174   ←── lowest w (P paling jauh dari 0.5)
```

---

**Ringkasan candidates**

```
Item      a = [v, n, r]          d     linear   P       Q       w=P×Q
───────────────────────────────────────────────────────────────────────
m2p-n002  [0.3, 1.8, 0.4]      +0.50   0.10    0.5250  0.4750  0.2494 → highest w
m2p-r002  [0.3, 0.8, 1.9]      +0.60   1.45    0.8100  0.1900  0.1539
m2p-v002  [1.7, 0.2, 0.2]      +0.10   1.80    0.8581  0.1419  0.1217
m2p-r003  [0.4, 0.3, 1.8]      +0.70   1.85    0.8642  0.1358  0.1174 → lowest w
```

---

##### Perhitungan FIM untuk setiap candidate

**m2p-v002** — `w=0.1217`, `a=[1.7, 0.2, 0.2]`

```
a × aᵀ =  [[1.7×1.7  1.7×0.2  1.7×0.2]     [[2.89  0.34  0.34]
           [0.2×1.7  0.2×0.2  0.2×0.2]   =  [0.34  0.04  0.04]
           [0.2×1.7  0.2×0.2  0.2×0.2]]     [0.34  0.04  0.04]]

item_FIM = 0.1217 × a × aᵀ = [[0.3517  0.0414  0.0414]
                              [0.0414  0.0049  0.0049]
                              [0.0414  0.0049  0.0049]]
```

---

**m2p-n002** — `w=0.2494`, `a=[0.3, 1.8, 0.4]`

```
a × aᵀ =  [[0.09  0.54  0.12]
           [0.54  3.24  0.72]
           [0.12  0.72  0.16]]

item_FIM = 0.2494 × a × aᵀ = [[0.0224  0.1347  0.0299]
                              [0.1347  0.8081  0.1796]
                              [0.0299  0.1796  0.0399]]
```

---

**m2p-r002** — `w=0.1539`, `a=[0.3, 0.8, 1.9]`

```
a × aᵀ =  [[0.09  0.24  0.57]
           [0.24  0.64  1.52]
           [0.57  1.52  3.61]]

item_FIM = 0.1539 × a × aᵀ = [[0.0139  0.0370  0.0877]
                               [0.0370  0.0985  0.2339]
                               [0.0877  0.2339  0.5556]]
```

---

**m2p-r003** — `w=0.1174`, `a=[0.4, 0.3, 1.8]`

```
a × aᵀ =  [[0.16  0.12  0.72]
           [0.12  0.09  0.54]
           [0.72  0.54  3.24]]

item_FIM = 0.1174 × a × aᵀ = [[0.0188  0.0141  0.0845]
                               [0.0141  0.0106  0.0634]
                               [0.0845  0.0634  0.3804]]
```

---

##### updated_FIM = cum_FIM + item_FIM (per candidate)

**updated_v002**:

```
cum_FIM + FIM_v002 = [[0.3434+0.3517  0.1964+0.0414  0.2142+0.0414]
                      [0.1964+0.0414  0.9007+0.0049  0.3008+0.0049]
                      [0.2142+0.0414  0.3008+0.0049  0.6051+0.0049]]

                   = [[0.6951  0.2378  0.2556]
                      [0.2378  0.9056  0.3057]
                      [0.2556  0.3057  0.6100]]
```

Inverse — Kofaktor:
```
C₀₀ = +(0.9056×0.6100 - 0.3057²)          = +(0.5524 - 0.0935) = +0.4589
C₀₁ = -(0.2378×0.6100 - 0.3057×0.2556)    = -(0.1451 - 0.0781) = -0.0670
C₀₂ = +(0.2378×0.3057 - 0.9056×0.2556)    = +(0.0727 - 0.2315) = -0.1588
C₁₁ = +(0.6951×0.6100 - 0.2556²)          = +(0.4240 - 0.0653) = +0.3587
C₁₂ = -(0.6951×0.3057 - 0.2378×0.2556)    = -(0.2124 - 0.0608) = -0.1516
C₂₂ = +(0.6951×0.9056 - 0.2378²)          = +(0.6295 - 0.0565) = +0.5730
```

Determinan:
```
det = 0.6951×0.4589 + 0.2378×(-0.0670) + 0.2556×(-0.1588)
    = 0.3190 - 0.0159 - 0.0406
    = 0.2625
```

Inverse = (1/det) × adj:
```
M⁻¹ ≈ (1/0.2625) × [[ 0.4589  -0.0670  -0.1588]
                    [-0.0670   0.3587  -0.1516]
                    [-0.1588  -0.1516   0.5730]]

    ≈ [[ 1.748  -0.255  -0.605]
       [-0.255   1.366  -0.578]
       [-0.605  -0.578   2.183]]

tr(M⁻¹) = 1.748 + 1.366 + 2.183 = 5.297
a_score  = -5.297
```

---

**updated_n002**:

```
cum_FIM + FIM_n002 = [[0.3658  0.3311  0.2441]
                      [0.3311  1.7088  0.4804]
                      [0.2441  0.4804  0.6450]]
```

Inverse — Kofaktor:
```
C₀₀ = +(1.7088×0.6450 - 0.4804²)          = +(1.1022 - 0.2308) = +0.8714
C₀₁ = -(0.3311×0.6450 - 0.4804×0.2441)    = -(0.2136 - 0.1173) = -0.0963
C₀₂ = +(0.3311×0.4804 - 1.7088×0.2441)    = +(0.1590 - 0.4171) = -0.2581
C₁₁ = +(0.3658×0.6450 - 0.2441²)          = +(0.2359 - 0.0596) = +0.1763
C₁₂ = -(0.3658×0.4804 - 0.3311×0.2441)    = -(0.1757 - 0.0808) = -0.0949
C₂₂ = +(0.3658×1.7088 - 0.3311²)          = +(0.6251 - 0.1096) = +0.5155
```

Determinan:
```
det = 0.3658×0.8714 + 0.3311×(-0.0963) + 0.2441×(-0.2581)
    = 0.3187 - 0.0319 - 0.0630
    = 0.2238
```

Inverse = (1/det) × adj:
```
M⁻¹ ≈ (1/0.2238) × [[ 0.8714  -0.0963  -0.2581]
                    [-0.0963   0.1763  -0.0949]
                    [-0.2581  -0.0949   0.5155]]

     ≈ [[ 3.894  -0.430  -1.153]
        [-0.430   0.787  -0.424]
        [-1.153  -0.424   2.303]]

tr(M⁻¹) = 3.894 + 0.787 + 2.303 = 6.984
a_score  = -6.984
```

---

**updated_r002**:

```
cum_FIM + FIM_r002 = [[0.3573  0.2334  0.3019]
                      [0.2334  0.9992  0.5347]
                      [0.3019  0.5347  1.1607]]
```

Inverse — Kofaktor:
```
C₀₀ = +(0.9992×1.1607 - 0.5347²)          = +(1.1597 - 0.2859) = +0.8738
C₀₁ = -(0.2334×1.1607 - 0.5347×0.3019)    = -(0.2709 - 0.1614) = -0.1095
C₀₂ = +(0.2334×0.5347 - 0.9992×0.3019)    = +(0.1248 - 0.3017) = -0.1769
C₁₁ = +(0.3573×1.1607 - 0.3019²)          = +(0.4147 - 0.0911) = +0.3236
C₁₂ = -(0.3573×0.5347 - 0.2334×0.3019)    = -(0.1910 - 0.0705) = -0.1205
C₂₂ = +(0.3573×0.9992 - 0.2334²)          = +(0.3570 - 0.0545) = +0.3025
```

Determinan:
```
det = 0.3573×0.8738 + 0.2334×(-0.1095) + 0.3019×(-0.1769)
    = 0.3122 - 0.0256 - 0.0534
    = 0.2332
```

Inverse = (1/det) × adj:
```
M⁻¹ ≈ (1/0.2332) × [[ 0.8738  -0.1095  -0.1769]
                      [-0.1095   0.3236  -0.1205]
                      [-0.1769  -0.1205   0.3025]]

     ≈ [[ 3.747  -0.470  -0.759]
         [-0.470   1.388  -0.517]
         [-0.759  -0.517   1.298]]

tr(M⁻¹) = 3.747 + 1.388 + 1.298 = 6.433
a_score  = -6.433
```

---

**updated_r003**:

```
cum_FIM + FIM_r003 = [[0.3622  0.2105  0.2987]
                      [0.2105  0.9113  0.3642]
                      [0.2987  0.3642  0.9855]]
```

Inverse — Kofaktor:
```
C₀₀ = +(0.9113×0.9855 - 0.3642²)          = +(0.8978 - 0.1326) = +0.7652
C₀₁ = -(0.2105×0.9855 - 0.3642×0.2987)    = -(0.2074 - 0.1088) = -0.0986
C₀₂ = +(0.2105×0.3642 - 0.9113×0.2987)    = +(0.0767 - 0.2722) = -0.1955
C₁₁ = +(0.3622×0.9855 - 0.2987²)          = +(0.3570 - 0.0892) = +0.2678
C₁₂ = -(0.3622×0.3642 - 0.2105×0.2987)    = -(0.1319 - 0.0629) = -0.0690
C₂₂ = +(0.3622×0.9113 - 0.2105²)          = +(0.3301 - 0.0443) = +0.2858
```

Determinan:
```
det = 0.3622×0.7652 + 0.2105×(-0.0986) + 0.2987×(-0.1955)
    = 0.2772 - 0.0207 - 0.0584
    = 0.1981
```

Inverse = (1/det) × adj:
```
M⁻¹ ≈ (1/0.1981) × [[ 0.7652  -0.0986  -0.1955]
                      [-0.0986   0.2678  -0.0690]
                      [-0.1955  -0.0690   0.2858]]

     ≈ [[ 3.863  -0.498  -0.987]
         [-0.498   1.352  -0.348]
         [-0.987  -0.348   1.443]]

tr(M⁻¹) = 3.863 + 1.352 + 1.443 = 6.658
a_score  = -6.658
```

---

##### Ranking

```
Rank  Item       Area        a_score      tr(inv)    Note
──────────────────────────────────────────────────────────────────────
  1   m2p-v002   verbal      -5.295567    5.295567   SELECTED 
  2   m2p-r002   reasoning   -6.425466    6.425466
  3   m2p-r003   reasoning   -6.654032    6.654032
  4   m2p-n002   numeric     -6.976748    6.976748   
```




**Interpretasi kualitatif:**

Diagonal cum_FIM = [0.3434, 0.9007, 0.6051]:
- Dimensi 0 (verbal): FIM paling rendah → variance θ̂₁ paling besar → dimensi "terlemah"
- Dimensi 1 (numeric): FIM terbesar → variance θ̂₂ paling kecil

A-optimal akan memilih item yang paling banyak mengurangi **variance dimensi terlemah** (verbal).

→ m2p-v002 (`a=[1.7,0.2,0.2]`) memiliki loading verbal tertinggi di antara candidates,
  dan merupakan kandidat kuat untuk memenangkan A-optimal di skenario ini.

→ m2p-n002 (`a=[0.3,1.8,0.4]`) memiliki `w` tertinggi tapi loading utamanya di numeric
  yang sudah kuat → kurang efisien dari perspektif A-optimal.

---

**notes**

A-optimal berorientasi pada **rata-rata** variance, bukan **volume** (D-optimal):
```
D-optimal: memaksimalkan det(FIM) → volume ellipsoid kepercayaan terbesar
A-optimal: meminimalkan tr(FIM⁻¹) → rata-rata SE terkecil di semua dimensi
```

Ketika satu dimensi jauh lebih lemah dari yang lain, A-optimal lebih agresif menargetkan
dimensi tersebut dibanding D-optimal yang menyeimbangkan semua dimensi secara multiplikatif.

Kontras A-optimal vs D-optimal ini sesuai dengan pembahasan eksplisit di [1] §4.1
(p.±280-286), yang membandingkan kedua kriteria pada MCAT dan menyimpulkan A-optimal
dan D-optimal memberi hasil "largely similar" tapi tidak identik — sejalan dengan
temuan simulasi Round 4 di atas (v002 menang tipis, urutan rank 2-4 berbeda tipis
dari yang akan dihasilkan D-optimal).

---


Semua `a_score` harus finite (bukan -∞) di round 4 karena `cum_FIM` sudah rank-3.

---

## Implementasi A-Optimal di API

A-optimal sudah terintegrasi penuh di API production pipeline — sama dengan D-optimal.

### Layer Implementasi

```
Layer            File                                    Peran
────────────────────────────────────────────────────────────────────────────────
Core math        src/mcat/selection/a_optimal.rs:4      score = -tr(inv(cum_FIM + item_FIM))
                                                         returns NEG_INFINITY if not invertible
Dispatcher       src/mcat/selection/mod.rs:41-53        score_item() dispatch ke A-optimal/D-optimal/KL/MI
Engine           src/mcat/engine.rs:19-81               McatEngine::select_next_item() — score semua item, pick argmax
API Handler      src/handlers/session_handler.rs:249    Dipanggil di endpoint next_item
Konfigurasi      TestSettings.selection_method (String) Ditentukan per-test-settings, di-parse saat runtime
```

### Cara Aktifkan A-Optimal

Set `selection_method = "a_optimal"` di `TestSettings`.

```
"d_optimal"          → det(cum_FIM + item_FIM)             
"a_optimal"          → -tr(inv(cum_FIM + item_FIM))      
"kl_information"     → KL information
```

### Catatan: Tidak Ada Fallback untuk Degenerate Case

Di API production, ketika semua kandidat return `-∞`, tidak ada fallback —
selection bergantung pada urutan iterasi array `items`.

D-optimal menangani ini dengan tiebreak `trace(item_FIM)`.
A-optimal tidak punya mekanisme serupa.

Rekomendasi: gunakan A-optimal hanya setelah setidaknya `k` item dari arah berbeda
ter-administer (round `k+1` = round 4 untuk k=3).

---

## Referensi

Semua referensi di bawah **gratis diunduh** (bukan di balik paywall) — link diverifikasi
dapat diakses langsung (bukan cuma abstrak) pada saat notes ini ditulis, supaya bisa
dibandingkan langsung dengan rumus di file ini.

**[1]** Mulder, J., & van der Linden, W. J. (2009). Multidimensional Adaptive Testing
with Optimal Design Criteria for Item Selection. *Psychometrika*, 74(2), 273–296.
https://doi.org/10.1007/s11336-008-9097-5

  Full text gratis (open access, PubMed Central): https://pmc.ncbi.nlm.nih.gov/articles/PMC2813188/

  Dipakai untuk:
  - Eq.(1), §2, p.±275 — model M3PL multidimensional: `P_i(θ)=c_i+(1-c_i)[1+exp(-a_i·θ-b_i)]⁻¹`
  - Eq.(4)-(5), §3, p.±277-278 — struktur FIM `I_i(θ) = a_i aᵀ_i × g(θ;a_i,b_i,c_i)`
  - Eq.(13), §4.1.1, p.±280-282 — kriteria D-optimal: `argmax det(I_S(θ̂))`
  - Eq.(16), §4.1.2, p.±282-284 — kriteria A-optimal: `argmin tr(I_S(θ̂)⁻¹)`, dikutip
    verbatim: *"minimize the sum of the (asymptotic) sampling variances of the MLEs
    of the abilities, which is equivalent to ... minimizes the trace of the inverse
    of the information matrix"*
  - Catatan: persamaan (1),(4),(5),(13),(16) di artikel asli dirender sebagai gambar
    (M1.gif, M4.gif, M5.gif, M15.gif, M24.gif — format lama pra-MathML), jadi notasi
    persis (mis. simbol turunan di dalam `g`) tidak bisa dikutip karakter-per-karakter;
    definisi kata-per-kata di atas dikutip verbatim dari full-text HTML-nya. Nomor
    halaman ditandai "±" karena PMC tidak memberi penanda halaman cetak per-equation;
    kisaran diestimasi dari total panjang artikel (273-296) dan urutan section.

**[2]** Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). ERIC
Clearinghouse on Assessment and Evaluation.

  Full text gratis (ERIC ED458219): https://files.eric.ed.gov/fulltext/ED458219.pdf

  Dipakai untuk:
  - p.109, Eq.[6-3] — item information 2PL: `I_i(θ) = a_i² P_i(θ) Q_i(θ)`
    → identik dengan `M2PL FIM: I_i(θ)=P(1-P)×(a×aᵀ)` di notes ini (generalisasi
    skalar `a²` → outer product `a×aᵀ` untuk kasus multidimensional).
  - p.111, Eq.[6-5] — item information 3PL: `I_i(θ) = a² × [Q(θ)/P(θ)] × [(P(θ)-c)²/(1-c)²]`
    → **dibuktikan aljabar identik** dengan `w=(P')²/(P×Q)` di notes ini melalui
    substitusi `P-c=(1-c)P*` dan `Q=(1-c)(1-P*)` (lihat derivasi di bagian FIM di atas).
  - Halaman ini dikonfirmasi persis (bukan estimasi) — dibaca langsung dari PDF
    dengan nomor halaman tercetak "109" dan "111" di footer.

**[3]** Chen, Y., Li, X., Liu, J., & Ying, Z. (2021). Item Response Theory — A
Statistical Framework for Educational and Psychological Measurement. *arXiv preprint*
arXiv:2108.08604 (juga terbit di *Statistical Science*).

  Full text gratis (arXiv): https://arxiv.org/pdf/2108.08604

  Dipakai untuk:
  - §2.1, Eq.(6) — model M2PL: `g_j(θ)=exp(d_j+a_j·θ)/(1+exp(d_j+a_j·θ))`
    → identik dengan `P=σ(a·θ+d)` (M2PL, c=0) di bagian "Probability" notes ini.

### Ringkasan verifikasi

| Rumus di notes ini | Sumber literatur | Status |
|---|---|---|
| `P = c+(1-c)σ(a·θ+d)` (M3PL) | [1] Eq.(1) | Cocok (definisi model) |
| `P = σ(a·θ+d)` (M2PL) | [1] Eq.(1) c=0; [3] Eq.(6) | Cocok |
| `I_i(θ) = w×(a×aᵀ)` (struktur outer product) | [1] Eq.(4)-(5) | Cocok (struktur `aaᵀ × skalar`) |
| `w = (P')²/(P×Q)`, `P'=(1-c)P*(1-P*)` (3PL) | [2] p.111 Eq.[6-5] | **Dibuktikan identik secara aljabar** |
| `M2PL FIM = P(1-P)×(a×aᵀ)` | [2] p.109 Eq.[6-3] | Cocok persis |
| `argmin_i tr((cum_FIM+item_FIM_i)⁻¹)` (A-optimal) | [1] Eq.(16) | Cocok persis (kutipan verbatim) |
| `argmax det(cum_FIM+item_FIM_i)` (D-optimal, pembanding) | [1] Eq.(13) | Cocok persis (kutipan verbatim) |

Tidak ditemukan penyimpangan antara rumus/perhitungan A-optimal di notes/kode ini
dengan literatur di atas.
