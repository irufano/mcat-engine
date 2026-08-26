### Configuration

```
Dimensions (k=3): [verbal (0), numeric (1), reasoning (2)]
Selection:        D-Optimal         [selection/d_optimal.rs:4]
```


**Probability:** [2, p.275, Eq.(1)]

```
General M3PL:  

    P = c + (1 - c) × σ(a·θ + d)

M2PL (c=0):   

    P = σ(a·θ + d) 
      = 1 / (1 + exp(-(a·θ + d)))
```

Persamaan (1) di Mulder & van der Linden (2009, hlm. 275) menuliskan model M3PL persis sebagai
`P_i(θ) ≡ c_i + (1-c_i) / (1 + exp(-(a_i·θ + b_i)))`, di mana `b_i` mereka ≡ `d` di notes ini
(bukan parameter kesulitan `b` gaya unidimensional — dikonfirmasi eksplisit di teks: *"b_i is not
a difficulty parameter in the same sense as a unidimensional IRT model"*). Jadi parameterisasi
`a·θ + d` pada notes ini **cocok persis** dengan literatur.

**Fisher Information Matrix (FIM):** [2, p.276–277, Eq.(4)–(5)]

```
I_i(θ) = w × (a × aᵀ)

Where:
    P* = σ(a·θ + d)              — sigmoid probability (without guessing)
    P' = (1-c) × P* × (1 - P*)   — derivative of P w.r.t. linear predictor
    Q  = 1 - P                   — probability of incorrect
    w  = (P')² / (P × Q)         — information weight
```

Mulder & van der Linden (2009, hlm. 276, Eq. 4) mendefinisikan FIM item secara formal sebagai
turunan kedua log-likelihood:

```
I_i(θ) ≡ -E[ ∂²/∂θ∂θᵀ log f(U_i|θ) ]  =  g(θ; a_i, b_i, c_i) × a_i aᵀ_i
```

dengan (hlm. 277, Eq. 5):

```
g(θ; a, b, c) = Q(θ) × [P(θ) - c]² / [P(θ) × (1-c)²]
```

Mereka juga secara eksplisit mencatat dua sifat struktural matriks ini (hlm. 276, dua bullet
pertama setelah Eq. 4): *"The matrix depends on the ability parameters only through P_i(θ)"* dan
**"The matrix has rank one"** — dua fakta ini penting untuk bagian degenerasi round-1 di bawah.

```
General: 

    P* = σ(a·θ + d)  
    P' = (1-c) × P* × (1-P*)
    w  = (P')² / (P × Q + ε)


M2PL (c=0): 

    P* = P,
    P' = P × (1-P) = P × Q

    → w  = (P×Q)² / (P×Q) = P × Q


M2PL FIM:  

    I_i(θ) = P(1-P) × (a × aᵀ)
```

**Bukti aljabar: `w = (P')²/(P×Q)` ≡ `g(θ;a,b,c)` di literatur**

Notes ini mendefinisikan bobot informasi umum sebagai `w = (P')²/(P×Q)`, sedangkan Mulder & van
der Linden (2009, hlm. 277, Eq. 5) dan Baker (2001, hlm. 111, Eq. 6-5) menuliskannya sebagai
`g = Q(P-c)²/[P(1-c)²]`. Berikut pembuktian bahwa keduanya **identik secara aljabar** (bukan
kebetulan notasi):

```
Diketahui:
    P* = σ(a·θ+d),  Q* = 1-P*
    P  = c + (1-c)P*        →  P* = (P-c)/(1-c)
    Q  = 1-P = (1-c)Q*       →  Q* = Q/(1-c)
    P' = (1-c) P* Q*

Maka:
    (P')² = (1-c)² P*² Q*²

    (P')²/(P×Q) = (1-c)² P*² Q*² / (P×Q)

    substitusi Q*² = Q²/(1-c)²:

    = (1-c)² P*² × [Q²/(1-c)²] / (P×Q)
    = P*² × Q / P
    = [(P-c)/(1-c)]² × Q/P
    = Q × (P-c)² / [P × (1-c)²]          ← identik dengan g(θ;a,b,c) [2, Eq.5] dan
                                            I(θ)=a²·Q/P·[(P-c)/(1-c)]² [1, Eq.6-5]
```

Untuk kasus M2PL (`c=0`): `P*=P`, `Q*=Q`, `w = Q×P²/P = P×Q` — cocok dengan Baker (2001, hlm.
109, Eq. 6-3): `I(θ) = a² P(θ) Q(θ)`, dan dengan generalisasi multidimensional `a² → a×aᵀ` yang
dinyatakan eksplisit di [2, hlm. 278, Eq. 9]. **Kesimpulan: rumus `w` di notes ini terbukti benar
secara teori, bukan hanya sesuai secara kebetulan numerik.**

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

#### Item Selection - D-Optimal

Script `item_selection_d_optimal.rs` mengikuti pola `HIST_THETA` / `INITIAL_THETA` /
`initial_administered()` yang sama dengan `item_selection_a_optimal.rs` dan
`item_selection_kl_information.rs`: `cum_FIM` **tidak** di-hardcode, melainkan dihitung
programatik dari daftar item yang sudah di-administer via `item_fim(HIST_THETA, a, d, c)`
yang dijumlahkan. Ini menghindari rounding-error/ketidaksinkronan antara `cum_FIM` dan
`initial_administered()` yang ada pada versi script sebelumnya (di mana keduanya di-hardcode
terpisah dan bisa saling tidak konsisten).

Ketiga round di bawah (**Round 1**, **Round 2**, **Round 3**) adalah **output asli** dari
`cargo run --example item_selection_d_optimal`, dijalankan tiga kali dengan
`initial_administered()` di-toggle antara `vec![]` (Round 1), `vec!["m2p-v001"]` (Round 2), dan
`vec!["m2p-v001", "m2p-v002"]` (Round 3), `HIST_THETA = INITIAL_THETA = [0.0, 0.0, 0.0]` di
ketiga round (konvensi yang sama dengan `item_selection_a_optimal.rs`: θ tetap di prior sampai
round 4, baru di-re-estimasi). Angka di bawah karena itu dijamin identik dengan hasil script —
bukan hasil hitung manual independen.

**Catatan penting:** script ini **tidak lagi punya tiebreak lokal berbasis `trace(item_FIM)`**.
Versi sebelumnya dari script menghitung `trace_score` sebagai fallback tampilan saat semua
`det_score` seri di 0 — tapi itu **tidak pernah merepresentasikan perilaku API production**
(`McatEngine::select_next_item`, `src/mcat/engine.rs`), yang murni membandingkan skor dengan
`score > best_score` (strict greater-than) tanpa fallback apa pun. Saat semua kandidat seri di
skor yang sama, kandidat **pertama dalam urutan iterasi** (urutan `items` yang dikirim ke
`select_next_item`, yaitu urutan `sample_bank()` di script ini) yang menang — bukan karena
punya "informasi tunggal" terbesar, murni karena urutan array. Script sudah diperbarui agar
`sort_by` hanya mengurutkan berdasar `det_score` (stable sort, jadi item pertama di antara yang
seri tetap di posisi teratas) — ini mereproduksi persis perilaku `select_next_item` tanpa
menambahkan kriteria buatan. Round 1, 2, dan 3 di bawah sudah dijalankan ulang dengan versi
script yang diperbaiki ini.

##### Formulas (M2PL, c=0):

```
General Probability M3PL:  

   P  = c + (1 - c) × P*

   P* = σ(a·θ + d)

   P  = c + (1 - c) × σ(a·θ + d)

Probability M2PL (c=0):   

   P = σ(a·θ + d) 
      = 1 / (1 + exp(-(a·θ + d)))

General information weight:

   P* = σ(a·θ + d)  
   P' = (1-c) × P* × (1-P*)
   w  = (P')² / (P × Q + ε)

information weight M2PL (c=0): 

    P* = P
    P' = P × (1-P) = P × Q

    w  = (P×Q)² / (P×Q) 
       = P × Q
```

```
linear  =  a.θ̂ + d                                   (dot product + intercept)
P       =  σ(linear)  =  1 / (1 + e^(−linear))       (sigmoid)
Q       =  1 − P                                     (complement - probabilitas menjawab salah)
w       =  P × Q                                     (FIM weight)
```

**Konsep Dot Product**
[a1, a2, a3] . [θ̂1, θ̂2, θ̂3] = a1×θ̂2 + a2×θ̂2 + a3×θ̂3


##### Round 1 — 0 Item Administered

```
θ̂            = [0.0, 0.0, 0.0]   (initial estimate — belum ada informasi)
cum_FIM      = 3×3 zero matrix
Administered = []
Candidates   = all 7 items
```

##### Perhitungan Probability dan Weight

At `θ̂ = [0.0, 0.0, 0.0]`

e (Euler's number) = 2.71828  *(catatan koreksi: notes asli menulis "2.728" — ini typo/salah ketik.
Nilai e yang benar adalah 2.71828..., dan seluruh perhitungan `P` di bawah memang sudah
menggunakan nilai yang benar ini (mis. `e^-0.30 = 0.7408` cocok dengan `2.71828^-0.30`, bukan
dengan `2.728^-0.30 = 0.7413`), jadi hasil numerik tidak terpengaruh — hanya label konstanta yang
perlu diperbaiki.)*

**m2p-r001** — `a=[0.5, 0.4, 2.0]`, `d=0.30`

```
linear  =  0.5×0 + 0.4×0 + 2.0×0 + 0.30
        =  0 + 0 + 0 + 0.30
        =  0.30

P  =  1 / (1 + e^(−0.30))
   =  1 / (1 + 0.7408)
   =  1 / 1.7408
   =  0.5744

Q  =  1 − 0.5744  =  0.4256

w  =  0.5744 × 0.4256  =  0.2445
```

---

**m2p-r002** — `a=[0.3, 0.8, 1.9]`, `d=0.60`

```
linear  =  0.3×0 + 0.8×0 + 1.9×0 + 0.60
        =  0.60

P  =  1 / (1 + e^(−0.60))
   =  1 / (1 + 0.5488)
   =  1 / 1.5488
   =  0.6457

Q  =  1 − 0.6457  =  0.3543

w  =  0.6457 × 0.3543  =  0.2287
```

---

**m2p-r003** — `a=[0.4, 0.3, 1.8]`, `d=0.70`

```
linear  =  0.4×0 + 0.3×0 + 1.8×0 + 0.70
        =  0.70

P  =  1 / (1 + e^(−0.70))
   =  1 / (1 + 0.4966)
   =  1 / 1.4966
   =  0.6682

Q  =  1 − 0.6682  =  0.3318

w  =  0.6682 × 0.3318  =  0.2217
```

---

**m2p-v001** — `a=[1.9, 0.2, 0.3]`, `d=0.40`

```
linear  =  1.9×0 + 0.2×0 + 0.3×0 + 0.40
        =  0.40

P  =  1 / (1 + e^(−0.40))
   =  1 / (1 + 0.6703)
   =  1 / 1.6703
   =  0.5987

Q  =  1 − 0.5987  =  0.4013

w  =  0.5987 × 0.4013  =  0.2403
```

---

**m2p-v002** — `a=[1.7, 0.2, 0.2]`, `d=0.10`

```
linear  =  1.7×0 + 0.2×0 + 0.2×0 + 0.10
        =  0.10

P  =  1 / (1 + e^(−0.10))
   =  1 / (1 + 0.9048)
   =  1 / 1.9048
   =  0.5250

Q  =  1 − 0.5250  =  0.4750

w  =  0.5250 × 0.4750  =  0.2494   ←── highest w (P closest to 0.5)
```

---

**m2p-n001** — `a=[0.3, 1.9, 0.4]`, `d=0.80`

```
linear  =  0.3×0 + 1.9×0 + 0.4×0 + 0.80
        =  0.80

P  =  1 / (1 + e^(−0.80))
   =  1 / (1 + 0.4493)
   =  1 / 1.4493
   =  0.6900

Q  =  1 − 0.6900  =  0.3100

w  =  0.6900 × 0.3100  =  0.2139   ←── lowest w (P farthest from 0.5)
```

---

**m2p-n002** — `a=[0.3, 1.8, 0.4]`, `d=0.50`

```
linear  =  0.3×0 + 1.8×0 + 0.4×0 + 0.50
        =  0.50

P  =  1 / (1 + e^(−0.50))
   =  1 / (1 + 0.6065)
   =  1 / 1.6065
   =  0.6225

Q  =  1 − 0.6225  =  0.3775

w  =  0.6225 × 0.3775  =  0.2350
```

---

**Ringkasan**

```
Item      a = [v, n, r]          d     linear   P       Q       w=P×Q
───────────────────────────────────────────────────────────────────────
m2p-v002  [1.7, 0.2, 0.2]      +0.10   0.10    0.5250  0.4750  0.2494 → highest w
m2p-r001  [0.5, 0.4, 2.0]      +0.30   0.30    0.5744  0.4256  0.2445
m2p-v001  [1.9, 0.2, 0.3]      +0.40   0.40    0.5987  0.4013  0.2403
m2p-n002  [0.3, 1.8, 0.4]      +0.50   0.50    0.6225  0.3775  0.2350
m2p-r002  [0.3, 0.8, 1.9]      +0.60   0.60    0.6457  0.3543  0.2287
m2p-r003  [0.4, 0.3, 1.8]      +0.70   0.70    0.6682  0.3318  0.2217
m2p-n001  [0.3, 1.9, 0.4]      +0.80   0.80    0.6900  0.3100  0.2139 → lowest w
```

##### Perhitungan FIM untuk setiap item

**Formula:**

```
item_FIM_i  =  w_i × a_i^2

item_FIM_i  =  w_i × (a_i × a_iᵀ)
```

Rumus `item_FIM_i = w_i × (a_i × a_iᵀ)` sesuai dengan [2, hlm. 278, Eq. 9] Mulder & van der Linden
(2009), yang menuliskan `I_i(θ) = g(θ;a_i,b_i,c_i) × [a_il × a_ip]` (matriks `p×p` dari perkalian
komponen `a`) — identik dengan bentuk di atas.

*Catatan verifikasi: notes versi sebelumnya mengutip "Reckase, M.D. (2009). Multidimensional Item
Response Theory. Springer, hlm. 123, formula 5.17" untuk rumus yang sama. Buku Reckase (2009)
memang membahas topik ini, tetapi karena buku tersebut **tidak tersedia sebagai unduhan terbuka**
(tidak ada preview/PDF gratis yang bisa diverifikasi), klaim halaman 123/formula 5.17 itu **tidak
dapat diverifikasi langsung** pada sesi ini dan sebaiknya tidak dianggap pasti sampai ada akses ke
salinan fisik/institusional. Rumus di atas sudah diverifikasi lewat sumber terbuka [2] sehingga
tidak bergantung pada sitasi Reckase tersebut.*

**Outer product konsep**
```
a = [a1, a2, a3]
b = [b1, b2, b3]


a x b = a x bᵀ

a x bᵀ = [a1, a2, a3] x [[b1],
                         [b2],
                         [b3]]
       
       = [[a1.b1, a1.b2, a1.b3],
          [a2.b1, a2.b2, a2.b3],
          [a3.b1, a3.b2, a3.b3]]
```

**trace konsep**

Trace matriks adalah jumlah seluruh elemen pada diagonal utama dari sebuah matriks persegi.
```
 A = [[a11, a12, a13],
      [a21, a22, a23],
      [a31, a32, a33]]

trace dari A = tr(A) = a11 + a22 + a33
```


**Perhitungan**

**m2p-r001** — `w=0.2445`, `a=[0.5, 0.4, 2.0]`

```
a × aᵀ =  [[0.5×0.5  0.5×0.4  0.5×2.0]     [[0.25  0.20  1.00]
           [0.4×0.5  0.4×0.4  0.4×2.0]   =  [0.20  0.16  0.80]
           [2.0×0.5  2.0×0.4  2.0×2.0]]     [1.00  0.80  4.00]]

item_FIM = 0.2445 × a × aᵀ = [[0.0611  0.0489  0.2445]
                              [0.0489  0.0391  0.1956]
                              [0.2445  0.1956  0.9780]]

trace = 0.0611 + 0.0391 + 0.9780 = 1.078 
```

**m2p-r002** — `w=0.2287`, `a=[0.3, 0.8, 1.9]`

```
a × aᵀ =  [[0.09  0.24  0.57]
           [0.24  0.64  1.52]
           [0.57  1.52  3.61]]

item_FIM = 0.2287 × a × aᵀ = [[0.0206  0.0549  0.1304]
                              [0.0549  0.1464  0.3477]
                              [0.1304  0.3477  0.8257]]

trace = 0.0206 + 0.1464 + 0.8257 = 0.993
```

**m2p-v001** — `w=0.2403`, `a=[1.9, 0.2, 0.3]`

```
a × aᵀ =  [[3.61  0.38  0.57]
           [0.38  0.04  0.06]
           [0.57  0.06  0.09]]

item_FIM = 0.2403 × a × aᵀ = [[0.8675  0.0913  0.1370]
                              [0.0913  0.0096  0.0144]
                              [0.1370  0.0144  0.0216]]

trace = 0.8675 + 0.0096 + 0.0216 = 0.899
```

**m2p-n001** — `w=0.2139`, `a=[0.3, 1.9, 0.4]`

```
a × aᵀ =  [[0.09  0.57  0.12]
           [0.57  3.61  0.76]
           [0.12  0.76  0.16]]

item_FIM = 0.2139 × a × aᵀ = [[0.0192  0.1219  0.0257]
                              [0.1219  0.7722  0.1626]
                              [0.0257  0.1626  0.0342]]

trace = 0.0192 + 0.7722 + 0.0342 = 0.826
```

---

**notes**

Apa yang direpresentasikan setiap entri dalam FIM:
```
item_FIM = w × (a × aᵀ) =

  [[w×a₁²     w×a₁×a₂   w×a₁×a₃]
   [w×a₂×a₁   w×a₂²     w×a₂×a₃]
   [w×a₃×a₁   w×a₃×a₂   w×a₃²   ]]
```

Diagonal FIM[i,i] = w × aᵢ² : seberapa banyak soal ini menginformasikan dimensi i secara mandiri

Off-diagonal FIM[i,j] = w × aᵢ × aⱼ: korelasi antara informasi yang dibawa soal ini tentang dimensi i dan dimensi j secara bersamaan


---
##### Perhitungan determinan tiap item

**determinant score** [2, hlm. 277, Eq.(8); 3, hlm. 16, Definition 1 & Eq.(11)]
```
det_score = det(cum_FIM + item_FIM)
```

Kriteria ini adalah D-optimality klasik dari teori optimal design (`max det(M(ξ))`, St. John &
Draper, 1975, hlm. 16, Eq. 11 [3]), diterapkan ke CAT multidimensional pertama kali oleh Segall
(1996) dan diformalkan sebagai `I_{S_{k-1}}(θ̂) + I_{i_k}(θ̂)` oleh Mulder & van der Linden (2009,
hlm. 277, Eq. 8) [2] — persis bentuk `cum_FIM + item_FIM` di notes ini. Mulder & van der Linden
juga menjelaskan makna praktisnya: memaksimalkan determinan ini **meminimalkan volume ellipsoid
kepercayaan (confidence ellipsoid)** dari estimasi MLE θ̂ setelah item ke-k dijawab [2, hlm. 277].

cum_FIM  : matriks cumulative dari item yang sudah dikerjakan

item_FIM : current item FIM

**determinan konsep**

Untuk matriks berordo 3 x 3, salah satu cara paling umum untuk menghitungnya adalah menggunakan Metode Sarrus.

```
B = [[a, b, c], 
     [d, e, f],
     [g, h, i]]

<!-- sorrus -->
[a, b, c] [a, b, c]
[d, e, f] [d, e, f]
[g, h, i] [g, h, i]

det(B) = (a.e.i + b.f.g + c.d.h) - (g.e.c + h.f.a + i.d.b) 
```

**m2p-r001** 

```
item_FIM = [[0.0611  0.0489  0.2445]
            [0.0489  0.0391  0.1956]
            [0.2445  0.1956  0.9780]]


det_score = det(cum_FIM + item_FIM)

# karena belum ada soal yang ter-administer, maka `cum_FIM = 0`
# jadi :

det_score = det(0 + item_FIM)
          = det(item_FIM)

det(item_FIM) = ((0.0611*0.0391*0.9780) + (0.0489*0.1956*0.2445) + (0.2445*0.0489*0.1956) - (0.2445*0.0391*0.2445) + (0.1956*0.1956*0.0611) + (0.9780*0.0489*0.0489))
              = 0.007013 - 0.007013
              = 0
```

Karena `θ̂ = [0.0, 0.0, 0.0]` maka determinan tiap item FIM itu 0.

**Koreksi/klarifikasi teori:** determinan `item_FIM` = 0 di sini **bukan** karena kebetulan
`θ̂ = [0,0,0]` — determinan itu akan tetap 0 untuk **θ berapa pun**, karena `item_FIM = w×(a×aᵀ)`
adalah outer product satu vektor dengan dirinya sendiri, yang secara aljabar linear selalu punya
rank 1 (dua dari tiga eigenvalue-nya nol untuk ruang berdimensi 3). Ini dinyatakan eksplisit di
Mulder & van der Linden (2009, hlm. 276, bullet kedua setelah Eq. 4): **"The matrix has rank
one."** Nilai `θ` hanya memengaruhi besar `w` (skalarnya), bukan struktur rank matriksnya — jadi
determinan `item_FIM` sendirian akan selalu 0 di ability manapun. Yang membuat `det_score` bisa
> 0 adalah penjumlahan dengan `cum_FIM` dari item-item sebelumnya (lihat [2, hlm. 277, Eq. 6 & 8]).

---


##### Ranking

**⚠️ Koreksi (2026-07-13):** ranking di bawah sebelumnya memakai `trace(item_FIM)` sebagai
tiebreak dan memilih `m2p-r001` sebagai "pemenang" Round 1. Itu **tidak pernah mencerminkan
perilaku API production** — lihat catatan di awal §"Item Selection - D-Optimal" di atas.
`McatEngine::select_next_item` tidak pernah menghitung atau membandingkan trace; ia murni
mengambil kandidat pertama dalam urutan iterasi (urutan `items`/`sample_bank()`) dan hanya
menggantinya jika kandidat berikutnya punya skor **lebih besar secara ketat**. Karena semua
`det_score` seri di 0 pada Round 1, tidak ada kandidat yang pernah "lebih besar secara ketat"
dari yang pertama — jadi pemenangnya adalah item **pertama dalam `sample_bank()`**, yaitu
`m2p-v001`, bukan `m2p-r001`. Tabel di bawah adalah **output asli** dari script setelah
perbaikan ini (tanpa kolom `trace_score`):

```
Rank  ID           Area          det_score
──────────────────────────────────────────────
   1    m2p-v001     verbal         -0.00000000 ← SELECTED
   2    m2p-v002     verbal          0.00000000
   3    m2p-n001     numeric        -0.00000000
   4    m2p-n002     numeric        -0.00000000
   5    m2p-r001     reasoning       0.00000000
   6    m2p-r002     reasoning       0.00000000
   7    m2p-r003     reasoning      -0.00000000
```

Semua `det_score` seri di `0` (tanda `±0.00000000` murni artefak floating-point dari determinan
matriks rank-deficient, bukan nilai berbeda — lihat `snap()` di script yang menormalkannya
sebelum pengurutan). **Tidak ada kriteria yang membedakan item-item ini di Round 1** — pemenang
`m2p-v001` murni karena urutan array, bukan karena keunggulan matematis apa pun.

**Selected: `m2p-v001`** — verbal item, `a=[1.9, 0.2, 0.3]`, menang murni karena urutan iterasi
(item pertama di `sample_bank()`), **bukan** karena informasi/trace tertinggi.

---

##### Round 2 — 1 Item Administered

```
θ̂            = [0.0, 0.0, 0.0]   (masih prior — belum di-re-estimasi setelah 1 item)
Administered = [m2p-v001]         (pemenang Round 1 di atas)
Candidates   = 6 item sisanya
```

`cum_FIM` dihitung otomatis oleh script dari `item_fim(HIST_THETA, a_v001, d_v001, c_v001)` —
persis matriks `item_FIM` milik `m2p-v001` di Round 1 di atas (karena baru 1 item yang
administered, `cum_FIM = item_FIM_v001`):

```
cum_FIM = [[0.8673  0.0913  0.1369]
           [0.0913  0.0096  0.0144]
           [0.1369  0.0144  0.0216]]
```

Rank-1 (satu outer product) — sama seperti alasan `item_FIM` tunggal selalu rank-1 di Round 1.

**Perhitungan per candidate** (θ̂=[0,0,0], sama seperti Round 1 karena θ belum di-re-estimasi;
`linear`/`P`/`Q`/`w`/`item_FIM` untuk tiap item karena itu identik dengan nilai Round 1 di atas
— hanya `updated_FIM` dan `det_score` yang berubah karena sekarang dijumlah dengan `cum_FIM`
non-zero):

```
Item      a = [v, n, r]        d     w         det_score
───────────────────────────────────────────────────────────
m2p-v002  [1.7, 0.2, 0.2]    +0.10  0.249376  -0.00000000  ← SELECTED (urutan array)
m2p-n001  [0.3, 1.9, 0.4]    +0.80  0.213910   0.00000000
m2p-n002  [0.3, 1.8, 0.4]    +0.50  0.235004   0.00000000
m2p-r001  [0.5, 0.4, 2.0]    +0.30  0.244458   0.00000000
m2p-r002  [0.3, 0.8, 1.9]    +0.60  0.228784  -0.00000000
m2p-r003  [0.4, 0.3, 1.8]    +0.70  0.221713   0.00000000
```

(`m2p-v001` tidak muncul — sudah administered, di-skip di Stage 1 exposure gate.)

**updated_FIM = cum_FIM + item_FIM** (contoh, `m2p-v002` — pemenang round ini):

```
item_FIM_v002 = [[0.7207  0.0848  0.0848]
                 [0.0848  0.0100  0.0100]
                 [0.0848  0.0100  0.0100]]

updated_FIM = cum_FIM + item_FIM_v002
            = [[0.8673+0.7207  0.0913+0.0848  0.1369+0.0848]
               [0.0913+0.0848  0.0096+0.0100  0.0144+0.0100]
               [0.1369+0.0848  0.0144+0.0100  0.0216+0.0100]]

            = [[1.5880  0.1761  0.2217]
               [0.1761  0.0196  0.0244]
               [0.2217  0.0244  0.0316]]

det(updated_FIM) = -0.00000000
```

`updated_FIM = cum_FIM(rank-1) + item_FIM(rank-1)` — jumlah dua matriks rank-1 punya rank
**paling banyak 2** (aljabar linear standar: `rank(A+B) ≤ rank(A) + rank(B)`), masih `< 3` untuk
FIM berukuran 3×3 → determinan tetap 0 untuk **semua** kandidat, persis seperti Round 1.

**Ranking Round 2** (output asli script, setelah trace fallback dihapus):

```
Rank  ID           Area          det_score
──────────────────────────────────────────────
   1    m2p-v002     verbal         -0.00000000 ← SELECTED
   2    m2p-n001     numeric         0.00000000
   3    m2p-n002     numeric         0.00000000
   4    m2p-r001     reasoning       0.00000000
   5    m2p-r002     reasoning      -0.00000000
   6    m2p-r003     reasoning       0.00000000
```

**Selected: `m2p-v002`** — verbal item, `a=[1.7, 0.2, 0.2]`, menang karena `det_score` seri di 0
untuk semua kandidat dan `m2p-v002` adalah item **pertama** dalam urutan iterasi
(`sample_bank()`) di antara 6 kandidat yang tersisa (`m2p-v001` sudah administered dan di-skip).
Sama seperti Round 1, ini murni akibat urutan array — bukan karena `m2p-v002` punya informasi
lebih besar dari kandidat lain. `det_score` masih 0 di semua kandidat karena
`rank(cum_FIM + item_FIM) ≤ 2 < 3` — sesuai prediksi teori di §"Catatan: Degenerasi di Round 1
dan Round 2" di bawah (D-optimal baru non-degenerate mulai Round 3, setelah `cum_FIM` sendiri
mencapai rank ≥ 2 dari 2 item administered dengan arah `a` yang independen — dikonfirmasi di
§"Round 3 — 2 Item Administered" berikutnya).

---

##### Round 3 — 2 Item Administered (Non-Degenerate)

```
θ̂            = [0.0, 0.0, 0.0]   (masih prior — belum di-re-estimasi)
Administered = [m2p-v001, m2p-v002]   (pemenang Round 1 & Round 2)
Candidates   = [m2p-n001, m2p-n002, m2p-r001, m2p-r002, m2p-r003]
```

`cum_FIM` dihitung otomatis oleh script dari `item_fim_v001 + item_fim_v002`:

```
cum_FIM = [[1.5880  0.1761  0.2217]
           [0.1761  0.0196  0.0244]
           [0.2217  0.0244  0.0316]]
```

`a_v001=[1.9,0.2,0.3]` dan `a_v002=[1.7,0.2,0.2]` **tidak proporsional** (rasio komponen beda:
`1.9/1.7 ≠ 0.2/0.2`), sehingga jumlah dua outer product rank-1 ini mencapai **rank 2** — inilah
kondisi yang diprediksi di §"Catatan: Degenerasi di Round 1 dan Round 2" agar Round 3 berpotensi
non-degenerate. Diagonal `cum_FIM = [1.5880, 0.0196, 0.0316]` menunjukkan dimensi verbal sudah
sangat kuat (kedua item Round 1–2 sama-sama verbal), sedangkan numeric dan reasoning masih nyaris
tanpa informasi.

**Perhitungan per candidate** (θ̂=[0,0,0], sama seperti Round 1/2 — `linear`/`P`/`Q`/`w`/`item_FIM`
identik dengan nilai di round-round sebelumnya; hanya `updated_FIM` dan `det_score` yang berubah
— **output asli script**, tanpa kolom trace):

```
Item      a = [v, n, r]        d     w        det_score
──────────────────────────────────────────────────────────
m2p-n001  [0.3, 1.9, 0.4]    +0.80  0.213910  0.00084651  ← highest, SELECTED
m2p-n002  [0.3, 1.8, 0.4]    +0.50  0.235004  0.00083829
m2p-r002  [0.3, 0.8, 1.9]    +0.60  0.228784  0.00041501
m2p-r001  [0.5, 0.4, 2.0]    +0.30  0.244458  0.00021800
m2p-r003  [0.4, 0.3, 1.8]    +0.70  0.221713  0.00014093  ← lowest
```

(`m2p-v001`, `m2p-v002` di-skip — sudah administered.)

**updated_FIM = cum_FIM + item_FIM** (pemenang, `m2p-n001`):

```
item_FIM_n001 = [[0.0193  0.1219  0.0257]
                 [0.1219  0.7722  0.1626]
                 [0.0257  0.1626  0.0342]]

updated_FIM = cum_FIM + item_FIM_n001
            = [[1.5880+0.0193  0.1761+0.1219  0.2217+0.0257]
               [0.1761+0.1219  0.0196+0.7722  0.0244+0.1626]
               [0.2217+0.0257  0.0244+0.1626  0.0316+0.0342]]

            = [[1.6073  0.2980  0.2474]
               [0.2980  0.7918  0.1870]
               [0.2474  0.1870  0.0658]]

det(updated_FIM) = 0.00084651
```

`updated_FIM = cum_FIM(rank-2) + item_FIM_n001(rank-1)`. Karena `a_n001=[0.3,1.9,0.4]` tidak
berada di span 2-D yang dibentuk `a_v001` dan `a_v002`, penjumlahan ini mencapai **rank 3 penuh**
→ `det > 0` — pertama kalinya determinan lolos dari degenerasi sejak Round 1.

**Ranking Round 3** (output asli script, setelah trace fallback dihapus):

```
Rank  ID           Area          det_score
──────────────────────────────────────────────
   1    m2p-n001     numeric         0.00084651 ← SELECTED
   2    m2p-n002     numeric         0.00083829
   3    m2p-r002     reasoning       0.00041501
   4    m2p-r001     reasoning       0.00021800
   5    m2p-r003     reasoning       0.00014093
```

**Selected: `m2p-n001`** — numeric item, `a=[0.3, 1.9, 0.4]`, `det_score = 0.00084651`.

**Catatan — D-optimal sekarang membedakan item lewat determinan sungguhan, bukan urutan array:**
di Round 3 semua kandidat punya `det_score > 0` yang **berbeda-beda**, jadi tidak ada lagi
ambiguitas seperti Round 1–2. D-optimal memilih `m2p-n001`/`m2p-n002` (item **numeric**) karena
dimensi numeric adalah dimensi yang **paling lemah** di `cum_FIM` saat ini (`cum_FIM[1][1]=0.0196`
— lihat diagonal `[1.5880, 0.0196, 0.0316]`, jauh lebih kecil dari dimensi verbal yang sudah
"dikuatkan" oleh `v001` dan `v002`, dan sedikit lebih kecil dari reasoning `0.0316`). Menambah
item numeric memperbesar volume ellipsoid kepercayaan (`det`) paling signifikan; `m2p-n001`
menang tipis atas `m2p-n002` karena loading numeric-nya sedikit lebih tinggi (`a₁=1.9` vs `1.8`).
Item reasoning (`r001`, `r002`, `r003`) kalah telak karena reasoning bukan dimensi terlemah
(`0.0316` > `0.0196`). Ini persis perilaku D-optimal yang dijelaskan Mulder & van der Linden
(2009): memaksimalkan **volume**, bukan rata-rata (band. dengan A-optimal yang berorientasi
rata-rata variance — lihat `a_optimal_notes.md`).

---

## Implementasi D-Optimal di API

D-optimal sudah terintegrasi penuh di API production pipeline.

### Layer Implementasi

```
Layer            File                                    Peran
────────────────────────────────────────────────────────────────────────────────
Core math        src/mcat/selection/d_optimal.rs:4      score(updated_fim) = det(cum_FIM + item_FIM)
Dispatcher       src/mcat/selection/mod.rs:41-53        score_item() dispatch ke D-optimal/A-optimal/KL/MI
Engine           src/mcat/engine.rs:19-81               McatEngine::select_next_item() — score semua item, pick argmax
API Handler      src/handlers/session_handler.rs:249    Dipanggil di endpoint next_item
Konfigurasi      TestSettings.selection_method (String) Ditentukan per-test-settings, di-parse saat runtime
```

### Dua Titik Pemanggilan di `session_handler.rs`

```
Line ~249  →  seleksi normal dari pool item yang belum dijawab
Line ~274  →  fallback BankExhaustedAction::Extend — seleksi ulang tanpa filter (allow re-administration)
```

### Cara Aktifkan D-Optimal

Set `selection_method = "d_optimal"` di `TestSettings`.

API mendukung 3 metode:

```
"d_optimal"          → det(cum_FIM + item_FIM)
"a_optimal"          → trace(inv(cum_FIM + item_FIM))
"kl_information"     → KL information
```

### Catatan: Degenerasi di Round 1 dan Round 2

D-optimal **degenerasi** pada round 1 ketika `cum_FIM = 0`, karena:

```
item_FIM = w × (a × aᵀ)  →  matriks rank-1 di ℝ³
                           →  2 eigenvalue = 0
                           →  det = 0 untuk semua item
```

Degenerasi ini **berlanjut ke round 2**: setelah 1 item administered, `cum_FIM` sendiri sudah
rank-1 (lihat §"Round 2 — 1 Item Administered" di atas — output script mengonfirmasi
`det_score = 0` untuk semua 6 kandidat). `updated_FIM = cum_FIM(rank-1) + item_FIM(rank-1)`
punya rank paling banyak 2 (`rank(A+B) ≤ rank(A)+rank(B)`), masih `< 3` → determinan tetap 0.

D-optimal baru meaningful mulai **round 3+** setelah `cum_FIM` terakumulasi rank ≥ 2 (dua arah FIM yang independen).

**Verifikasi teori — klaim ini terbukti benar** menurut Mulder & van der Linden (2009, hlm. 277,
kalimat setelah Eq. 6): *"Although the item information matrix I_i(θ) of each item in S has rank
1, the rank of I_S(θ) is equal to p (unless the items in S have the same proportional relationship
between the discrimination parameters)."* Untuk `p = 3` dimensi (verbal/numeric/reasoning di bank
soal ini), `cum_FIM` dari 1–2 item pertama punya rank ≤ 2 (< p), sehingga `det = 0`. Determinan
baru berpotensi > 0 saat memilih item ke-3 (round 3), begitu total rank mencapai `p = 3` — ini
persis definisi non-singularity `M(ξ*)` yang disyaratkan D-optimality secara umum [3, hlm. 16,
Definition 1]: D-optimal hanya terdefinisi baik ketika matriks informasi kandidat **non-singular**.

**Verifikasi empiris — dikonfirmasi oleh script, bukan cuma teori:** §"Round 3 — 2 Item
Administered (Non-Degenerate)" di atas menunjukkan `det_score > 0` untuk **semua** 5 kandidat
begitu `cum_FIM` (dari `v001` + `v002`, rank-2) dijumlah dengan `item_FIM` kandidat manapun —
persis seperti diprediksi di atas. Determinan terkecil (`m2p-r003`, `0.00014093`) tetap `> 0`,
bukan `0`, karena `a_r003` walau dekat, tidak persis proporsional dengan span `a_v001`/`a_v002`.

Di round 1 dan 2, kondisi degenerate ini **tidak di-handle secara eksplisit di API production —
tidak ada tiebreak/fallback apa pun**, termasuk trace. `McatEngine::select_next_item`
(`src/mcat/engine.rs`) hanya membandingkan skor dengan `score > best_score` (strict greater-than);
saat semua skor seri di 0, kandidat pertama dalam urutan iterasi array yang menang — murni
kebetulan urutan, bukan kriteria yang disengaja. Mulai round 3, begitu `cum_FIM` mencapai rank ≥ 2,
`det_score` API kembali membedakan kandidat secara meaningful tanpa perlu fallback apapun.

---

## Referensi

Semua sitasi `[n]` di notes ini merujuk daftar berikut. Semua sumber bertanda **(unduh bebas)**
sudah diverifikasi bisa diunduh langsung tanpa login/institusi pada saat notes ini ditulis
(2026-07-12), sehingga bisa dipakai untuk membandingkan langsung dengan rumus di atas.

**[1]** Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). ERIC Clearinghouse on
Assessment and Evaluation, University of Maryland, College Park, MD.
— hlm. 109, Eq. (6-3): fungsi informasi item model 2PL, `I(θ) = a²P(θ)Q(θ)`.
— hlm. 111, Eq. (6-5): fungsi informasi item model 3PL, `I(θ) = a²[Q(θ)/P(θ)][(P(θ)-c)²/(1-c)²]`.
**(unduh bebas)**: https://www.ime.unicamp.br/~cnaber/Baker_Book.pdf (mirror lain: https://files.eric.ed.gov/fulltext/ED458219.pdf , https://www.fisica.net/enem/The-Basics-of-Item-Response-Theory.pdf)

**[2]** Mulder, J., & van der Linden, W. J. (2009). Multidimensional Adaptive Testing with Optimal
Design Criteria for Item Selection. *Psychometrika*, 74(2), 273–296.
https://doi.org/10.1007/s11336-008-9097-5
— hlm. 275, Eq. (1): model probabilitas M3PL.
— hlm. 276, Eq. (4) + dua bullet setelahnya ("depends on θ only through P", **"the matrix has rank
one"**): definisi FIM item & sifat rank-1-nya.
— hlm. 277, Eq. (5): faktor skalar `g(θ;a,b,c)`.
— hlm. 277, Eq. (6) + catatan rank `I_S(θ)=p` kecuali diskriminasi proporsional: aditivitas FIM &
syarat rank penuh.
— hlm. 277, Eq. (8) & paragraf setelahnya: kriteria `cum_FIM + item_FIM`, asal-usul dari Segall
(1996), dan penjelasan makna D-optimality (minimisasi volume confidence ellipsoid).
— hlm. 278, Eq. (9): bentuk elemen matriks `g × [a_il·a_ip]`.
**(unduh bebas, 2 sumber independen)**:
https://pmc.ncbi.nlm.nih.gov/articles/PMC2813188/ (PubMed Central, open access) dan
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A3BFF7744EDCE563819C31270D9C7E7D/S0033312300021608a.pdf/multidimensional-adaptive-testing-with-optimal-design-criteria-for-item-selection.pdf
(PDF asli dengan nomor halaman jurnal, dipakai untuk semua nomor halaman di atas)

**[3]** St. John, R. C., & Draper, N. R. (1975). D-Optimality for Regression Designs: A Review.
*Technometrics*, 17(1), 15–23.
— hlm. 15: asal-usul kriteria — Wald (1943) mengusulkan `max det(X'X)`; Kiefer & Wolfowitz (1959)
menamainya "D-optimality" dan memperluasnya ke model regresi umum.
— hlm. 16, Definition 1 & Eq. (11): definisi formal, `ξ* D-optimal ⟺ M(ξ*) nonsingular dan
max_ξ det(M(ξ)) = det(M(ξ*))`.
**(unduh bebas)**: https://www.stat.cmu.edu/technometrics/70-79/VOL-17-01/v1701015.pdf

**[4]** Ince Araci, F. G., & Tan, Ş. (2022). Multidimensional Computerized Adaptive Testing
Simulations in R. *International Journal of Assessment Tools in Education*, 9(1), 118–137.
https://doi.org/10.21449/ijate.909616
— hlm. 121: konfirmasi sekunder — *"D-rule method is based on maximizing the determinant of the
information matrix"* (mengutip Segall 2001; Mulder & van der Linden 2009; dll).
**(unduh bebas)**: https://files.eric.ed.gov/fulltext/EJ1339579.pdf

**[5]** Segall, D. O. (1996). Multidimensional Adaptive Testing. *Psychometrika*, 61(2), 331–354.
Sumber asli/historis dari kriteria seleksi item D-optimal untuk MIRT-CAT (dirujuk lewat [2], hlm.
277). **Tidak berhasil diakses gratis** pada sesi ini — Cambridge Core & SpringerLink meminta
akses institusional/berbayar untuk artikel ini, berbeda dengan [2] yang ternyata bisa diunduh
bebas. Klaim yang berasal dari Segall (1996) di notes ini hanya diverifikasi secara tidak
langsung, lewat kutipan & rumus yang direproduksi di [2].

### Ringkasan temuan verifikasi

- Model probabilitas, FIM, faktor bobot `w`/`g`, dan kriteria `det(cum_FIM+item_FIM)` di notes ini
  **terbukti sesuai literatur** ([1], [2]) — termasuk pembuktian aljabar penuh (lihat bagian
  "Bukti aljabar" di atas), bukan asumsi.
- **1 typo ditemukan**: nilai Euler's number ditulis "2.728", seharusnya "2.71828" (perhitungan
  numerik lain di notes sudah memakai nilai yang benar).
- **1 klaim diklarifikasi**: determinan 0 pada round 1 disebabkan struktur rank-1 `item_FIM`
  (berlaku untuk θ berapa pun), bukan spesifik karena `θ̂=[0,0,0]`.
- **1 sitasi tidak dapat diverifikasi**: "Reckase (2009), hlm. 123, formula 5.17" — buku tidak
  tersedia sebagai unduhan terbuka; rumus yang sama sudah divalidasi ulang lewat [2] sehingga tidak
  bergantung pada sitasi ini.
- **Koreksi besar (2026-07-13): trace fallback dihapus dari script & notes.** Versi sebelumnya
  dari `item_selection_d_optimal.rs` menghitung `trace(item_FIM)` sebagai tiebreak lokal saat
  semua `det_score` seri di 0, dan Round 1–3 di notes ini disusun berdasarkan output itu (pemenang
  Round 1 = `m2p-r001` karena trace tertinggi). Setelah diverifikasi terhadap kode production
  (`src/mcat/engine.rs::select_next_item`), ternyata **API tidak pernah menghitung atau
  membandingkan trace** — ia murni memakai `score > best_score` (strict greater-than), sehingga
  saat seri, kandidat **pertama dalam urutan iterasi array** yang menang. Trace fallback di script
  contoh karena itu **tidak pernah merepresentasikan perilaku API sungguhan** dan berpotensi
  menyesatkan. Script dan seluruh Round 1–3 di notes ini sudah dijalankan ulang tanpa trace
  fallback (`cargo run --example item_selection_d_optimal`, tiga kali dengan
  `initial_administered()` di-toggle) — pemenang berubah: Round 1 → `m2p-v001` (bukan `m2p-r001`),
  Round 2 → `m2p-v002` (bukan `m2p-r002`), Round 3 → `m2p-n001` (bukan `m2p-v001`). Round 3 tetap
  non-degenerate dengan kesimpulan teori yang sama (D-optimal memaksimalkan volume, menargetkan
  dimensi terlemah), hanya item konkretnya yang berbeda karena rantai keputusan Round 1–2 berbeda.
