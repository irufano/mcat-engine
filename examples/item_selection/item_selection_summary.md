---
title: "Item Selection Criteria - MCAT (A-Optimal, D-Optimal, KL-Information)"
date: "2026-07-13"
description: "Dokumen ini merangkum tiga metode item selection multidimensional CAT (A-Optimal, D-Optimal, KL-Information)"
author: "irufano"
tags:
  - AI
  - CAT
  - MCAT
  - Adaptive Test
image: "https://www.assessmentworkshop.com/wp-content/uploads/2022/04/CAT-Infographic.png"
---

Dokumen ini merangkum tiga metode item selection multidimensional CAT (A-Optimal, D-Optimal,
KL-Information). Fokus dokumen: teori tiap metode, contoh perhitungan manual per item,
serta kelebihan/kekurangan masing-masing. 

---

## Model & Notasi Dasar (dipakai oleh ketiga metode)

Model respons item yang dipakai adalah M3PL/M2PL multidimensional-compensatory [1][2]:

$$
\text{M3PL:}\quad P = c + (1-c)\,\sigma(\mathbf{a}\cdot\boldsymbol{\theta} + d)
$$

$$
\text{M2PL } (c=0):\quad P = \sigma(\mathbf{a}\cdot\boldsymbol{\theta} + d) = \frac{1}{1+\exp\!\big(-(\mathbf{a}\cdot\boldsymbol{\theta}+d)\big)}
$$

**Keterangan variabel (model probabilitas):**

| Simbol | Arti |
|---|---|
| $P$ | Peluang examinee menjawab item benar (*probability of correct response*) |
| $c$ | Parameter *guessing* — peluang menjawab benar secara asal-asalan; $c=0$ untuk M2PL |
| $\sigma(\cdot)$ | Fungsi sigmoid/logistik, $\sigma(x) = 1/(1+e^{-x})$ |
| $\mathbf{a}$ | Vektor parameter diskriminasi item — satu nilai per dimensi kemampuan (mis. $[a_{verbal}, a_{numeric}, a_{reasoning}]$) |
| $\boldsymbol{\theta}$ | Vektor kemampuan (*ability*) examinee — satu nilai per dimensi, ditaksir selama tes berjalan |
| $d$ | Parameter intercept item — **bukan** parameter kesulitan unidimensional biasa (lihat catatan literatur [1]) |
| $\mathbf{a}\cdot\boldsymbol{\theta}$ | Dot product vektor $\mathbf{a}$ dan $\boldsymbol\theta$ — disebut *linear predictor* saat ditambah $d$ (sering ditulis $z=\mathbf{a}\cdot\theta+d$ di [#1.2](#12-perhitungan-manual-per-item) dst.) |

**Fisher Information Matrix (FIM) item** [1, Eq.4–5][2, Eq.6-5]:

$$
I_i(\boldsymbol{\theta}) = w \left(\mathbf{a} \cdot \mathbf{a}^\top\right)
$$

$$
P^* = \sigma(\mathbf{a}\cdot\boldsymbol{\theta}+d)
$$

$$
P' = (1-c)\,P^*(1-P^*)
$$

$$
Q = 1-P
$$

$$
w = \frac{(P')^2}{P\,Q}
$$

$$
\text{M2PL } (c=0):\quad w = P\,Q \;\;\Rightarrow\;\; I_i(\boldsymbol{\theta}) = P(1-P)\left(\mathbf{a} \cdot \mathbf{a}^\top\right)
$$

$I_i(\boldsymbol{\theta})$ adalah **outer product** $\mathbf{a} \cdot \mathbf{a}^\top$ dikali skalar $w$ —
secara aljabar linear matriks ini selalu **rank 1** [1, p.276: "the matrix has rank one"], apa pun
nilai $\theta$. Fakta rank-1 ini adalah akar dari kasus degenerate yang dialami D-optimal dan
A-optimal di round-round awal (lihat [#4](#4-ringkasan-perbandingan)).

**Keterangan variabel (FIM):**

| Simbol | Arti |
|---|---|
| $I_i(\boldsymbol\theta)$ | Fisher Information Matrix (FIM) item $i$ pada kemampuan $\theta$ — matriks $k\times k$ ($k$ = jumlah dimensi, $k=3$ di notes ini) |
| $w$ | Skalar bobot informasi item (*information weight*) — mengukur seberapa informatif item pada $\theta$ saat ini |
| $\mathbf{a} \cdot \mathbf{a}^\top$ | Outer product vektor $\mathbf{a}$ dengan dirinya sendiri → matriks $k\times k$, elemen $(l,p) = a_l \cdot a_p$ |
| $P^*$ | Peluang benar **tanpa** komponen guessing — bagian sigmoid murni dari model M3PL, $P^*=\sigma(\mathbf{a}\cdot\theta+d)$ |
| $P'$ | Turunan $P$ terhadap linear predictor $\mathbf{a}\cdot\theta+d$ — mengukur kecuraman kurva ICC di titik $\theta$ |
| $Q$ | Peluang menjawab **salah**, $Q=1-P$ (komplemen dari $P$) |

## Item Bank Snapshot

**Item bank snapshot** (dipakai konsisten di seluruh notes & contoh di bawah):

| ID | Content area | $\mathbf{a}$ = [verbal, numeric, reasoning] | $d$ | $c$ |
|---|---|---|---|---|
| m2p-v001 | verbal | [1.9, 0.2, 0.3] | +0.40 | 0.0 |
| m2p-v002 | verbal | [1.7, 0.2, 0.2] | +0.10 | 0.0 |
| m2p-n001 | numeric | [0.3, 1.9, 0.4] | +0.80 | 0.0 |
| m2p-n002 | numeric | [0.3, 1.8, 0.4] | +0.50 | 0.0 |
| m2p-r001 | reasoning | [0.5, 0.4, 2.0] | +0.30 | 0.0 |
| m2p-r002 | reasoning | [0.3, 0.8, 1.9] | +0.60 | 0.0 |
| m2p-r003 | reasoning | [0.4, 0.3, 1.8] | +0.70 | 0.0 |

---

## 1. D-Optimal

### 1.1 Teori

D-optimality berasal dari teori optimal design klasik ($\max \det(X^\top X)$, Wald 1943; dinamai
"D-optimal" oleh Kiefer & Wolfowitz 1959) [4, p.15–16], diterapkan pertama kali ke CAT
multidimensional oleh Segall (1996) [6] dan diformalkan oleh Mulder & van der Linden (2009)
[1, Eq.8, p.277]:

$$
\arg\max_i \; \det\!\left(\mathbf{FIM}_{cum} + \mathbf{FIM}_i\right)
$$

Memaksimalkan determinan FIM gabungan **meminimalkan volume ellipsoid kepercayaan** (confidence
ellipsoid) dari estimasi MLE $\hat{\boldsymbol{\theta}}$ setelah item ke-$k$ dijawab [1, p.277].
$\det(\cdot)$ di sini berperan sebagai ukuran "volume informasi" multidimensional — makin besar
determinan, makin sempit/presisi ellipsoid kepercayaan $\hat{\boldsymbol{\theta}}$ di semua dimensi
sekaligus.

**Keterangan variabel:**

| Simbol | Arti |
|---|---|
| $i$ | Indeks kandidat item yang sedang dievaluasi (menjalankan seluruh item di pool eligible) |
| $\arg\max_i$ | Pilih item $i$ yang **memaksimalkan** ekspresi di sebelah kanan |
| $\det(\cdot)$ | Determinan matriks — ukuran "volume" informasi multidimensional |
| $\mathbf{FIM}_{cum}$ | FIM **kumulatif** dari seluruh item yang sudah di-administer sebelumnya ($=\sum_{j\in S} I_j(\theta)$, $S$ = himpunan item administered) |
| $\mathbf{FIM}_i$ | FIM item kandidat $i$ (identik dengan $I_i(\theta)$ di [#0](#model--notasi-dasar-dipakai-oleh-ketiga-metode)) |
| $\hat{\boldsymbol\theta}$ | Estimasi kemampuan examinee **saat ini** (hasil MLE/MAP/EAP dari respons sebelumnya, bukan $\theta$ sebenarnya yang tidak diketahui) |
| $k$ (pada "item ke-$k$") | Indeks urutan item dalam tes (item ke berapa yang sedang dipilih) — **catatan:** simbol $k$ di sini berbeda dari $k$ = jumlah dimensi yang dipakai di bagian lain notes ini |

---

### 1.2 Perhitungan Manual Per Item

#### 1.2.1 Case Degenerate

**Round 1, $\hat{\boldsymbol{\theta}} = [0,0,0]$, 0 item administered**

Langkah per item: hitung $z = \mathbf{a}\cdot\hat{\boldsymbol{\theta}} + d$ → $P=\sigma(z)$ →
$Q=1-P$ → $w=PQ$ → $\mathbf{FIM}_i = w\,(\mathbf{a} \cdot \mathbf{a}^\top)$ → $\det\_score =
\det(\mathbf{FIM}_{cum}+\mathbf{FIM}_i)$.

Contoh lengkap **m2p-v001** ($\mathbf{a}=[1.9, 0.2, 0.3]$, $d=0.40$):

$$z = 1.9(0) + 0.2(0) + 0.3(0) + 0.40 = 0.40$$

$$P = \sigma(0.40) = 0.598688, \qquad Q = 0.401312, \qquad w = PQ = 0.240261$$

$$\mathbf{a} \cdot \mathbf{a}^\top =\begin{bmatrix} 3.61 & 0.38 & 0.57 \\ 0.38 & 0.04 & 0.06 \\ 0.57 & 0.06 & 0.09 \end{bmatrix}$$

$$\mathbf{FIM}_i = 0.240261 \times \mathbf{a} \cdot \mathbf{a}^\top =\begin{bmatrix} 0.8673 & 0.0913 & 0.1369 \\ 0.0913 & 0.0096 & 0.0144 \\ 0.1369 & 0.0144 & 0.0216 \end{bmatrix}$$

$$\det\_score = \det(\mathbf{FIM}_{cum} + \mathbf{FIM}_i) = \det(0 + \mathbf{FIM}_i) = 0 \quad(\text{rank-1, selalu } 0 \text{ di round } 1)$$

**Probability & weight seluruh item** (langkah yang sama, $z=\mathbf{a}\cdot[0,0,0]+d=d$, urutan
sesuai [Item Bank Snapshot](#item-bank-snapshot)):

| Item | $\mathbf{a}$ | $d$ | $z$ | $P$ | $Q$ | $w=PQ$ |
|---|---|---|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | 0.40 | 0.598688 | 0.401312 | 0.240261 |
| m2p-v002 | [1.7,0.2,0.2] | 0.10 | 0.10 | 0.524979 | 0.475021 | 0.249376 |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | 0.80 | 0.689974 | 0.310026 | 0.213910 |
| m2p-n002 | [0.3,1.8,0.4] | 0.50 | 0.50 | 0.622459 | 0.377541 | 0.235004 |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | 0.30 | 0.574443 | 0.425557 | 0.244458 |
| m2p-r002 | [0.3,0.8,1.9] | 0.60 | 0.60 | 0.645656 | 0.354344 | 0.228784 |
| m2p-r003 | [0.4,0.3,1.8] | 0.70 | 0.70 | 0.668188 | 0.331812 | 0.221713 |

Determinan $\mathbf{FIM}_i$ sendirian **selalu 0 untuk $\theta$ berapa pun** — konsekuensi langsung
dari sifat rank-1 outer product $\mathbf{a} \cdot \mathbf{a}^\top$ [1, p.276]. Round 1 D-optimal karena itu
**degenerate**: semua item mendapat $\det\_score \approx 0$ (tanda $\pm$ murni artefak
floating-point dari determinan matriks rank-deficient). **Tidak ada mekanisme yang membedakan
item-item ini**:

| Rank | Item | Area | $\det\_score$ | Note |
|---|---|---|---|---|
| 1 | m2p-v001 | verbal | $-0.00000000$ | SELECTED |
| 2 | m2p-v002 | verbal | $0.00000000$ | |
| 3 | m2p-n001 | numeric | $-0.00000000$ | |
| 4 | m2p-n002 | numeric | $-0.00000000$ | |
| 5 | m2p-r001 | reasoning | $0.00000000$ | |
| 6 | m2p-r002 | reasoning | $0.00000000$ | |
| 7 | m2p-r003 | reasoning | $-0.00000000$ | |

**Selected: `m2p-v001`** — menang **murni karena urutan iterasi array** (item pertama di
[Item Bank Snapshot](#item-bank-snapshot)), bukan karena keunggulan matematis. Ini konsisten dengan perilaku
`select_next_item` yang sesungguhnya: karena `score > best_score` gagal untuk skor yang sama,
kandidat pertama yang ditemukan tidak pernah tergeser oleh kandidat lain yang skornya sama-sama 0.

D-optimal baru "meaningful" ($\det > 0$) mulai **round 3+**, setelah $\mathbf{FIM}_{cum}$
terakumulasi rank $\geq 2$ dari item-item sebelumnya [1, p.277: rank $I_S(\theta) = p$, kecuali
item-item punya diskriminasi proporsional identik].

---

**Round 2** ($\hat{\boldsymbol{\theta}}=[0,0,0]$, administered $=$ [`m2p-v001`]) masih degenerate
dengan alasan sama: $\mathbf{FIM}_{cum}$ (rank-1, dari `v001`) $+$ $\mathbf{FIM}_i$ (rank-1) hanya
mencapai rank $\leq 2 < 3$, sehingga $\det\_score \approx 0$ untuk **semua 6 kandidat tersisa**.
Winner-nya: **`m2p-v002`** — lagi-lagi item pertama dalam urutan iterasi di
antara kandidat yang tersisa, bukan karena kriteria apa pun.

---

#### 1.2.2 Case Non-Degenerate

**Round 3, $\hat{\boldsymbol{\theta}} = [0,0,0]$, 2 item administered ($\texttt{m2p-v001}$,
$\texttt{m2p-v002}$ — pemenang Round 1 & 2 di atas)**

$\mathbf{FIM}_{cum}$ adalah **hasil penjumlahan elemen-per-elemen** dari FIM dua item yang sudah
di-administer, $\mathbf{FIM}_{cum} = \mathbf{FIM}_{v001} + \mathbf{FIM}_{v002}$ — bukan dihitung
ulang dari awal, melainkan diakumulasi dari $\mathbf{FIM}_i$ tiap item yang sudah dijawab
sebelumnya (lihat definisi $\mathbf{FIM}_{cum}=\sum_{j\in S} I_j(\theta)$ di [#1.1](#11-teori)).

**$\mathbf{FIM}_{v001}$** (sudah dihitung lengkap di [#1.2.1](#121-case-degenerate) sebagai "Contoh lengkap m2p-v001",
$\theta=[0,0,0]$, $w=0.240261$):

$$\mathbf{FIM}_{v001} = \begin{bmatrix} 0.8673 & 0.0913 & 0.1369 \\ 0.0913 & 0.0096 & 0.0144 \\ 0.1369 & 0.0144 & 0.0216 \end{bmatrix}$$

**$\mathbf{FIM}_{v002}$** ($\mathbf{a}=[1.7,0.2,0.2]$, $d=0.10$, $w=0.249376$ dari tabel
probability & weight di [#1.2.1](#121-case-degenerate)):

$$\mathbf{a} \cdot \mathbf{a}^\top = \begin{bmatrix} 2.89 & 0.34 & 0.34 \\ 0.34 & 0.04 & 0.04 \\ 0.34 & 0.04 & 0.04 \end{bmatrix} \quad\Rightarrow\quad \mathbf{FIM}_{v002} = 0.249376\times\mathbf{a} \cdot \mathbf{a}^\top = \begin{bmatrix} 0.7207 & 0.0848 & 0.0848 \\ 0.0848 & 0.0100 & 0.0100 \\ 0.0848 & 0.0100 & 0.0100 \end{bmatrix}$$

Dijumlahkan elemen-per-elemen:

$$\mathbf{FIM}_{cum} = \mathbf{FIM}_{v001} + \mathbf{FIM}_{v002} = \begin{bmatrix} 0.8673+0.7207 & 0.0913+0.0848 & 0.1369+0.0848 \\ 0.0913+0.0848 & 0.0096+0.0100 & 0.0144+0.0100 \\ 0.1369+0.0848 & 0.0144+0.0100 & 0.0216+0.0100 \end{bmatrix}$$

$$= \begin{bmatrix} 1.5880 & 0.1761 & 0.2217 \\ 0.1761 & 0.0196 & 0.0244 \\ 0.2217 & 0.0244 & 0.0316 \end{bmatrix}$$

Karena $\mathbf{a}_{v001}=[1.9,0.2,0.3]$ dan $\mathbf{a}_{v002}=[1.7,0.2,0.2]$ **tidak proporsional**
(rasio komponen beda: $1.9/1.7 \neq 0.2/0.2$), jumlah dua outer product rank-1 ini mencapai
**rank 2 penuh** — bukan rank ≤ 1, karena $\mathbf{a}_{v002}$ tidak berada di garis (span 1-D) yang
dibentuk $\mathbf{a}_{v001}$.

Diagonal $[1.5880, 0.0196, 0.0316]$: dimensi verbal sudah sangat kuat (kedua item Round 1–2
sama-sama verbal), numeric dan reasoning nyaris tanpa informasi — numeric sedikit lebih lemah.

---

Contoh lengkap kandidat **m2p-n001** ($\mathbf{a}=[0.3, 1.9, 0.4]$, $d=0.80$, pemenang round ini):

**Langkah 1 — probability & weight** (sama seperti di [#1.2.1](#121-case-degenerate), karena $\theta=[0,0,0]$ belum berubah):

$$z = \mathbf{a}\cdot\theta + d = 0.3(0)+1.9(0)+0.4(0)+0.80 = 0.80$$

$$P = \sigma(0.80) = 0.689974, \qquad Q = 1-P = 0.310026, \qquad w = P\,Q = 0.689974\times0.310026 = 0.213910$$

**Langkah 2 — outer product $\mathbf{a} \cdot \mathbf{a}^\top$** (perkalian tiap pasangan komponen $\mathbf{a}$):

$$\mathbf{a} \cdot \mathbf{a}^\top = \begin{bmatrix} 0.3\times0.3 & 0.3\times1.9 & 0.3\times0.4 \\ 1.9\times0.3 & 1.9\times1.9 & 1.9\times0.4 \\ 0.4\times0.3 & 0.4\times1.9 & 0.4\times0.4 \end{bmatrix} = \begin{bmatrix} 0.09 & 0.57 & 0.12 \\ 0.57 & 3.61 & 0.76 \\ 0.12 & 0.76 & 0.16 \end{bmatrix}$$

**Langkah 3 — $\mathbf{FIM}_i = w\times\mathbf{a} \cdot \mathbf{a}^\top$** (tiap elemen matriks dikali skalar $w=0.213910$):

$$\mathbf{FIM}_i = 0.213910\times\begin{bmatrix} 0.09 & 0.57 & 0.12 \\ 0.57 & 3.61 & 0.76 \\ 0.12 & 0.76 & 0.16 \end{bmatrix} = \begin{bmatrix} 0.213910{\times}0.09 & 0.213910{\times}0.57 & 0.213910{\times}0.12 \\ 0.213910{\times}0.57 & 0.213910{\times}3.61 & 0.213910{\times}0.76 \\ 0.213910{\times}0.12 & 0.213910{\times}0.76 & 0.213910{\times}0.16 \end{bmatrix}$$

$$\mathbf{FIM}_i = \begin{bmatrix} 0.0193 & 0.1219 & 0.0257 \\ 0.1219 & 0.7722 & 0.1626 \\ 0.0257 & 0.1626 & 0.0342 \end{bmatrix}$$

**Langkah 4 — $\mathbf{FIM}_{updated} = \mathbf{FIM}_{cum} + \mathbf{FIM}_i$** (penjumlahan elemen-per-elemen, $\mathbf{FIM}_{cum}$ dari [#1.2.2](#122-case-non-degenerate) di atas):

$$\mathbf{FIM}_{updated} = \begin{bmatrix} 1.5880{+}0.0193 & 0.1761{+}0.1219 & 0.2217{+}0.0257 \\ 0.1761{+}0.1219 & 0.0196{+}0.7722 & 0.0244{+}0.1626 \\ 0.2217{+}0.0257 & 0.0244{+}0.1626 & 0.0316{+}0.0342 \end{bmatrix}$$

$$\mathbf{FIM}_{updated} = \begin{bmatrix} 1.6073 & 0.2980 & 0.2474 \\ 0.2980 & 0.7918 & 0.1870 \\ 0.2474 & 0.1870 & 0.0658 \end{bmatrix}$$

Determinan via metode Sarrus, $\det(B) = (aei+bfg+cdh)-(gec+hfa+idb)$:

$$\det(\mathbf{FIM}_{updated}) = (1.6073{\times}0.7918{\times}0.0658 + 0.2980{\times}0.1870{\times}0.2474 + 0.2474{\times}0.2980{\times}0.1870)$$
$$\qquad - (0.2474{\times}0.7918{\times}0.2474 + 0.1870{\times}0.1870{\times}1.6073 + 0.0658{\times}0.2980{\times}0.2980) = 0.00084651$$

`updated_FIM = cum_FIM(rank-2) + item_FIM_n001(rank-1)`. Karena $\mathbf{a}_{n001}$ tidak berada
di span 2-D yang dibentuk $\mathbf{a}_{v001}$ dan $\mathbf{a}_{v002}$, penjumlahan mencapai
**rank 3 penuh** → $\det > 0$ — pertama kalinya determinan lolos dari degenerasi sejak Round 1.

Dengan cara yang sama untuk 4 kandidat lain:

| Item | $\mathbf{a}$ | $w$ | $\det(\mathbf{FIM}_{updated})$ |
|---|---|---|---|
| m2p-n001 | [0.3,1.9,0.4] | 0.213910 | 0.00084651 |
| m2p-n002 | [0.3,1.8,0.4] | 0.235004 | 0.00083829 |
| m2p-r002 | [0.3,0.8,1.9] | 0.228784 | 0.00041501 |
| m2p-r001 | [0.5,0.4,2.0] | 0.244458 | 0.00021800 |
| m2p-r003 | [0.4,0.3,1.8] | 0.221713 | 0.00014093 |

**Ranking Round 3:**

| Rank | Item | Area | $\det\_score$ | Note |
|---|---|---|---|---|
| 1 | m2p-n001 | numeric | 0.00084651 | SELECTED |
| 2 | m2p-n002 | numeric | 0.00083829 | |
| 3 | m2p-r002 | reasoning | 0.00041501 | |
| 4 | m2p-r001 | reasoning | 0.00021800 | |
| 5 | m2p-r003 | reasoning | 0.00014093 | |

**Selected: `m2p-n001`** — di Round 3 semua kandidat punya $\det\_score>0$ yang **berbeda-beda**,
jadi tidak ada lagi ambiguitas seperti Round 1–2. D-optimal memilih item **numeric** karena
dimensi numeric adalah dimensi terlemah di $\mathbf{FIM}_{cum}$ saat ini
($\mathbf{FIM}_{cum}[1][1]=0.0196$, lebih kecil dari reasoning $0.0316$ dan jauh lebih kecil dari
verbal $1.5880$ yang sudah dikuatkan `v001`+`v002`). `m2p-n001` menang tipis atas `m2p-n002`
karena loading numeric-nya sedikit lebih tinggi ($a_1=1.9$ vs $1.8$). Item reasoning kalah telak
karena reasoning bukan dimensi terlemah. Ini persis perilaku D-optimal: memaksimalkan **volume**,
menargetkan arah yang paling menambah determinan — bukan rata-rata varians (band. A-optimal [#2](#2-a-optimal)).

---

### 1.3 Kelebihan & Kekurangan

**Kelebihan:**
- Landasan teori sangat matang — asal-usulnya dari teori optimal design regresi klasik yang sudah
  dipakai puluhan tahun [4], dan diadaptasi formal ke MIRT-CAT [1][6].
- Interpretasi geometris jelas: memaksimalkan volume informasi (meminimalkan volume ellipsoid
  kepercayaan) — menyeimbangkan presisi di **semua dimensi secara multiplikatif** sekaligus.
- Cocok saat tujuan tes adalah estimasi $\theta$ multidimensional yang presisi merata di semua
  dimensi tanpa bias ke satu dimensi tertentu.

**Kekurangan:**
- **Degenerate di round-round awal**: $\mathbf{FIM}_i$ per item selalu rank-1, sehingga $\det=0$
  sampai $\mathbf{FIM}_{cum}$ mencapai rank $\geq 2$ (butuh minimal 2 item dari arah berbeda). Tidak
  ada. fallback ketika tiebreak pada round-round ini — pemilihan murni bergantung pada
  urutan iterasi array (lihat [#1.2.1](#121-case-degenerate)), bukan kriteria matematis.
- Skalanya bersifat multiplikatif (produk determinan) — kurang agresif menargetkan dimensi yang
  jauh lebih lemah dibanding A-optimal (lihat [#2.3](#23-kelebihan--kekurangan) untuk kontras).
- Bergantung pada $\mathbf{FIM}_{cum}$ (riwayat item sebelumnya), bukan murni informasi item saat
  ini — beda filosofi dengan KL-information ([#3](#3-kl-information-kullback-leibler-information)) yang selalu lokal terhadap $\hat\theta$ terkini.

---

## 2. A-Optimal

### 2.1 Teori

A-optimality [1, Eq.16, §4.1.2, p.±282–284] didefinisikan sebagai kriteria yang **"minimize the sum
of the (asymptotic) sampling variances of the MLEs of the abilities"**, ekuivalen dengan
meminimalkan trace dari invers FIM:

$$
\arg\min_i \; \operatorname{tr}\!\left(\left(\mathbf{FIM}_{cum} + \mathbf{FIM}_i\right)^{-1}\right)
$$

Skor dinegasikan agar higher = better:

$$
a\_score_i = -\operatorname{tr}\!\left(\left(\mathbf{FIM}_{cum} + \mathbf{FIM}_i\right)^{-1}\right)
$$

$\operatorname{tr}(\mathbf{FIM}^{-1})$ adalah jumlah asymptotic variance MLE
$\hat{\boldsymbol{\theta}}$ di semua dimensi (Cramér–Rao lower bound:
$\mathbf{FIM}^{-1} \approx \operatorname{cov}(\hat{\boldsymbol{\theta}})$). Berbeda dari D-optimal
yang memaksimalkan **volume** (produk/determinan), A-optimal meminimalkan **rata-rata** varians
(jumlah/trace) — lebih sensitif terhadap dimensi yang paling lemah secara individual.

**Keterangan variabel:**

| Simbol | Arti |
|---|---|
| $i$ | Indeks kandidat item yang sedang dievaluasi |
| $\arg\min_i$ | Pilih item $i$ yang **meminimalkan** ekspresi di sebelah kanan |
| $\operatorname{tr}(\cdot)$ | Trace matriks — jumlah elemen pada diagonal utama ($\operatorname{tr}(M)=\sum_i M_{ii}$) |
| $(\cdot)^{-1}$ | Invers matriks |
| $\mathbf{FIM}_{cum}$, $\mathbf{FIM}_i$ | Sama seperti di [#1.1](#11-teori) D-Optimal — FIM kumulatif dan FIM item kandidat |
| $a\_score_i$ | Skor akhir A-optimal untuk item $i$, **dinegasikan** dari $\operatorname{tr}((\cdot)^{-1})$ agar "skor lebih tinggi = lebih baik" (konsisten dengan D-optimal/KL yang juga $\arg\max$) |
| $\operatorname{cov}(\hat{\boldsymbol\theta})$ | Matriks kovarians estimasi $\hat\theta$ — makin kecil diagonalnya, makin presisi estimasi kemampuan |

---

### 2.2 Perhitungan Manual Per Item

#### 2.2.1 Case Degenerate

**Round 1, $\hat{\boldsymbol{\theta}} = [0,0,0]$, 0 item administered**

Karena belum ada item administered, $\mathbf{FIM}_{cum} = \mathbf{0}$, sehingga
$\mathbf{FIM}_{updated} = \mathbf{FIM}_i$ — matriks rank-1 (lihat [#0](#model--notasi-dasar-dipakai-oleh-ketiga-metode)). Contoh
**m2p-r001** ($\mathbf{a}=[0.5,0.4,2.0]$, $d=0.30$; $w=0.244458$ — nilai yang sama seperti entri
`m2p-r001` di tabel probability & weight [#1.2.1](#121-case-degenerate), karena keduanya dihitung pada $\theta=[0,0,0]$
yang sama):

$$\mathbf{FIM}_{updated} = \mathbf{FIM}_i = \begin{bmatrix} 0.0611 & 0.0489 & 0.2445 \\ 0.0489 & 0.0391 & 0.1956 \\ 0.2445 & 0.1956 & 0.9778 \end{bmatrix}$$

Cek rank via Row Echelon Form (Gaussian elimination):

$$R_2 \leftarrow R_2 - \frac{0.0489}{0.0611} R_1 = R_2 - 0.8000\,R_1 \;\Rightarrow\; [\,0,\;\approx 0,\;\approx 0\,]$$

$$R_3 \leftarrow R_3 - \frac{0.2445}{0.0611} R_1 = R_3 - 4.0000\,R_1 \;\Rightarrow\; [\,0,\;\approx 0,\;\approx 0\,]$$

$$\text{REF} = \begin{bmatrix} 0.0611 & 0.0489 & 0.2445 \\ 0 & 0 & 0 \\ 0 & 0 & 0 \end{bmatrix} \quad\Rightarrow\quad \operatorname{rank} = 1 < 3$$

(Faktor eliminasi $R_2, R_3$ persis $a_2/a_1=0.4/0.5=0.8$ dan $a_3/a_1=2.0/0.5=4.0$ — konsekuensi
aljabar langsung dari struktur outer product $\mathbf{a} \cdot \mathbf{a}^\top$, berlaku untuk $w$ berapa
pun, bukan kebetulan numerik.)

Rank $<3$ (tidak full rank) $\Rightarrow$ $\mathbf{FIM}_{updated}$ **singular**
($\det = 0$) $\Rightarrow$ invers **tidak terdefinisi** $\Rightarrow$
$a\_score = -\operatorname{tr}(\mathbf{FIM}_{updated}^{-1}) = -\infty$. Ini berlaku untuk **semua**
item di round 1 (semua $\mathbf{FIM}_i$ adalah outer product rank-1):

| Rank | Item | Area | $a\_score$ | Note |
|---|---|---|---|---|
| 1–7 | (semua item) | — | $-\infty$ | tiebreak = urutan array (tidak ada mekanisme eksplisit) |

---

**Round 2** (1 item administered): $\mathbf{FIM}_{cum}$ = 1 matriks rank-1. Untuk kandidat mana pun,
$\mathbf{FIM}_{updated} = \text{rank-1} + \text{rank-1}$, yang secara aljabar menghasilkan
$\operatorname{rank} \leq 2 < 3$ — **masih singular**, $a\_score = -\infty$ untuk semua kandidat.
A-optimal baru berpotensi non-degenerate mulai **round $k+1$** (round 4 untuk $k=3$ dimensi), yaitu
setelah 3 item dari arah yang cukup berbeda ter-administer sehingga $\mathbf{FIM}_{cum}$ mencapai
rank penuh.

---

#### 2.2.2 Case Non-Degenerate

**Round 4, $\hat{\boldsymbol{\theta}} = [1.0, -0.5, 0.5]$, 3 item administered**

Setelah 3 item (`r001, v001, n001`) di-administer, $\mathbf{FIM}_{cum}$ terakumulasi rank-3 (full
rank — prasyarat A-optimal agar invers terdefinisi):

$$\mathbf{FIM}_{cum} = \begin{bmatrix} 0.3434 & 0.1964 & 0.2142 \\ 0.1964 & 0.9007 & 0.3008 \\ 0.2142 & 0.3008 & 0.6051 \end{bmatrix}$$

Diagonal $[0.3434, 0.9007, 0.6051]$ → dimensi verbal (indeks 0) punya informasi paling rendah →
A-optimal diharapkan memprioritaskan item dengan loading verbal tinggi.

Contoh lengkap kandidat **m2p-v002** ($\mathbf{a}=[1.7, 0.2, 0.2]$, $d=0.10$):

$$z = 1.7(1.0) + 0.2(-0.5) + 0.2(0.5) + 0.10 = 1.80$$

$$P = \sigma(1.80) = 0.8581$$

$$Q = 0.1419$$

$$w = PQ = 0.1217$$

$$\mathbf{FIM}_i = 0.1217 \times \mathbf{a} \cdot \mathbf{a}^\top = \begin{bmatrix} 0.3517 & 0.0414 & 0.0414 \\ 0.0414 & 0.0049 & 0.0049 \\ 0.0414 & 0.0049 & 0.0049 \end{bmatrix}$$

$$\mathbf{FIM}_{updated} = \mathbf{FIM}_{cum} + \mathbf{FIM}_i = \begin{bmatrix} 0.6951 & 0.2378 & 0.2556 \\ 0.2378 & 0.9056 & 0.3057 \\ 0.2556 & 0.3057 & 0.6100 \end{bmatrix}$$

Inverse via metode adjugate/kofaktor ($\det = 0.2625$):

$$\mathbf{FIM}_{updated}^{-1} \approx \begin{bmatrix} 1.748 & -0.255 & -0.605 \\ -0.255 & 1.366 & -0.578 \\ -0.605 & -0.578 & 2.183 \end{bmatrix}$$

$$\operatorname{tr}\!\left(\mathbf{FIM}_{updated}^{-1}\right) = 1.748 + 1.366 + 2.183 = 5.297$$

$$a\_score = -5.297$$

---

**3 kandidat lain — prosedur identik** ($z=\mathbf{a}\cdot[1.0,-0.5,0.5]+d$, $\mathbf{FIM}_{cum}$ sama seperti di atas):

**m2p-n002** ($\mathbf{a}=[0.3,1.8,0.4]$, $d=0.50$):
$$z = 0.3(1.0)+1.8(-0.5)+0.4(0.5)+0.50 = 0.10$$

$$P=\sigma(0.10)=0.5250$$

$$Q=0.4750$$

$$w=0.2494$$
$$\mathbf{FIM}_i=\begin{bmatrix}0.0224&0.1347&0.0299\\0.1347&0.8081&0.1796\\0.0299&0.1796&0.0399\end{bmatrix}$$

$$\mathbf{FIM}_{updated}=\begin{bmatrix}0.3658&0.3311&0.2441\\0.3311&1.7088&0.4804\\0.2441&0.4804&0.6450\end{bmatrix}\;(\det=0.2238)$$
$$\mathbf{FIM}_{updated}^{-1}\approx\begin{bmatrix}3.894&-0.430&-1.153\\-0.430&0.787&-0.424\\-1.153&-0.424&2.303\end{bmatrix}$$
$$\operatorname{tr}\!\left(\mathbf{FIM}_{updated}^{-1}\right)=3.894+0.787+2.303=6.984$$

$$a\_score=-6.984$$

---

**m2p-r002** ($\mathbf{a}=[0.3,0.8,1.9]$, $d=0.60$):
$$z = 0.3(1.0)+0.8(-0.5)+1.9(0.5)+0.60 = 1.45$$

$$P=\sigma(1.45)=0.8100$$

$$Q=0.1900$$

$$w=0.1539$$
$$\mathbf{FIM}_i=\begin{bmatrix}0.0139&0.0370&0.0877\\0.0370&0.0985&0.2339\\0.0877&0.2339&0.5556\end{bmatrix}$$

$$\mathbf{FIM}_{updated}=\begin{bmatrix}0.3573&0.2334&0.3019\\0.2334&0.9992&0.5347\\0.3019&0.5347&1.1607\end{bmatrix}\;(\det=0.2332)$$
$$\mathbf{FIM}_{updated}^{-1}\approx\begin{bmatrix}3.747&-0.470&-0.759\\-0.470&1.388&-0.517\\-0.759&-0.517&1.298\end{bmatrix}$$
$$\operatorname{tr}\!\left(\mathbf{FIM}_{updated}^{-1}\right)=3.747+1.388+1.298=6.433$$

$$a\_score=-6.433$$

---

**m2p-r003** ($\mathbf{a}=[0.4,0.3,1.8]$, $d=0.70$):
$$z = 0.4(1.0)+0.3(-0.5)+1.8(0.5)+0.70 = 1.85$$

$$P=\sigma(1.85)=0.8642$$

$$Q=0.1358$$

$$w=0.1174$$
$$\mathbf{FIM}_i=\begin{bmatrix}0.0188&0.0141&0.0845\\0.0141&0.0106&0.0634\\0.0845&0.0634&0.3804\end{bmatrix}$$

$$\mathbf{FIM}_{updated}=\begin{bmatrix}0.3622&0.2105&0.2987\\0.2105&0.9113&0.3642\\0.2987&0.3642&0.9855\end{bmatrix}\;(\det=0.1981)$$
$$\mathbf{FIM}_{updated}^{-1}\approx\begin{bmatrix}3.863&-0.498&-0.987\\-0.498&1.352&-0.348\\-0.987&-0.348&1.443\end{bmatrix}$$
$$\operatorname{tr}\!\left(\mathbf{FIM}_{updated}^{-1}\right)=3.863+1.352+1.443=6.658$$

$$a\_score=-6.658$$

---

**Ranking Round 4** (semua 4 kandidat dihitung dengan cara sama):

| Rank | Item | Area | $a\_score$ | $\operatorname{tr}(\text{inv})$ | Note |
|---|---|---|---|---|---|
| 1 | m2p-v002 | verbal | -5.2956 | 5.2956 | SELECTED |
| 2 | m2p-r002 | reasoning | -6.4255 | 6.4255 | |
| 3 | m2p-r003 | reasoning | -6.6540 | 6.6540 | |
| 4 | m2p-n002 | numeric | -6.9767 | 6.9767 | |

`m2p-v002` menang karena loading verbal tertinggi di antara kandidat, dan verbal adalah dimensi
"terlemah" (informasi diagonal $\mathbf{FIM}_{cum}$ paling rendah) — A-optimal secara agresif
menargetkannya. Sama seperti D-optimal ([#1.2.1](#121-case-degenerate)), **A-optimal juga tidak punya mekanisme tiebreak**
apa pun di API production untuk kasus degenerate di [#2.2.1](#221-case-degenerate) — bedanya, A-optimal mengembalikan
$-\infty$ secara eksplisit (bukan $0$ seperti D-optimal) saat matriks singular, tapi selection
round awal tetap murni bergantung pada urutan array item di kedua metode.

---

### 2.3 Kelebihan & Kekurangan

**Kelebihan:**
- Landasan teori sama kuatnya dengan D-optimal, berasal dari kerangka optimal-design/Mulder & van
  der Linden yang sama [1, Eq.16].
- Interpretasi statistik langsung: meminimalkan **rata-rata standard error** $\hat\theta$ di semua
  dimensi — target yang mudah dijelaskan ke stakeholder non-teknis (skor makin presisi).
- Lebih **agresif menargetkan dimensi terlemah** dibanding D-optimal — cocok ketika satu dimensi
  jauh tertinggal dan perlu diprioritaskan segera.

**Kekurangan:**
- **Degenerate lebih parah** dari D-optimal: butuh $\mathbf{FIM}_{cum}$ full rank ($\text{rank}=k$,
  seluruh $k$ dimensi), bukan hanya rank $\geq 2$ — sehingga degenerasi berlangsung sampai round
  $k+1$ (round 4 untuk $k=3$ dimensi), lebih lama dari D-optimal (round 3).
- **Tidak ada fallback/tiebreak** saat degenerate ($-\infty$ untuk semua kandidat) — sama seperti
  D-optimal, tidak ada mekanisme apa pun di API production untuk kasus ini (lihat [#2.2.1](#221-case-degenerate)/[#1.2.1](#121-case-degenerate)).
  Rekomendasi: gunakan A-optimal hanya setelah minimal $k$ item dari arah berbeda ter-administer.
- Perhitungan invers matriks lebih mahal secara komputasi dibanding determinan (D-optimal) atau
  integral tertutup (KL-information), dan berisiko numerically unstable saat matriks nyaris
  singular.
- Menurut [1, §4.1] hasil A-optimal dan D-optimal pada MCAT umumnya **"largely similar"** tapi
  tidak identik — jadi manfaat tambahannya di atas D-optimal marginal pada kebanyakan skenario,
  kecuali saat ketimpangan antar-dimensi besar.

---

## 3. KL-Information (Kullback-Leibler Information)

### 3.1 Teori

Dikembangkan oleh Chang & Ying (1996) [7], diverifikasi silang dari dua sumber terbuka
[3, Eq.9–10][5, Eq.4&6]. Berbeda dari D-optimal/A-optimal, KL-information **tidak bergantung pada
$\mathbf{FIM}_{cum}$** — skor dihitung langsung dari parameter item ($a$, $d$, $c$) dan
$\hat\theta$ saat ini.

**KL pointwise** (KL divergence eksak antar dua distribusi Bernoulli):

$$
K_i(\theta \,\Vert\, \hat\theta) = P_i(\hat\theta)\ln\frac{P_i(\hat\theta)}{P_i(\theta)} + \big(1-P_i(\hat\theta)\big)\ln\frac{1-P_i(\hat\theta)}{1-P_i(\theta)}
$$

**KL information global** (moving average, diintegralkan di sekitar $\hat\theta$):

$$
\bar{K}_i(\hat\theta) = \int_{\hat\theta-\delta}^{\hat\theta+\delta} K_i(\theta \,\Vert\, \hat\theta)\, d\theta
$$

$$
\delta = \frac{C}{\sqrt{m+1}}
$$

$m$ = jumlah item administered, $C$ = konstanta (default $C=3$, ~3 galat baku asimtotik). Item
dipilih dengan $\arg\max_i \bar{K}_i(\hat\theta)$. Karena model compensatory MIRT $P_i(\theta)$
hanya bergantung pada $\theta$ lewat skalar $z = \mathbf{a}\cdot\theta + d$, reduksi
multidimensional→skalar berlaku eksak (bukan aproksimasi):

$$
K_i(\theta \,\Vert\, \hat\theta) = K_i(z \,\Vert\, \hat{z})
$$

— integral dihitung dalam ruang $z$ satu dimensi.

Karena ini adalah KL divergence Bernoulli baku, nilainya **selalu $\geq 0$** (Gibbs' inequality)
dan **selalu finite** — tidak ada kasus degenerate di round manapun.

**Keterangan variabel:**

| Simbol | Arti |
|---|---|
| $K_i(\theta\Vert\hat\theta)$ | KL divergence *pointwise* item $i$ — "jarak informasi" antara distribusi Bernoulli respons pada $\theta$ vs $\hat\theta$ |
| $\theta$ (di dalam $K_i(\theta\Vert\hat\theta)$ dan integral) | Nilai kemampuan yang **diintegralkan** (variabel dummy di sekitar $\hat\theta$) — **bukan** kemampuan examinee sebenarnya |
| $\hat\theta$ | Estimasi kemampuan examinee **saat ini** (titik pusat integrasi) |
| $P_i(\theta)$, $P_i(\hat\theta)$ | Peluang benar item $i$ pada kemampuan $\theta$ / $\hat\theta$ (fungsi model M2PL/M3PL di [#0](#model--notasi-dasar-dipakai-oleh-ketiga-metode)) |
| $\ln$ | Logaritma natural |
| $\bar K_i(\hat\theta)$ | KL information **global** item $i$ — hasil integral $K_i$ di sekitar $\hat\theta$ (dipakai untuk ranking item) |
| $\int_{\hat\theta-\delta}^{\hat\theta+\delta}$ | Integral pada interval sepanjang $2\delta$ berpusat di $\hat\theta$ |
| $\delta$ | Setengah lebar interval integrasi — mengecil seiring bertambahnya item ($\delta=C/\sqrt{m+1}$) |
| $C$ | Konstanta skala interval (default $C=3$, ~3 galat baku asimtotik) |
| $m$ | Jumlah item yang sudah di-administer sejauh ini di sesi berjalan |
| $z$, $\hat z$ | Linear predictor $z=\mathbf{a}\cdot\theta+d$ dan $\hat z=\mathbf{a}\cdot\hat\theta+d$ — reduksi skalar dari $\theta$ multidimensional (lihat pembuktian $K_i(\theta\Vert\hat\theta)=K_i(z\Vert\hat z)$ di atas) |

---

### 3.2 Perhitungan Manual Per Item — Tidak Ada Kasus Degenerate

Berbeda dari D-optimal ([#1.2.1](#121-case-degenerate)) dan A-optimal ([#2.2.1](#221-case-degenerate)), KL-information **tidak punya kasus
degenerate**: skor tidak bergantung pada $\mathbf{FIM}_{cum}$ sama sekali (lihat [#3.1](#31-teori)), sehingga
$kl\_score$ selalu finite dan $\geq 0$ di round manapun — termasuk round 1 tanpa item administered
sama sekali. Dua contoh di bawah (Round 1 dan Round 4) sama-sama merupakan kasus **non-degenerate**;
disajikan berdampingan untuk menunjukkan sifat adaptif $\delta$ terhadap $m$.

---

#### 3.2.1 Round 1

**$\hat{\boldsymbol{\theta}} = [0,0,0]$, $\delta = 3/\sqrt{1} = 3.0$**

Contoh lengkap **m2p-v002** ($\mathbf{a}=[1.7, 0.2, 0.2]$, $d=0.10$):

$$\hat{z} = 1.7(0) + 0.2(0) + 0.2(0) + 0.10 = 0.1000$$

$$P_i(\hat z) = \sigma(0.1000) = 0.524979$$

$$[\hat z - \delta,\; \hat z + \delta] = [-2.9000,\; 3.1000]$$

$$P(\text{lower}) = \sigma(-2.9000) = 0.052154$$

$$P(\text{upper}) = \sigma(3.1000) = 0.956893$$

$$K_i(\text{lower}\Vert \hat z) = 0.524979\ln\!\frac{0.524979}{0.052154} + 0.475021\ln\!\frac{0.475021}{0.947846} = 0.884104$$

$$K_i(\text{upper}\Vert \hat z) = 0.524979\ln\!\frac{0.524979}{0.956893} + 0.475021\ln\!\frac{0.475021}{0.043107} = 0.824730$$

$$kl\_score = \int_{-2.9}^{3.1} K_i(z \Vert 0.10)\, dz \;\;(\text{Simpson 40-panel}) = 1.884718 \quad (\text{highest, Round 1})$$

Verifikasi manual dengan Simpson $n=4$ (5 titik: lower, $q_1$, $\hat z$, $q_3$, upper),
$K_i(\hat z \Vert \hat z) = 0$ selalu:

$$h = \frac{3.1-(-2.9)}{4} = 1.5$$

$$q_1 = -1.4 \to K_{q_1}=0.263490$$

$$q_3 = 1.6 \to K_{q_3}=0.252035$$

$$\text{simpson}(n{=}4) = \frac{1.5}{3}\Big[0.884104 + 4(0.263490) + 0 + 4(0.252035) + 0.824730\Big] = 1.885467$$

(selisih kecil vs $1.884718$ dari kode murni akibat resolusi kuadratur $n=4$ vs $n=40$).

---

**6 item lain — prosedur identik** ($\delta=3.0$ untuk semua, karena $m=0$ untuk seluruh item di round ini):

**m2p-v001** ($\mathbf{a}=[1.9,0.2,0.3]$, $d=0.40$):
$$\hat z = 0.40$$

$$P_i(\hat z)=\sigma(0.40)=0.598688$$
$$[\hat z-\delta,\hat z+\delta]=[-2.6000,3.4000]$$

$$P(\text{lower})=0.069138$$

$$P(\text{upper})=0.967705$$
$$K_i(\text{lower}\Vert\hat z)=0.598688\ln\tfrac{0.598688}{0.069138}+0.401312\ln\tfrac{0.401312}{0.930862}=0.954692$$
$$K_i(\text{upper}\Vert\hat z)=0.598688\ln\tfrac{0.598688}{0.967705}+0.401312\ln\tfrac{0.401312}{0.032295}=0.723750$$
$$kl\_score=\int_{-2.6}^{3.4}K_i(z\Vert0.40)\,dz=1.840805$$

---

**m2p-n001** ($\mathbf{a}=[0.3,1.9,0.4]$, $d=0.80$):
$$\hat z = 0.80$$

$$P_i(\hat z)=\sigma(0.80)=0.689974$$
$$[\hat z-\delta,\hat z+\delta]=[-2.2000,3.8000]$$

$$P(\text{lower})=0.099750$$

$$P(\text{upper})=0.978119$$
$$K_i(\text{lower}\Vert\hat z)=0.689974\ln\tfrac{0.689974}{0.099750}+0.310026\ln\tfrac{0.310026}{0.900250}=1.003906$$
$$K_i(\text{upper}\Vert\hat z)=0.689974\ln\tfrac{0.689974}{0.978119}+0.310026\ln\tfrac{0.310026}{0.021881}=0.581100$$
$$kl\_score=\int_{-2.2}^{3.8}K_i(z\Vert0.80)\,dz=1.708207 \quad(\text{terendah, Round 1})$$

---

**m2p-n002** ($\mathbf{a}=[0.3,1.8,0.4]$, $d=0.50$):
$$\hat z = 0.50$$

$$P_i(\hat z)=\sigma(0.50)=0.622459$$
$$[\hat z-\delta,\hat z+\delta]=[-2.5000,3.5000]$$

$$P(\text{lower})=0.075858$$

$$P(\text{upper})=0.970688$$
$$K_i(\text{lower}\Vert\hat z)=0.622459\ln\tfrac{0.622459}{0.075858}+0.377541\ln\tfrac{0.377541}{0.924142}=0.972191$$
$$K_i(\text{upper}\Vert\hat z)=0.622459\ln\tfrac{0.622459}{0.970688}+0.377541\ln\tfrac{0.377541}{0.029312}=0.688295$$
$$kl\_score=\int_{-2.5}^{3.5}K_i(z\Vert0.50)\,dz=1.815040$$

---

**m2p-r001** ($\mathbf{a}=[0.5,0.4,2.0]$, $d=0.30$):
$$\hat z = 0.30$$

$$P_i(\hat z)=\sigma(0.30)=0.574443$$
$$[\hat z-\delta,\hat z+\delta]=[-2.7000,3.3000]$$

$$P(\text{lower})=0.062973$$

$$P(\text{upper})=0.964429$$
$$K_i(\text{lower}\Vert\hat z)=0.574443\ln\tfrac{0.574443}{0.062973}+0.425557\ln\tfrac{0.425557}{0.937027}=0.934016$$
$$K_i(\text{upper}\Vert\hat z)=0.574443\ln\tfrac{0.574443}{0.964429}+0.425557\ln\tfrac{0.425557}{0.035571}=0.758536$$
$$kl\_score=\int_{-2.7}^{3.3}K_i(z\Vert0.30)\,dz=1.861145$$

---

**m2p-r002** ($\mathbf{a}=[0.3,0.8,1.9]$, $d=0.60$):
$$\hat z = 0.60$$

$$P_i(\hat z)=\sigma(0.60)=0.645656$$
$$[\hat z-\delta,\hat z+\delta]=[-2.4000,3.6000]$$

$$P(\text{lower})=0.083173$$

$$P(\text{upper})=0.973403$$
$$K_i(\text{lower}\Vert\hat z)=0.645656\ln\tfrac{0.645656}{0.083173}+0.354344\ln\tfrac{0.354344}{0.916827}=0.986317$$
$$K_i(\text{upper}\Vert\hat z)=0.645656\ln\tfrac{0.645656}{0.973403}+0.354344\ln\tfrac{0.354344}{0.026597}=0.652500$$
$$kl\_score=\int_{-2.4}^{3.6}K_i(z\Vert0.60)\,dz=1.784127$$

---

**m2p-r003** ($\mathbf{a}=[0.4,0.3,1.8]$, $d=0.70$):
$$\hat z = 0.70$$

$$P_i(\hat z)=\sigma(0.70)=0.668188$$
$$[\hat z-\delta,\hat z+\delta]=[-2.3000,3.7000]$$

$$P(\text{lower})=0.091123$$

$$P(\text{upper})=0.975873$$
$$K_i(\text{lower}\Vert\hat z)=0.668188\ln\tfrac{0.668188}{0.091123}+0.331812\ln\tfrac{0.331812}{0.908877}=0.996923$$
$$K_i(\text{upper}\Vert\hat z)=0.668188\ln\tfrac{0.668188}{0.975873}+0.331812\ln\tfrac{0.331812}{0.024127}=0.616673$$
$$kl\_score=\int_{-2.3}^{3.7}K_i(z\Vert0.70)\,dz=1.748393$$

---

**Ranking Round 1:**

| Rank | Item | Area | $kl\_score$ | $\hat z$ | $\delta$ | Note |
|---|---|---|---|---|---|---|
| 1 | m2p-v002 | verbal | 1.884718 | 0.1000 | 3.0000 | SELECTED |
| 2 | m2p-r001 | reasoning | 1.861145 | 0.3000 | 3.0000 | |
| 3 | m2p-v001 | verbal | 1.840805 | 0.4000 | 3.0000 | |
| 4 | m2p-n002 | numeric | 1.815040 | 0.5000 | 3.0000 | |
| 5 | m2p-r002 | reasoning | 1.784127 | 0.6000 | 3.0000 | |
| 6 | m2p-r003 | reasoning | 1.748393 | 0.7000 | 3.0000 | |
| 7 | m2p-n001 | numeric | 1.708207 | 0.8000 | 3.0000 | |

`m2p-v002` menang  karena $\hat z$ paling dekat ke pusat interval integrasi ($P\approx 0.5$, titik kemiringan ICC maksimum) — konsisten dengan temuan Chang & Ying bahwa KLI dapat berbeda dari seleksi berbasis Fisher information murni pada tahap awal tes [3].

---

#### 3.2.2 Round 4

**$\hat{\boldsymbol{\theta}} = [1.0, -0.5, 0.5]$, 3 item administered, $\delta = 3/\sqrt{4} = 1.5$**

Contoh lengkap **m2p-n002** ($\mathbf{a}=[0.3, 1.8, 0.4]$, $d=0.50$):

$$\hat{z} = 0.3(1.0) + 1.8(-0.5) + 0.4(0.5) + 0.50 = 0.1000$$

$$P_i(\hat z) = \sigma(0.1000) = 0.524979$$

$$[\hat z - \delta,\; \hat z + \delta] = [-1.4000,\; 1.6000]$$

$$P(\text{lower}) = \sigma(-1.4000) = 0.197816$$

$$P(\text{upper}) = \sigma(1.6000) = 0.832018$$

$$K_i(\text{lower}\Vert \hat z) = 0.524979\ln\!\frac{0.524979}{0.197816} + 0.475021\ln\!\frac{0.475021}{0.802184} = 0.263490$$

$$K_i(\text{upper}\Vert \hat z) = 0.524979\ln\!\frac{0.524979}{0.832018} + 0.475021\ln\!\frac{0.475021}{0.167982} = 0.252035$$

$$kl\_score = \int_{-1.4}^{1.6} K_i(z \Vert 0.10)\, dz \;\;(\text{Simpson 40-panel}) = 0.266355 \quad (\text{highest, Round 4})$$

Verifikasi manual Simpson $n=4$:

$$h = \frac{1.6-(-1.4)}{4} = 0.75$$

$$q_1 = -0.65 \to K_{q_1}=0.069393$$

$$q_3 = 0.85 \to K_{q_3}=0.067734$$

$$\text{simpson}(n{=}4) = \frac{0.75}{3}\Big[0.263490 + 4(0.069393) + 0 + 4(0.067734) + 0.252035\Big] = 0.266008$$

---

**3 kandidat lain — prosedur identik** ($\delta=1.5$ untuk semua, karena $m=3$ untuk seluruh kandidat di round ini):

**m2p-v002** ($\mathbf{a}=[1.7,0.2,0.2]$, $d=0.10$):
$$\hat z = 1.7(1.0)+0.2(-0.5)+0.2(0.5)+0.10 = 1.80$$

$$P_i(\hat z)=\sigma(1.80)=0.858149$$
$$[\hat z-\delta,\hat z+\delta]=[0.3000,3.3000]$$

$$P(\text{lower})=0.574443$$

$$P(\text{upper})=0.964429$$
$$K_i(\text{lower}\Vert\hat z)=0.858149\ln\tfrac{0.858149}{0.574443}+0.141851\ln\tfrac{0.141851}{0.425557}=0.188601$$
$$K_i(\text{upper}\Vert\hat z)=0.858149\ln\tfrac{0.858149}{0.964429}+0.141851\ln\tfrac{0.141851}{0.035571}=0.096018$$
$$kl\_score=\int_{0.3}^{3.3}K_i(z\Vert1.80)\,dz=0.140419$$

---

**m2p-r002** ($\mathbf{a}=[0.3,0.8,1.9]$, $d=0.60$):
$$\hat z = 0.3(1.0)+0.8(-0.5)+1.9(0.5)+0.60 = 1.45$$

$$P_i(\hat z)=\sigma(1.45)=0.809998$$
$$[\hat z-\delta,\hat z+\delta]=[-0.0500,2.9500]$$

$$P(\text{lower})=0.487503$$

$$P(\text{upper})=0.950263$$
$$K_i(\text{lower}\Vert\hat z)=0.809998\ln\tfrac{0.809998}{0.487503}+0.190002\ln\tfrac{0.190002}{0.512497}=0.222734$$
$$K_i(\text{upper}\Vert\hat z)=0.809998\ln\tfrac{0.809998}{0.950263}+0.190002\ln\tfrac{0.190002}{0.049737}=0.125295$$
$$kl\_score=\int_{-0.05}^{2.95}K_i(z\Vert1.45)\,dz=0.173915$$

---

**m2p-r003** ($\mathbf{a}=[0.4,0.3,1.8]$, $d=0.70$):
$$\hat z = 0.4(1.0)+0.3(-0.5)+1.8(0.5)+0.70 = 1.85$$

$$P_i(\hat z)=\sigma(1.85)=0.864127$$
$$[\hat z-\delta,\hat z+\delta]=[0.3500,3.3500]$$

$$P(\text{lower})=0.586618$$

$$P(\text{upper})=0.966105$$
$$K_i(\text{lower}\Vert\hat z)=0.864127\ln\tfrac{0.864127}{0.586618}+0.135873\ln\tfrac{0.135873}{0.413382}=0.183537$$
$$K_i(\text{upper}\Vert\hat z)=0.864127\ln\tfrac{0.864127}{0.966105}+0.135873\ln\tfrac{0.135873}{0.033895}=0.092257$$
$$kl\_score=\int_{0.35}^{3.35}K_i(z\Vert1.85)\,dz=0.135822 \quad(\text{terendah, Round 4})$$

---

**Ranking Round 4:**

| Rank | Item | Area | $kl\_score$ | $\hat z$ | $\delta$ | Note |
|---|---|---|---|---|---|---|
| 1 | m2p-n002 | numeric | 0.266355 | 0.1000 | 1.5000 | SELECTED |
| 2 | m2p-r002 | reasoning | 0.173915 | 1.4500 | 1.5000 | |
| 3 | m2p-v002 | verbal | 0.140419 | 1.8000 | 1.5000 | |
| 4 | m2p-r003 | reasoning | 0.135822 | 1.8500 | 1.5000 | |

**Selected: `m2p-n002`** — interval integrasi mengecil dari $\delta=3.0$ (Round 1) menjadi
$\delta=1.5$ (Round 4), skor keseluruhan turun skala ($\sim$0.14–0.27 vs $\sim$1.7–1.9 di Round 1),
dan pemenangnya berganti dari `m2p-v002` ke `m2p-n002` ($\hat z=0.10$ paling dekat pusat kurva ICC
untuk $\hat\theta$ baru ini) — menunjukkan sifat **adaptif**: $\delta$ mengecil seiring bertambahnya
item administered, membuat kriteria makin "lokal"/mirip Fisher information murni. Berbeda dengan
D-optimal/A-optimal, transisi Round 1 → Round 4 di sini **tidak melibatkan perubahan status
degenerate → non-degenerate** — keduanya sama-sama valid dan finite sejak awal.

---

### 3.3 Kelebihan & Kekurangan

**Kelebihan:**
- **Tidak pernah degenerate** — skor selalu finite dan $\geq 0$ (Gibbs' inequality) di round
  manapun, termasuk round 1 tanpa $\mathbf{FIM}_{cum}$ sama sekali. D-optimal dan A-optimal
  **butuh** tiebreak di round-round awal dan **tidak punya** fallback (murni
  urutan array, lihat [#1.2.1](#121-case-degenerate)/[#2.2.1](#221-case-degenerate)) — KL-information tidak pernah menghadapi masalah ini sama
  sekali karena skornya tidak pernah seri/tak terdefinisi.
- Terbukti secara empiris **mengurangi bias dan MSE** estimasi kemampuan dibanding Fisher
  information murni (MFI), khususnya pada tes pendek ($m<30$) atau tahap awal tes ketika
  $\hat\theta$ masih jauh dari $\theta$ sebenarnya [3, p.6].
- Mekanisme adaptif built-in: interval integrasi $\delta$ mengecil otomatis seiring bertambahnya
  item ($\delta = C/\sqrt{m+1}$) — "global" di awal (toleran ketidakpastian $\hat\theta$ awal),
  "lokal" di akhir (presisi).
- Reduksi multidimensional→skalar bersifat **eksak** (bukan aproksimasi) untuk model compensatory
  MIRT, karena $P_i$ hanya bergantung pada $\theta$ lewat $z = \mathbf{a}\cdot\theta+d$.

**Kekurangan:**
- **Greedy murni terhadap $\hat\theta$ saat ini** — tidak mempertimbangkan
  $\mathbf{FIM}_{cum}$/informasi kumulatif dari item-item sebelumnya sama sekali, berbeda dari
  D-optimal/A-optimal yang eksplisit menyeimbangkan kontribusi terhadap total informasi terkumpul.
  Berpotensi kurang optimal untuk menyeimbangkan informasi **antar-dimensi** (tidak ada insentif
  eksplisit menargetkan dimensi yang lemah).
- Bergantung pada pilihan konstanta $C$ (default 3) yang menurut Chang & Ying sendiri "ambigu" —
  tidak ada nilai tunggal yang ditentukan secara pasti oleh teori [3, p.6].
- Definisi generalisasi multidimensional (interval integrasi dalam ruang $z$) adalah **keputusan
  implementasi**, bukan dikutip langsung dari paper multidimensional formal (Veldkamp & van der
  Linden 2002; Mulder & van der Linden 2010) yang tidak berhasil diakses full-text saat notes
  ditulis — meski kesetaraan $K_i(\theta\Vert\hat\theta)=K_i(z\Vert\hat z)$ tetap terbukti benar
  secara independen dari model sendiri.
- Biaya komputasi integral numerik (Simpson, 40 panel per item per kandidat) lebih tinggi
  per-evaluasi dibanding determinan/trace tunggal, meski ini bukan bottleneck signifikan dalam
  praktik.
- Butuh parameter mentah item ($a$, $d$, $c$) langsung, bukan hanya $\mathbf{FIM}_i$ — sedikit
  lebih kompleks secara komputasi dibanding 3 metode lain yang berbasis FIM murni.

---

## 4. Ringkasan Perbandingan

| Metode | Formula | Bergantung $\mathbf{FIM}_{cum}$? | Degenerate? |
|---|---|---|---|
| D-optimal | $\arg\max \det(\mathbf{FIM}_{cum} + \mathbf{FIM}_i)$ | Ya | Round 1–2 |
| A-optimal | $\arg\max -\operatorname{tr}\big((\mathbf{FIM}_{cum} + \mathbf{FIM}_i)^{-1}\big)$ | Ya | Round 1–3 |
| KL-information | $\arg\max \int_{\hat z-\delta}^{\hat z+\delta} K_i(z\Vert\hat z)\,dz$ | Tidak | Tidak pernah |

| Aspek | D-Optimal | A-Optimal | KL-Information |
|---|---|---|---|
| Basis teori | Volume ellipsoid kepercayaan (determinan) | Rata-rata varians (trace invers) | KL divergence Bernoulli lokal |
| Perlu $\mathbf{FIM}_{cum}$ full rank? | Rank ≥ 2 | Rank = $k$ (semua dimensi) | Tidak perlu |
| Round mulai efektif | Round 3 | Round $k+1$ (round 4 utk $k=3$) | Round 1 |
| Tiebreak saat degenerate | Tidak ada (urutan array) | Tidak ada (urutan array) | N/A (tidak pernah degenerate) |
| Menargetkan dimensi lemah? | Ya (multiplikatif, seimbang) | Ya (agresif ke dimensi terlemah) | Tidak eksplisit (lokal per-item) |
| Cocok untuk tahap tes | Menengah–akhir | Menengah–akhir (setelah rank penuh) | Awal tes ($\hat\theta$ belum presisi) |

---

## Referensi

**[1]** Mulder, J., & van der Linden, W. J. (2009). Multidimensional Adaptive Testing with Optimal
Design Criteria for Item Selection. *Psychometrika*, 74(2), 273–296.
https://doi.org/10.1007/s11336-008-9097-5 — Full text gratis (PubMed Central, open access):
https://pmc.ncbi.nlm.nih.gov/articles/PMC2813188/ (mirror PDF jurnal dengan nomor halaman asli:
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A3BFF7744EDCE563819C31270D9C7E7D/S0033312300021608a.pdf/multidimensional-adaptive-testing-with-optimal-design-criteria-for-item-selection.pdf).
Sumber utama untuk: model M3PL (Eq.1, p.275), struktur FIM & sifat rank-1 (Eq.4–5, p.276–278),
kriteria D-optimal (Eq.8&13, p.277&280–282), dan kriteria A-optimal (Eq.16, §4.1.2, p.±282–284).

**[2]** Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). ERIC Clearinghouse on
Assessment and Evaluation, University of Maryland. Full text gratis (ERIC ED458219):
https://files.eric.ed.gov/fulltext/ED458219.pdf (mirror: https://www.ime.unicamp.br/~cnaber/Baker_Book.pdf).
Dipakai untuk item information 2PL (p.109, Eq.6-3: $I(\theta)=a^2 P(\theta) Q(\theta)$) dan 3PL
(p.111, Eq.6-5: $I(\theta)=a^2\frac{Q(\theta)}{P(\theta)}\frac{(P(\theta)-c)^2}{(1-c)^2}$) —
dibuktikan identik secara aljabar dengan formula $w=(P')^2/(PQ)$ yang dipakai di ketiga metode
pada dokumen ini.

**[3]** Han, K. T. (2018). Components of the item selection algorithm in computerized adaptive
testing. *Journal of Educational Evaluation for Health Professions*, 15, Article 7.
https://doi.org/10.3352/jeehp.2018.15.7 — Open access (CC-BY). PDF:
https://www.jeehp.org/upload/pdf/jeehp-15-7.pdf (mirror: https://pmc.ncbi.nlm.nih.gov/articles/PMC5968224/).
Sumber rumus KL pointwise (Eq.9) dan KL information global (Eq.10), §"Kullback-Leibler information
criterion", p.6 — termasuk penjelasan $\delta=C/\sqrt{m}$ dan sitasi ke Chang & Ying (1996) [7].

**[4]** St. John, R. C., & Draper, N. R. (1975). D-Optimality for Regression Designs: A Review.
*Technometrics*, 17(1), 15–23. Free download: https://www.stat.cmu.edu/technometrics/70-79/VOL-17-01/v1701015.pdf.
Sumber asal-usul historis kriteria D-optimality (Wald 1943; Kiefer & Wolfowitz 1959) dan definisi
formal $\max_\xi \det(M(\xi))$ (p.16, Definition 1 & Eq.11).

**[5]** Sorrel, M. A., Barrada, J. R., de la Torre, J., & Abad, F. J. (2020). Adapting cognitive
diagnosis computerized adaptive testing item selection rules to traditional item response theory.
*PLOS ONE*, 15(1), e0227196. https://doi.org/10.1371/journal.pone.0227196 — Open access (CC-BY). PDF:
https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0227196&type=printable.
Verifikasi silang independen untuk rumus KL information (Eq.4 & Eq.6, p.3–4), identik dengan [3].

**[6]** Segall, D. O. (1996). Multidimensional Adaptive Testing. *Psychometrika*, 61(2), 331–354.
Sumber asli/historis kriteria D-optimal untuk MIRT-CAT, dirujuk lewat [1] p.277. Tidak berhasil
diakses gratis (Cambridge Core/SpringerLink berbayar) — klaim yang berasal dari Segall (1996) pada
dokumen ini hanya diverifikasi secara tidak langsung lewat kutipan & rumus yang direproduksi di [1].

**[7]** Chang, H.-H., & Ying, Z. (1996). A global information approach to computerized adaptive
testing. *Applied Psychological Measurement*, 20(3), 213–229. https://doi.org/10.1177/014662169602000303
— Sumber primer/asli rumus KL-information. PDF asli tidak dapat diakses bebas-unduh saat penelusuran
(SAGE berbayar; University of Minnesota Digital Conservancy & ResearchGate menolak akses). Rumus pada
dokumen ini diverifikasi lewat [3] dan [5] yang mereproduksi rumus Chang & Ying secara verbatim dan
saling cocok satu sama lain (verifikasi silang independen).

### Peta Sitasi per Formula

| Formula | Dipakai di metode | Sumber |
|---|---|---|
| $P = c+(1-c)\sigma(\mathbf{a}\cdot\theta+d)$ (M3PL/M2PL) | Ketiganya | [1] Eq.1 |
| $I_i(\theta) = w(\mathbf{a} \cdot \mathbf{a}^\top)$, rank-1 | Ketiganya (FIM dasar) | [1] Eq.4–5; [2] Eq.6-3/6-5 |
| $\arg\max \det(\mathbf{FIM}_{cum}+\mathbf{FIM}_i)$ | D-Optimal | [1] Eq.8&13; asal-usul [4][6] |
| $\arg\min \operatorname{tr}\big((\mathbf{FIM}_{cum}+\mathbf{FIM}_i)^{-1}\big)$ | A-Optimal | [1] Eq.16 |
| $K_i(\theta\Vert\hat\theta)$, $\bar K_i(\hat\theta)=\int K_i\, d\theta$, $\delta=C/\sqrt{m}$ | KL-Information | [3] Eq.9–10; [5] Eq.4&6; asal-usul [7] |
