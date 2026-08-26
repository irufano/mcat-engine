---
title: "Ability Estimation - MCAT (MLE, MAP, EAP)"
date: "2026-07-14"
description: "Dokumen ini merangkum tiga metode estimasi kemampuan multidimensional CAT (MLE, MAP, EAP)"
author: "irufano"
tags:
  - AI
  - CAT
  - MCAT
  - Adaptive Test
image: "https://www.assessmentworkshop.com/wp-content/uploads/2022/04/CAT-Infographic.png"
---

Dokumen ini merangkum tiga metode estimasi kemampuan (ability, $\theta$) multidimensional CAT:
**Maximum Likelihood Estimation (MLE)**, **Maximum A Posteriori (MAP / Bayes Modal)**, dan
**Expected A Posteriori (EAP)**. Fokus dokumen: landasan teori tiap metode, pembuktian aljabar bahwa perhitungan identik dengan literatur, dan contoh perhitungan manual
per metode.

---

## 1. Konsep Fondasi: Prior, Likelihood, dan Posterior

Sebelum masuk ke formula teknis, berikut intuisi ketiga konsep yang menjadi inti ketiga metode estimasi:


### **Prior: "Apa yang kita tahu tentang $\theta$ sebelum ada data?"**

Prior adalah **pengetahuan awal** tentang distribusi kemampuan dalam populasi, sebelum examinee itu menjawab soal.

- **Contoh:** Asumsi umum populasi peserta tes adalah $\theta \sim N(0,1)$ (normal dengan mean 0, variance 1)
  - Ini berarti: "Sebelum tes, kami percaya kebanyakan orang punya kemampuan dekat 0, dan semakin jauh dari 0 semakin jarang"
  - Misal $\theta=+3$ dianggap **sangat jarang** di populasi (hanya 0.13% dalam normal)

- Prior **tidak bergantung pada respons** - pure belief/asumsi tentang populasi, bukan tentang satu examinee
- Prior adalah **penyeimbang** antara data (likelihood) dan asumsi awal tentang populasi

**Rumus sederhana:** $\pi(\theta) = N(\mu, \sigma^2)$ (biasanya prior normal dengan mean $\mu$ dan variance $\sigma^2$)

---


### **Likelihood: "Seberapa cocok data dengan parameter $\theta$?"**

Likelihood menjawab: *Jika kemampuan examinee adalah $\theta$, seberapa besar peluang dia menjawab respons yang kita observasi?*

- **Contoh:** Examinee menjawab 3 item: benar, salah, benar (respons $\mathbf{u}=[1,0,1]$)
  - Jika kemampuannya $\theta=0$ (median), likelihood mungkin 0.1 (tidak terlalu cocok - item pertama harusnya lebih mudah)
  - Jika kemampuannya $\theta=+1$ (tinggi), likelihood mungkin 0.5 (lebih cocok - pola respons sesuai dengan kemampuan lebih tinggi)
  
- Likelihood adalah **fungsi dari $\theta$** yang menggukur "bukti yang ada mendukung $\theta$ berapa"
- Semakin tinggi likelihood, semakin "masuk akal" nilai $\theta$ tersebut berdasarkan data respons

**Rumus sederhana:** $L(\theta) = \prod_i P_i(\theta)^{u_i} \cdot Q_i(\theta)^{1-u_i}$ (produk probabilitas per item)

---

### **Posterior: "Apa yang kita tahu tentang $\theta$ setelah melihat data?"**

Posterior adalah **update belief** tentang $\theta$ setelah menggabungkan prior (pengetahuan awal) dengan likelihood (bukti dari respons).

**Formula Bayes:**
$$p(\theta \mid \mathbf{u}) = \frac{p(\mathbf{u} \mid \theta) \cdot p(\theta)}{p(\mathbf{u})} = \frac{\text{Likelihood} \times \text{Prior}}{\text{Normalisasi}}$$

- **Contoh interpretasi:** 
  - Prior: "Mayoritas populasi punya $\theta$ dekat 0" → $N(0,1)$
  - Likelihood dari data: "Respons ini cocok dengan $\theta=+1$" → peak di +1
  - Posterior: "Setelah data ini, estimate kita adalah $\theta=+0.5$" → compromise antara prior (0) dan likelihood (+1)

- Posterior adalah **distribusi probabilitas atas $\theta$** (bukan single point)
- Posterior bergantung pada **keduanya**: prior (populasi) dan likelihood (data eksaminee)

**Intuisi numeric:** Jika prior sangat kuat (variance kecil), estimasi akan tertarik ke mean prior. Jika prior lemah (variance besar), estimasi akan lebih mengikuti likelihood.

---

### Ringkasan: Bayesian dalam Tiga Konsep

![Bayesian concept](/posts/cat/ability-estimation-mcat/bayesian-concept.png)

Dalam pendekatan Bayesian, terdapat tiga konsep penting yang menjadi fondasi semua metode estimasi:

| Konsep | Definisi | Peran |
|---|---|---|
| **Prior** | Keyakinan awal tentang kemampuan peserta **sebelum** mengerjakan soal | Pengetahuan tentang populasi secara umum |
| **Likelihood** | Probabilitas peserta memberikan respons tertentu **jika** memiliki kemampuan tertentu | Bukti dari data yang terobservasi (respons peserta) |
| **Posterior** | Keyakinan yang telah diperbarui **setelah** melihat respons peserta | Kombinasi prior + likelihood (Bayesian update) |

---

### Contoh Numerik Sederhana

**Setup:** Misalkan kemampuan peserta ($\theta$) hanya mungkin berada pada tiga nilai diskrit: Rendah, Sedang, atau Tinggi.

#### Step 1: Prior (sebelum ada data)

Sebelum peserta mengerjakan soal, kita belum tahu kemampuannya. Asumsi awal (prior) hampir merata:

| Kemampuan ($\theta$) | Prior $\pi(\theta)$ | Interpretasi |
|---|---|---|
| **Rendah** | 0.33 | Peluang peserta rendah = 33% |
| **Sedang** | 0.33 | Peluang peserta sedang = 33% |
| **Tinggi** | 0.34 | Peluang peserta tinggi = 34% |

Distribusi hampir seragam karena belum ada informasi dari peserta.

---

#### Step 2: Likelihood (dari satu respons)

Sekarang peserta menjawab **satu soal yang SULIT dengan BENAR**.

Kemudian kita tanya: *"Jika peserta punya kemampuan X, seberapa besar peluang dia bisa jawab soal sulit ini dengan benar?"*

| Kemampuan | Likelihood $L(\text{benar soal sulit} \mid \theta)$ | Interpretasi |
|---|---|---|
| **Rendah** | 0.1 | Jika rendah, peluang benar soal sulit hanya 10% |
| **Sedang** | 0.4 | Jika sedang, peluang benar soal sulit 40% |
| **Tinggi** | 0.85 | Jika tinggi, peluang benar soal sulit 85% |

**Observasi:** Peserta **bisa** benar soal sulit di semua kemampuan level, tapi **paling cocok** dengan kemampuan tinggi.

---

#### Step 3: Posterior (setelah melihat respons)

Sekarang kita gabungkan prior + likelihood menggunakan formula Bayes:

$$p(\theta \mid \text{benar soal sulit}) \propto L \times \pi$$

| Kemampuan | Prior | Likelihood | Prior × Likelihood | Posterior (dinormalisasi) |
|---|---|---|---|---|
| **Rendah** | 0.33 | 0.1 | 0.033 | 0.05 |
| **Sedang** | 0.33 | 0.4 | 0.132 | 0.30 |
| **Tinggi** | 0.34 | 0.85 | 0.289 | **0.65** |
| **Sum** | | | 0.454 | 1.00 |

**Interpretasi:** Berdasarkan jawaban benar untuk soal sulit tersebut:
- Peluang Rendah turun drastis: 33% → 5%
- Peluang Sedang sedikit meningkat: 33% → 30%
- Peluang Tinggi meningkat signifikan: 34% → **65%**

**Kesimpulan:** Setelah satu respons, kita sekarang **65% yakin** bahwa peserta memiliki kemampuan tinggi (vs 34% sebelumnya).

---

### Pola Umum Bayesian Update

```
Posterior ∝ Prior × Likelihood

Artinya: Belief terbaru = (Keyakinan awal) × (Dukungan dari data)
```

**Intuisi:**
- Jika prior kuat tapi likelihood lemah → posterior condong ke prior
- Jika prior lemah tapi likelihood kuat → posterior condong ke likelihood
- Jika prior dan likelihood seimbang → posterior adalah compromise

---

## 2. Model dan Notasi Dasar (dipakai oleh ketiga metode)

Model respons item M3PL/M2PL [1, Eq.1, p.275]:

$$
P_i(\boldsymbol\theta) = c_i + (1-c_i)\,\sigma(\mathbf{a}_i\cdot\boldsymbol\theta + d_i), \qquad
\sigma(z) = \frac{1}{1+e^{-z}}
$$

**Likelihood dari seluruh respons yang sudah di-observasi** [1, Eq.2–3, p.276]:

$$
\hat{\boldsymbol\theta} \equiv \arg\max_{\boldsymbol\theta} f(\mathbf{u}\mid\boldsymbol\theta), \qquad
f(\mathbf{u}\mid\boldsymbol\theta) = \prod_{i=1}^{n} P_i(\boldsymbol\theta)^{u_i}\,Q_i(\boldsymbol\theta)^{1-u_i}
$$

dengan $u_i\in\{0,1\}$ respons examinee pada item $i$ yang sudah di-administer, dan
$Q_i(\boldsymbol\theta)=1-P_i(\boldsymbol\theta)$. Mulder & van der Linden menyatakan langsung
setelah Eq.3 [1, p.276]: *"The MLE can [be] found by setting the derivative of the logarithm of
(3) equal to zero and solv[ing] the system for $\theta$ using a numerical method such as
Newton–Raphson (e.g., Segall, 1996) or an EM algorithm."* - namun paper tidak menuliskan bentuk
eksplisit turunannya. Turunan berikut dibuktikan sendiri secara aljabar (bukan dikutip), lalu
diverifikasi identik dengan kode produksi.

---

### Bagaimana Tiga Metode Berbeda Menggunakan Prior dan Likelihood

| Metode | Filosofi | Rumus | Gunakan Prior? | Kapan Cocok |
|---|---|---|---|---|
| **MLE** | Maksimalkan likelihood murni | $\hat\theta = \arg\max L(\theta)$ | **Tidak** | Banyak item, prior tidak penting |
| **MAP** | Maksimalkan posterior (mode) | $\hat\theta = \arg\max g(\theta) = L(\theta) \times \pi(\theta)$ | **Ya** | Awal tes (item sedikit), prior bisa menahan divergen |
| **EAP** | Rata-rata posterior | $\hat\theta = E[\theta \mid \mathbf{u}] = \int \theta \cdot g(\theta) d\theta$ | **Ya** | Awal tes, ketika distribusi posterior penting (bukan hanya titik estimasi) |

**Perbedaan intuitif:**

1. **MLE**: *"Cari $\theta$ yang paling menjelaskan data yang ada, tanpa asumsi tentang populasi"* → Estimasi "murni dari data"
   - Risiko: Bisa divergen jika data pattern khusus (semua benar/salah) karena tidak ada penahan dari prior
   
2. **MAP**: *"Cari $\theta$ yang paling menjelaskan data SEKALIGUS konsisten dengan prior populasi"* → Estimasi "data + prior pengetahuan"
   - Keuntungan: Prior bertindak sebagai "penalti" yang mencegah divergen
   - Risiko: Bisa over-shrink ke mean prior jika prior terlalu kuat
   
3. **EAP**: *"Hitung rata-rata $\theta$ dari distribusi posterior (bukan hanya modus)"* → Estimasi "rerata yang realistic"
   - Keuntungan: Selalu finite (integral atas domain terbatas), stabil
   - Risiko: Rata-rata bisa berbeda dari mode jika posterior skewed
   - Bonus: SE otomatis dihitung (variance posterior)

Mulder & van der Linden [1, p.276-277] merekomendasikan **MAP untuk round awal CAT** (saat item sedikit & divergen risk tinggi) dan **EAP untuk keseimbangan** antara stabilitas & efisiensi.

---



### 2.1 Pembuktian: Skor (gradien log-likelihood)

$$
\log f(\mathbf{u}\mid\boldsymbol\theta) = \sum_{i=1}^n \Big[u_i\log P_i(\boldsymbol\theta) + (1-u_i)\log Q_i(\boldsymbol\theta)\Big]
$$

Karena $\partial Q_i/\partial\boldsymbol\theta = -\partial P_i/\partial\boldsymbol\theta$:

$$
\frac{\partial \log f}{\partial\boldsymbol\theta} = \sum_i \left[\frac{u_i}{P_i} - \frac{1-u_i}{Q_i}\right]\frac{\partial P_i}{\partial\boldsymbol\theta}
= \sum_i \frac{u_iQ_i - (1-u_i)P_i}{P_iQ_i}\,\frac{\partial P_i}{\partial\boldsymbol\theta}
$$

Aljabar pembilang: $u_iQ_i-(1-u_i)P_i = u_i(1-P_i)-P_i+u_iP_i = u_i-P_i$. Dan dengan aturan rantai
pada $P_i(\boldsymbol\theta)=c_i+(1-c_i)\sigma(z_i)$, $z_i=\mathbf{a}_i\cdot\boldsymbol\theta+d_i$:

$$
\frac{\partial P_i}{\partial\boldsymbol\theta} = (1-c_i)\,\sigma(z_i)\big(1-\sigma(z_i)\big)\,\mathbf{a}_i = P'_i\,\mathbf{a}_i
$$

($P'_i$ = notasi yang sama seperti [item_selection_summary #0](/posts/cat/item-selection-criteria-mcat#model--notasi-dasar-dipakai-oleh-ketiga-metode)). Maka:

$$
\boxed{\nabla\log f(\boldsymbol\theta) = \sum_i \mathbf{a}_i\,\frac{(u_i-P_i)\,P'_i}{P_iQ_i}}
$$

Ini **identik** dengan `mle.rs:32-33`: `residual=(x-p)*p_prime/(p*q); grad += a*residual`.

### 2.2 Fisher Information Matrix sebagai pengganti Hessian (Fisher scoring)

Hessian eksak (turunan kedua $\log f$) melibatkan turunan kedua $P'_i$ yang rumit. Praktik standar
-dipakai baik oleh Baker (2001, lihat [#3.1](#31-teori)) maupun Mulder & van der Linden-adalah
mengganti Hessian dengan negatif ekspektasinya, yaitu **Fisher Information Matrix** [1, Eq.4,
p.276]:

$$
\mathbf{I}_i(\boldsymbol\theta) \equiv -E\!\left[\frac{\partial^2}{\partial\boldsymbol\theta\partial\boldsymbol\theta^\top}\log f(U_i\mid\boldsymbol\theta)\right]
= \frac{Q_i(\theta)\big[P_i(\theta)-c_i\big]^2}{P_i(\theta)(1-c_i)^2}\,\mathbf{a}_i\mathbf{a}_i^\top
$$

Substitusi ini disebut **Fisher scoring** (metode skor). Pembuktian bahwa formula ini identik
dengan `w=(P')²/(PQ)` yang dipakai `mle.rs`/`map.rs`/`mirt::item_fim`: substitusikan
$P^*=(P_i-c_i)/(1-c_i)$ (invers dari $P_i=c_i+(1-c_i)P^*$), maka $1-P^*=(1-P_i)/(1-c_i)=Q_i/(1-c_i)$,
sehingga

$$
P'_i = (1-c_i)P^*(1-P^*) = (1-c_i)\cdot\frac{P_i-c_i}{1-c_i}\cdot\frac{Q_i}{1-c_i} = \frac{(P_i-c_i)Q_i}{1-c_i}
$$

$$
w_i = \frac{(P'_i)^2}{P_iQ_i} = \frac{(P_i-c_i)^2Q_i^2/(1-c_i)^2}{P_iQ_i} = \frac{Q_i(P_i-c_i)^2}{P_i(1-c_i)^2} \quad\blacksquare
$$

- sama persis dengan $\mathbf{I}_i(\boldsymbol\theta)$ di atas. Untuk M2PL ($c_i=0$):
$w_i=P_i(1-P_i)$, dan $\mathbf{I}_i(\boldsymbol\theta)=P_i(1-P_i)\,\mathbf{a}_i\mathbf{a}_i^\top$.
FIM total teradditif atas item yang sudah dijawab [1, Eq.6, p.277]:
$\mathbf{I}_S(\boldsymbol\theta)=\sum_{i\in S}\mathbf{I}_i(\boldsymbol\theta)$, dan estimator ini
terdistribusi asimtotik normal [1, Eq.7, p.277]:
$\hat{\boldsymbol\theta}\sim N\big(\theta_0,\mathbf{I}_S^{-1}(\theta_0)\big)$ - generalisasi
multivariat dari batas bawah Cramér–Rao.

**Keterangan variabel (tambahan untuk estimasi):**

| Simbol | Arti |
|---|---|
| $\mathbf{u}=(u_1,\dots,u_n)$ | Vektor respons examinee pada $n$ item yang sudah di-administer |
| $f(\mathbf{u}\mid\boldsymbol\theta)$ | Fungsi likelihood - peluang bersama seluruh respons pada $\boldsymbol\theta$ |
| $\nabla\log f(\boldsymbol\theta)$ | Skor (*score function*) - gradien log-likelihood, $=0$ pada MLE |
| $\mathbf{I}_i(\boldsymbol\theta)$, $\mathbf{I}_S(\boldsymbol\theta)$ | FIM item $i$ / FIM kumulatif himpunan item $S$ (identik dengan $I_i(\theta)$ di item_selection notes) |
| $P^*$ | $\sigma(z_i)$, bagian sigmoid murni tanpa guessing (dipakai pada pembuktian $w_i$) |

---

### 2.3 Metode Numerik: Iterasi dan Konvergensi

Ketiga metode estimasi (MLE, MAP, EAP) menggunakan **algoritma numerik iteratif** untuk menemukan estimasi kemampuan $\hat{\boldsymbol\theta}$. Bagian ini menjelaskan konsep umum yang berlaku di semua metode.

#### Apa itu Iterasi?

Iterasi adalah **proses berulang menebak dan menyempurnakan** untuk mencari jawaban yang tepat. Berbeda dengan rumus sederhana yang langsung memberi hasil, algoritma numerik bekerja seperti ini:

**Analogi Konkret: Menyetel Radio**

```
Tebakan awal (θ₀):  Tombol di posisi random → Suara berisik ❌
     ↓ (Dengarkan dan perbaiki posisi)
Tebakan 1 (θ₁):     Tombol digeser → Suara mulai jernih
     ↓ (Dengarkan dan perbaiki lagi)
Tebakan 2 (θ₂):     Tombol digeser lagi → Suara lebih jernih
     ↓
Tebakan 3 (θ₃):     Tombol fine-tuning → Suara jernih sempurna
     ↓
Tebakan 4 (θ₄):     Tombol tidak perlu digeser lagi ✓ SELESAI
```

Dalam notasi matematika, setiap iterasi mengikuti pola:

$$
\hat{\boldsymbol\theta}_{s+1} = \hat{\boldsymbol\theta}_s + \Delta\boldsymbol\theta_s
$$

di mana:
- $s$ = nomor iterasi (0, 1, 2, 3, ...)
- $\hat{\boldsymbol\theta}_s$ = estimasi pada iterasi ke-$s$
- $\Delta\boldsymbol\theta_s$ = perubahan parameter (step size) pada iterasi $s$

#### Apa itu Konvergensi?

**Konvergensi** adalah kondisi ketika perubahan parameter menjadi **sangat sangat kecil** sehingga iterasi bisa dihentikan. Kriteria konvergensi formal yang dipakai di semua metode:

$$
\left\|\hat{\boldsymbol\theta}_{s+1} - \hat{\boldsymbol\theta}_s\right\| < 10^{-6} \quad\Rightarrow\quad \text{KONVERGEN - Iterasi Berhenti}
$$

Artinya: **Jika perubahan norm kurang dari 0.000001, maka nilai $\hat{\boldsymbol\theta}$ sudah stabil dan siap digunakan sebagai estimasi final.**

#### Contoh Numerik: Pola Konvergensi

| Iterasi | $\hat{\boldsymbol\theta}$ | Perubahan ($\|\Delta\boldsymbol\theta\|$) | Status |
|---|---|---|---|
| 0 | 1.0000 | - | Tebakan awal (arbitrary) |
| 1 | 0.2267 | 0.7733 | Perubahan **BESAR** ← masih jauh dari optimal |
| 2 | 0.3239 | 0.0973 | Perubahan lebih kecil ← semakin dekat |
| 3 | 0.3248 | 0.0009 | Perubahan sangat kecil |
| 4 | 0.3248 | 0.0000001 | < 10⁻⁶ ✓ **KONVERGEN!** |

**Interpretasi:**
- **Awal** (Iterasi 1-2): Perubahan besar, algoritma masih "mencari arah"
- **Tengah** (Iterasi 3): Perubahan kecil, algoritma sudah dekat ke jawaban
- **Akhir** (Iterasi 4): Perubahan sangat kecil (< 10⁻⁶), **berhenti dan gunakan θ = 0.3248 sebagai hasil final**

#### Mengapa Perlu Konvergensi?

1. **Sebelum konvergen**: Nilai $\hat{\boldsymbol\theta}$ masih berubah-ubah, belum stabil
2. **Setelah konvergen**: Nilai $\hat{\boldsymbol\theta}$ sudah stabil, aman digunakan sebagai estimasi kemampuan final

#### Implementasi Praktis

Dalam kode produksi (`mle.rs`, `map.rs`, dll), konvergensi diimplementasikan sebagai:

```rust
// Setiap iterasi:
if (theta_next - theta_current).norm() < 1e-6 {
    break;  // ← Keluar loop, konvergen ditemukan
}
// Batasi juga iterasi maksimal (misal 100) untuk menghindari infinite loop
```

**Catatan**: Ketiga metode (MLE, MAP, EAP) menggunakan kriteria konvergensi yang sama, meskipun cara menghitung $\Delta\boldsymbol\theta$ berbeda:
- **MLE**: Update berbasis likelihood saja
- **MAP**: Update berbasis likelihood + prior penalty
- **EAP**: Biasanya tidak iteratif (langsung integrasi numerik)

---

## 3. Maximum Likelihood Estimation (MLE)

### 3.1 Teori

MLE mencari $\hat{\boldsymbol\theta}$ yang memaksimalkan $f(\mathbf{u}\mid\boldsymbol\theta)$
[1, Eq.2, p.276] - lihat [#2](#2-model-dan-notasi-dasar-dipakai-oleh-ketiga-metode) untuk definisi
lengkap dan pembuktian skor/FIM. Iterasi Newton–Raphson (Fisher scoring) univariat, dibuktikan
identik dengan kode produksi, pertama kali dituliskan eksplisit dengan angka oleh Baker (2001),
*The Basics of Item Response Theory* (2nd ed.), Bab 5 "Estimating an Examinee's Ability",
**Eq.[5-1], p.86** [2]:

$$
\hat\theta_{s+1} = \hat\theta_s + \frac{\displaystyle\sum_{i=1}^N a_i\big[u_i-P_i(\hat\theta_s)\big]}{\displaystyle\sum_{i=1}^N a_i^2\,P_i(\hat\theta_s)\,Q_i(\hat\theta_s)}
\tag{5-1}
$$

Untuk M2PL univariat ($c=0$, $k=1$), $\nabla\log f=\sum a_i(u_i-P_i)$ ([#4.1](#21-pembuktian-skor-gradien-log-likelihood)) dan
$\mathbf{I}_S=\sum a_i^2P_iQ_i$ ([#4.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring)) - Eq.[5-1] Baker **adalah** Fisher scoring
$\hat\theta_{s+1}=\hat\theta_s+\mathbf{I}_S^{-1}\nabla\log f$ pada kasus 1-dimensi, dituliskan
dengan notasi $(a,b,c)$ alih-alih $(a,d,c)$ (lihat [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1) untuk konversi $d=-ab$).

Generalisasi ke $k>1$ dimensi mengganti pembagian skalar dengan perkalian
matriks invers:

$$
\hat{\boldsymbol\theta}_{s+1} = \hat{\boldsymbol\theta}_s + \mathbf{I}_S(\hat{\boldsymbol\theta}_s)^{-1}\,\nabla\log f(\hat{\boldsymbol\theta}_s), \qquad \left\|\hat{\boldsymbol\theta}_{s+1}-\hat{\boldsymbol\theta}_s\right\| < 10^{-6} \Rightarrow \text{stop}
$$

Mulder & van der Linden mencatat langsung setelah definisi
MLE [1, p.276]: *"The likelihood function may not have a maximum (e.g., when only correct or
incorrect item responses are observed), or a local instead of a global maximum may be found."* -
dibuktikan ulang secara eksperimental di [#5.2.3](#323-demo-3-kasus-divergen-all-correct).

**Keterangan variabel:**

| Simbol | Arti |
|---|---|
| $\hat\theta_s$ | Estimasi kemampuan pada iterasi ke-$s$ |
| $a_i$ (Baker) $\equiv \mathbf{a}_i$ (notasi vektor) | Parameter diskriminasi item $i$ |
| $N$ | Jumlah item yang sudah di-administer |
| $\mathbf{I}_S(\boldsymbol\theta)^{-1}$ | Invers FIM kumulatif - berperan sebagai "step size" matriks pada Newton step |

#### 3.1.1 Teori: Standard Error (SE) untuk MLE

SE bukan formula terpisah - ia jatuh langsung dari sifat asimtotik normal yang sudah dibuktikan di
[#2.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring) [1, Eq.7, p.277]:
$\hat{\boldsymbol\theta}\sim N(\theta_0,\mathbf{I}_S^{-1}(\theta_0))$. Jika $\mathbf{I}_S^{-1}(\theta_0)$
adalah matriks kovarians (asimtotik) dari estimator, maka **variance per-dimensi adalah elemen
diagonalnya**, dan SE adalah akarnya:

$$
\boxed{SE(\hat\theta_j) = \sqrt{\left[\mathbf{I}_S(\hat{\boldsymbol\theta})^{-1}\right]_{jj}}}, \qquad j=1,\dots,k
$$

dengan $\mathbf{I}_S(\hat{\boldsymbol\theta})$ dievaluasi pada estimasi **final** (konvergen), bukan
pada titik awal. Untuk kasus univariat ($k=1$), ini tereduksi ke bentuk skalar yang memakai penyebut
yang sama persis dengan Newton step Eq.[5-1] Baker [2, p.86] (lihat [#3.1](#31-teori)):

$$
SE(\hat\theta) = \frac{1}{\sqrt{\displaystyle\sum_{i=1}^N a_i^2\,P_i(\hat\theta)\,Q_i(\hat\theta)}}
$$

**Identik dengan kode produksi:** `se_vector()` (`mirt.rs:62-71`) menghitung
$\sqrt{\text{diag}(\mathbf{I}_S^{-1})}$ persis seperti rumus di atas, dipanggil lewat
`McatEngine::compute_se` (`engine.rs:159`): `EstimationMethod::Mle => se_vector(cum_fim, k)`.
`cum_fim` sendiri adalah $\mathbf{I}_S(\hat{\boldsymbol\theta})$ - dijumlahkan dari `item_fim()`
(`mirt.rs:26-36`, definisi identik [#2.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring))
atas seluruh item yang sudah dijawab, dievaluasi pada $\hat{\boldsymbol\theta}$ hasil konvergensi -
**bukan** dihitung di dalam `mle.rs` itu sendiri (lihat komentar `mle.rs:34-36`: *"MLE's SE is derived
afterward from the cumulative FIM (McatEngine::compute_se), not produced here"*), melainkan satu
langkah terpisah setelah loop Newton-Raphson berhenti.

**Interpretasi:** SE mengukur presisi estimasi $\hat\theta$ - semakin banyak item terjawab (atau
semakin diskriminatif item-nya, $a_i$ besar), semakin besar $\mathbf{I}_S$, semakin kecil
$\mathbf{I}_S^{-1}$, dan semakin kecil SE (estimasi makin presisi). Karena MLE tidak punya suku prior
penambah informasi, SE-nya selalu $\geq$ SE MAP pada data identik - dibuktikan di
[#4.1.2](#412-teori-standard-error-se-untuk-map): $\mathbf{H}_{MAP}\succeq\mathbf{I}_S \Rightarrow
\mathbf{H}_{MAP}^{-1}\preceq\mathbf{I}_S^{-1}$.

### 3.2 Perhitungan Manual

#### 3.2.1 DEMO 1: Reproduksi Baker (2001), k=1

Item Baker (2001, p.87) dalam parameterisasi $(a,b,c)$, dikonversi ke $(a,d,c)$ produksi via
$d=-ab$ (karena $z=a\theta+d=a(\theta-b)$):

| Item | $a$ | $b$ (Baker) | $d=-ab$ | $c$ | $u$ |
|---|---|---|---|---|---|
| 1 | 1.0 | $-1$ | $+1.0$ | 0 | 1 |
| 2 | 1.2 | $0$ | $0.0$ | 0 | 0 |
| 3 | 0.8 | $1$ | $-0.8$ | 0 | 1 |

*A priori* $\hat\theta_0=1.0$ - dikutip langsung, Baker (2001, p.87): *"Initially, the $\hat\theta_s$
on the right side of the equal sign is set to some arbitrary value, such as 1."*

**Iterasi 1** (M2PL sehingga $P=P^*=\sigma(a\theta+d)$):

**Step 1: Hitung $z_i = a\theta + d$ untuk setiap item** dengan $\hat\theta_0=1.0$:

| Item | $a$ | $d$ | $z_i = a(1.0) + d$ |
|---|---|---|---|
| 1 | 1.0 | +1.0 | $1.0(1.0) + 1.0 = 2.0$ |
| 2 | 1.2 | 0.0 | $1.2(1.0) + 0.0 = 1.2$ |
| 3 | 0.8 | -0.8 | $0.8(1.0) - 0.8 = 0.0$ |

**Step 2: Hitung $P_i = \sigma(z_i) = \frac{1}{1+e^{-z_i}}$ dan $Q_i=1-P_i$:**

| Item | $z_i$ | $P_i = \sigma(z_i)$ | $Q_i$ |
|---|---|---|---|
| 1 | 2.0 | $\frac{1}{1+e^{-2.0}} = \frac{1}{1+0.1353} = 0.8808$ | $1-0.8808=0.1192$ |
| 2 | 1.2 | $\frac{1}{1+e^{-1.2}} = \frac{1}{1+0.3012} = 0.7685$ | $1-0.7685=0.2315$ |
| 3 | 0.0 | $\frac{1}{1+e^{0}} = \frac{1}{2} = 0.5000$ | $1-0.5000=0.5000$ |

**Step 3: Hitung residual & weight menggunakan Eq.[5-1] Baker untuk setiap item:**

Residual: $\text{resid}_i = a_i(u_i - P_i)$ dan weight: $\text{wt}_i = a_i^2 P_i Q_i$

| Item | $u$ | $a_i(u_i-P_i)$ | $a_i^2P_iQ_i$ | Perhitungan |
|---|---|---|---|---|
| 1 | 1 | $1.0(1-0.8808) = +0.1192$ | $(1.0)^2(0.8808)(0.1192) = 0.1050$ | $1.0 \times 0.1192 = 0.1192$ |
| 2 | 0 | $1.2(0-0.7685) = -0.9222$ | $(1.2)^2(0.7685)(0.2315) = 0.2562$ | $1.44 \times 0.1779 = 0.2562$ |
| 3 | 1 | $0.8(1-0.5000) = +0.4000$ | $(0.8)^2(0.5000)(0.5000) = 0.1600$ | $0.64 \times 0.2500 = 0.1600$ |
| **sum** | | **-0.4030** | **0.5212** | |

**Step 4: Hitung update parameter Newton-Raphson:**

$$
\Delta\hat\theta = \frac{\sum a_i(u_i-P_i)}{\sum a_i^2P_iQ_i} = \frac{-0.4030}{0.5212} = -0.7733
$$

$$
\hat\theta_1 = \hat\theta_0 + \Delta\hat\theta = 1.0 + (-0.7733) = 0.2267
$$

**Iterasi 2** dengan $\hat\theta_1 = 0.2267$:

**Step 1: Hitung $z_i$ untuk setiap item:**

| Item | $a$ | $d$ | $z_i = a(0.2267) + d$ |
|---|---|---|---|
| 1 | 1.0 | +1.0 | $1.0(0.2267) + 1.0 = 1.2267$ |
| 2 | 1.2 | 0.0 | $1.2(0.2267) + 0.0 = 0.2720$ |
| 3 | 0.8 | -0.8 | $0.8(0.2267) - 0.8 = -0.6186$ |

**Step 2: Hitung $P_i$ dan $Q_i$:**

| Item | $z_i$ | $P_i$ | $Q_i$ |
|---|---|---|---|
| 1 | 1.2267 | $\frac{1}{1+e^{-1.2267}} = 0.7732$ | 0.2268 |
| 2 | 0.2720 | $\frac{1}{1+e^{-0.2720}} = 0.5676$ | 0.4324 |
| 3 | -0.6186 | $\frac{1}{1+e^{0.6186}} = 0.3501$ | 0.6499 |

**Step 3: Hitung residual & weight:**

| Item | $u$ | $a_i(u_i-P_i)$ | $a_i^2P_iQ_i$ |
|---|---|---|---|
| 1 | 1 | $1.0(1-0.7732) = +0.2268$ | $(1.0)^2(0.7732)(0.2268) = 0.1753$ |
| 2 | 0 | $1.2(0-0.5676) = -0.6811$ | $(1.2)^2(0.5676)(0.4324) = 0.3534$ |
| 3 | 1 | $0.8(1-0.3501) = +0.5199$ | $(0.8)^2(0.3501)(0.6499) = 0.1456$ |
| **sum** | | **+0.0656** | **0.6744** |

**Step 4: Hitung update:**

$$
\Delta\hat\theta = \frac{0.0656}{0.6744} = +0.0973
$$

$$
\hat\theta_2 = 0.2267 + 0.0973 = 0.3239
$$

**Cross-check langsung terhadap buku** (Baker 2001, p.88, angka asli):

$$
\Delta\hat\theta_s = -.403/.520 = -.773,\;\; \hat\theta_{s+1}=1.0-.773=0.227 \qquad
\Delta\hat\theta_s = .066/.674 = .097,\;\; \hat\theta_{s+1}=0.227+.097=0.324
$$

**cocok dengan buku sampai 3 desimal**
($0.227$, $0.324$; selisih dari pembulatan tampilan 3-desimal Baker vs 4-desimal disini).

**Iterasi 3** dengan $\hat\theta_2=0.323937$:

$$
\Delta\hat\theta = \frac{0.0006}{0.6616} = +0.0009, \qquad \hat\theta_3 = \hat\theta_2+\Delta\hat\theta = 0.324846
$$

**Verifikasi konvergensi** (kriteria [#2.3](#23-metode-numerik-iterasi-dan-konvergensi): $|\hat\theta_{s+1}-\hat\theta_s|<10^{-6}$):
$$|\hat\theta_3-\hat\theta_2| = |\Delta\hat\theta| = 9.10\times10^{-4} \gg 10^{-6} \Rightarrow \textbf{BELUM KONVERGEN}, \text{ lanjut iterasi.}$$

**Iterasi 4** dengan $\hat\theta_3=0.324846$:

$$
\Delta\hat\theta = \frac{0.0000}{0.6614} \approx +9.05\times10^{-8}, \qquad \hat\theta_4 = \hat\theta_3+\Delta\hat\theta = 0.324846
$$

**Verifikasi konvergensi:**
$$|\hat\theta_4-\hat\theta_3| = |\Delta\hat\theta| = 9.05\times10^{-8} < 10^{-6} \Rightarrow \textbf{KONVERGEN} \text{ - iterasi berhenti.}$$

→ **$\hat\theta_{MLE}\approx 0.324846$** (presisi penuh: $0.3248462760...$).

**Step 5: Hitung Standard Error (SE)** (teori [#3.1.1](#311-teori-standard-error-se-untuk-mle))

Evaluasi FIM pada estimasi **final** $\hat\theta_{MLE}=0.3248462760$ (bukan pada $\hat\theta_0$):

| Item | $z_i=a\hat\theta+d$ | $P_i$ | $Q_i$ | $a_i^2P_iQ_i$ |
|---|---|---|---|---|
| 1 | $1.0(0.324846)+1.0=1.324846$ | 0.789987 | 0.210013 | $(1.0)^2(0.789987)(0.210013)=0.165908$ |
| 2 | $1.2(0.324846)+0.0=0.389816$ | 0.596238 | 0.403762 | $(1.2)^2(0.596238)(0.403762)=0.346663$ |
| 3 | $0.8(0.324846)-0.8=-0.540123$ | 0.368159 | 0.631841 | $(0.8)^2(0.368159)(0.631841)=0.148875$ |
| **sum** | | | | $\mathbf{I}_S(\hat\theta)=0.661446$ |

$$
SE(\hat\theta_{MLE}) = \frac{1}{\sqrt{\mathbf{I}_S(\hat\theta)}} = \frac{1}{\sqrt{0.661446}} \approx 1.229569
$$

(Bukan kebetulan bahwa $\mathbf{I}_S(\hat\theta)=0.661446$ nyaris identik dengan Hessian di
denominator Iterasi 4 ($0.6614$, lihat langkah update terakhir) - Newton-Raphson berhenti persis saat
$\hat\theta$ konvergen, sehingga Hessian pada iterasi konvergensi terakhir memang dievaluasi pada
titik yang sama dengan $\hat\theta_{MLE}$ final.)

#### 3.2.2 DEMO 2: Multidimensional (k=3), 7 item

Menggunakan seluruh 7-item bank yang sama seperti
[Item Bank Snapshot](/posts/cat/item-selection-criteria-mcat#item-bank-snapshot), dengan pola
respons **campuran** (bukan seragam per content-area - lihat catatan Divergen di
[#3.2.3](#323-demo-3-kasus-divergen-all-correct)):

| Item | $\mathbf{a}$ | $d$ | $u$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | 1 |
| m2p-v002 | [1.7,0.2,0.2] | 0.10 | 0 |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | 0 |
| m2p-n002 | [0.3,1.8,0.4] | 0.50 | 1 |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | 1 |
| m2p-r002 | [0.3,0.8,1.9] | 0.60 | 0 |
| m2p-r003 | [0.4,0.3,1.8] | 0.70 | 1 |

Starting $\hat{\boldsymbol\theta}_0=[0,0,0]$.

**Iterasi 1** - $\nabla\log f$ dan $\mathbf{I}_S$ dihitung persis seperti [#2.1](#21-pembuktian-skor-gradien-log-likelihood)/[#2.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring), dijumlahkan atas ke-7 item:

**Step 1: Hitung (linear predictor $z_i$) $z_i = \mathbf{a}_i \cdot \boldsymbol\theta_0 + d_i$ untuk setiap item** dengan $\boldsymbol\theta_0=[0,0,0]$:

| Item | $\mathbf{a}_i$ | $d_i$ | $z_i = [0,0,0] \cdot \mathbf{a}_i + d_i$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | $0 + 0 + 0 + 0.40 = 0.40$ |
| m2p-v002 | [1.7,0.2,0.2] | 0.10 | $0.10$ |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | $0.80$ |
| m2p-n002 | [0.3,1.8,0.4] | 0.50 | $0.50$ |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | $0.30$ |
| m2p-r002 | [0.3,0.8,1.9] | 0.60 | $0.60$ |
| m2p-r003 | [0.4,0.3,1.8] | 0.70 | $0.70$ |

**Step 2: Hitung (peluang peserta menjawab benar $P$) $P_i = \sigma(z_i)$, (peluang peserta menjawab salah $Q$) $Q_i = 1-P_i$, dan (seberapa sensitif peluang benar berubah) $P'_i = P_iQ_i$ (untuk M2PL, $c=0$):**

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i = P_iQ_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.40 | 0.5987 | 0.4013 | 0.2403 |
| m2p-v002 | 0.10 | 0.5250 | 0.4750 | 0.2494 |
| m2p-n001 | 0.80 | 0.6900 | 0.3100 | 0.2139 |
| m2p-n002 | 0.50 | 0.6225 | 0.3775 | 0.2350 |
| m2p-r001 | 0.30 | 0.5744 | 0.4256 | 0.2446 |
| m2p-r002 | 0.60 | 0.6456 | 0.3544 | 0.2290 |
| m2p-r003 | 0.70 | 0.6682 | 0.3318 | 0.2217 |

**Step 3: Hitung residual per item** menggunakan $\text{residual}_i = (u_i - P_i) \cdot P'_i / (P_iQ_i) = (u_i - P_i)$ (untuk M2PL):

Residual ini adalah untuk mengukur seberapa jauh prediksi model meleset dari respons aktual peserta pada item itu, dan menjadi bahan baku untuk membangun gradien (skor) yang menggerakkan update $\hat\theta$.

| Item | $u$ | $(u_i - P_i)$ | Kontribusi ke gradien = $\mathbf{a}_i \times (u_i - P_i)$ |
|---|---|---|---|
| m2p-v001 | 1 | $1 - 0.5987 = 0.4013$ | $[1.9,0.2,0.3] \times 0.4013 = [0.7625, 0.0803, 0.1204]$ |
| m2p-v002 | 0 | $0 - 0.5250 = -0.5250$ | $[1.7,0.2,0.2] \times (-0.5250) = [-0.8925, -0.1050, -0.1050]$ |
| m2p-n001 | 0 | $0 - 0.6900 = -0.6900$ | $[0.3,1.9,0.4] \times (-0.6900) = [-0.2070, -1.3110, -0.2760]$ |
| m2p-n002 | 1 | $1 - 0.6225 = 0.3775$ | $[0.3,1.8,0.4] \times 0.3775 = [0.1133, 0.6795, 0.1510]$ |
| m2p-r001 | 1 | $1 - 0.5744 = 0.4256$ | $[0.5,0.4,2.0] \times 0.4256 = [0.2128, 0.1702, 0.8512]$ |
| m2p-r002 | 0 | $0 - 0.6456 = -0.6456$ | $[0.3,0.8,1.9] \times (-0.6456) = [-0.1937, -0.5165, -1.2266]$ |
| m2p-r003 | 1 | $1 - 0.6682 = 0.3318$ | $[0.4,0.3,1.8] \times 0.3318 = [0.1327, 0.0995, 0.5972]$ |

**Step 4: Agregasi gradien** (jumlah semua kontribusi):

$$
\nabla\log f = [0.7625 - 0.8925 - 0.2070 + 0.1133 + 0.2128 - 0.1937 + 0.1327, \ldots] = [-0.0719, -0.9029, 0.1121]
$$

**Step 5: Hitung FIM per item** menggunakan $\mathbf{I}_i = P'_i \cdot \mathbf{a}_i \mathbf{a}_i^\top$ (untuk M2PL):

Sebagai contoh, untuk item 1:
$$
\mathbf{I}_1 = 0.2403 \times \begin{bmatrix} 1.9 \\ 0.2 \\ 0.3 \end{bmatrix} \begin{bmatrix} 1.9 & 0.2 & 0.3 \end{bmatrix} = 0.2403 \times \begin{bmatrix} 3.61 & 0.38 & 0.57 \\ 0.38 & 0.04 & 0.06 \\ 0.57 & 0.06 & 0.09 \end{bmatrix} = \begin{bmatrix} 0.8679 & 0.0913 & 0.1370 \\ 0.0913 & 0.0096 & 0.0144 \\ 0.1370 & 0.0144 & 0.0216 \end{bmatrix}
$$

(Dilakukan untuk semua 7 item, kemudian dijumlahkan)

**Step 6: Agregasi FIM:**

$$
\mathbf{I}_S = \sum_{i=1}^{7} \mathbf{I}_i = \begin{bmatrix} 1.7456 & 0.5553 & 0.8101 \\ 0.5553 & 1.7587 & 1.0192 \\ 0.8101 & 1.0192 & 2.6255 \end{bmatrix}
$$

**Step 7: Hitung invers FIM dan update parameter:**

$$
\mathbf{I}_S^{-1} = \begin{bmatrix} 0.6882 & -0.1216 & -0.1651 \\ -0.1216 & 0.7551 & -0.2556 \\ -0.1651 & -0.2556 & 0.5311 \end{bmatrix}
$$

$$
\Delta\hat{\boldsymbol\theta} = \mathbf{I}_S^{-1} \nabla\log f = \begin{bmatrix} 0.6882 & -0.1216 & -0.1651 \\ -0.1216 & 0.7551 & -0.2556 \\ -0.1651 & -0.2556 & 0.5311 \end{bmatrix} \begin{bmatrix} -0.0719 \\ -0.9029 \\ 0.1121 \end{bmatrix}
$$

$$
= \begin{bmatrix} -0.0494 + 0.1099 - 0.0185 \\ 0.0087 - 0.6824 - 0.0286 \\ 0.0119 + 0.2315 + 0.0595 \end{bmatrix} = \begin{bmatrix} 0.0418 \\ -0.7017 \\ 0.3022 \end{bmatrix}
$$

$$
\hat{\boldsymbol\theta}_1 = \hat{\boldsymbol\theta}_0 + \Delta\hat{\boldsymbol\theta} = [0,0,0] + [0.0418,\,-0.7017,\,0.3022] = [0.0418,\,-0.7017,\,0.3022]
$$

**Iterasi 2** dengan $\hat{\boldsymbol\theta}_1=[0.0418,\,-0.7017,\,0.3022]$:

**Step 1: Hitung $z_i = \mathbf{a}_i \cdot \hat{\boldsymbol\theta}_1 + d_i$ untuk setiap item:**

| Item | $\mathbf{a}_i \cdot \hat{\boldsymbol\theta}_1$ | $d_i$ | $z_i$ |
|---|---|---|---|
| m2p-v001 | $[1.9,0.2,0.3] \cdot [0.0418,-0.7017,0.3022] = 0.0794 - 0.1403 + 0.0907$ | +0.40 | 0.4298 |
| m2p-v002 | $[1.7,0.2,0.2] \cdot [0.0418,-0.7017,0.3022] = 0.0711 - 0.1403 + 0.0604$ | +0.10 | -0.0088 |
| m2p-n001 | $[0.3,1.9,0.4] \cdot [0.0418,-0.7017,0.3022] = 0.0125 - 1.3332 + 0.1209$ | +0.80 | 0.0002 |
| m2p-n002 | $[0.3,1.8,0.4] \cdot [0.0418,-0.7017,0.3022] = 0.0125 - 1.2631 + 0.1209$ | +0.50 | -0.6297 |
| m2p-r001 | $[0.5,0.4,2.0] \cdot [0.0418,-0.7017,0.3022] = 0.0209 - 0.2807 + 0.6044$ | +0.30 | 0.7446 |
| m2p-r002 | $[0.3,0.8,1.9] \cdot [0.0418,-0.7017,0.3022] = 0.0125 - 0.5614 + 0.5742$ | +0.60 | 0.6253 |
| m2p-r003 | $[0.4,0.3,1.8] \cdot [0.0418,-0.7017,0.3022] = 0.0167 - 0.2105 + 0.5440$ | +0.70 | 1.0502 |

**Step 2: Hitung $P_i$ dan $Q_i$ berdasarkan $z_i$ baru:**

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.4298 | 0.6057 | 0.3943 | 0.2388 |
| m2p-v002 | -0.0088 | 0.4978 | 0.5022 | 0.2500 |
| m2p-n001 | 0.0002 | 0.5000 | 0.5000 | 0.2500 |
| m2p-n002 | -0.6297 | 0.3476 | 0.6524 | 0.2268 |
| m2p-r001 | 0.7446 | 0.6781 | 0.3219 | 0.2184 |
| m2p-r002 | 0.6253 | 0.6517 | 0.3483 | 0.2272 |
| m2p-r003 | 1.0502 | 0.7408 | 0.2592 | 0.1922 |

**Step 3: Hitung kontribusi ke gradien per item** $\text{kontribusi}_i = \mathbf{a}_i \times (u_i - P_i)$:

| Item | $u$ | $(u_i - P_i)$ | Kontribusi |
|---|---|---|---|
| m2p-v001 | 1 | 0.3943 | [0.7492, 0.0789, 0.1183] |
| m2p-v002 | 0 | -0.4978 | [-0.8463, -0.0996, -0.0996] |
| m2p-n001 | 0 | -0.5000 | [-0.1500, -0.9500, -0.2000] |
| m2p-n002 | 1 | 0.6524 | [0.1957, 1.1743, 0.2610] |
| m2p-r001 | 1 | 0.3219 | [0.1610, 0.1288, 0.6438] |
| m2p-r002 | 0 | -0.6517 | [-0.1955, -0.5214, -1.2382] |
| m2p-r003 | 1 | 0.2592 | [0.1037, 0.0778, 0.4666] |

Agregasi: $\nabla\log f \approx [0.0159,\,0.0803,\,0.0314]$ (nilai semakin mendekati nol)

**Step 4: Hitung FIM baru dan update:**

$$
\mathbf{I}_S^{\text{(iter 2)}} = \begin{bmatrix} 1.7327 & 0.5577 & 0.7704 \\ 0.5577 & 1.8204 & 0.9996 \\ 0.7704 & 0.9996 & 2.4510 \end{bmatrix}
$$

$$
\Delta\hat{\boldsymbol\theta} = \mathbf{I}_S^{-1} \nabla\log f = [-0.0039,\,0.0485,\,-0.0057]
$$

$$
\hat{\boldsymbol\theta}_2 = [0.0418,\,-0.7017,\,0.3022] + [-0.0039,\,0.0485,\,-0.0057] = [0.0379,\,-0.6532,\,0.2964]
$$

**Iterasi 3** dengan $\hat{\boldsymbol\theta}_2=[0.0379,\,-0.6532,\,0.2964]$:

Setelah perhitungan serupa, hitung step size untuk next iteration:

$$
\Delta\hat{\boldsymbol\theta} = [2.67 \times 10^{-5}, -4.81 \times 10^{-4}, 1.43 \times 10^{-4}]
$$

$$
\hat{\boldsymbol\theta}_3 = \hat{\boldsymbol\theta}_2 + \Delta\hat{\boldsymbol\theta} = [0.037969,\,-0.653704,\,0.296573]
$$

**Verifikasi konvergensi** iterasi 3, sesuai kriteria [#2.3](#23-metode-numerik-iterasi-dan-konvergensi)/[#3.1](#31-teori) $\|\hat{\boldsymbol\theta}_{s+1}-\hat{\boldsymbol\theta}_s\|<10^{-6}$ - dan karena $\Delta\hat{\boldsymbol\theta}=\hat{\boldsymbol\theta}_{s+1}-\hat{\boldsymbol\theta}_s$ persis oleh konstruksi Newton step, kedua notasi ini nilainya identik:

$$
\hat{\boldsymbol\theta}_3-\hat{\boldsymbol\theta}_2 = [0.037969,\,-0.653704,\,0.296573] - [0.0379,\,-0.6532,\,0.2964] 
$$
$$
= \Delta\hat{\boldsymbol\theta} = [2.67 \times 10^{-5},\,-4.81 \times 10^{-4},\,1.43 \times 10^{-4}]
$$

$$\left\|\hat{\boldsymbol\theta}_3-\hat{\boldsymbol\theta}_2\right\| = \|\Delta\hat{\boldsymbol\theta}\|_2 = \sqrt{(2.67 \times 10^{-5})^2 + (-4.81 \times 10^{-4})^2 + (1.43 \times 10^{-4})^2} \approx 5.02 \times 10^{-4}$$

Hasil: $5.02 \times 10^{-4} \gg 10^{-6}$ → **BELUM KONVERGEN**, lanjut ke iterasi 4.

**Iterasi 4** dengan $\hat{\boldsymbol\theta}_3=[0.037969,\,-0.653704,\,0.296573]$:

Di sekitar akar persamaan skor, Newton-Raphson konvergen kuadratik - gradien dan FIM sudah sangat dekat dengan titik optimum, sehingga step berikutnya mengecil drastis dibanding iterasi 3:

$$
\Delta\hat{\boldsymbol\theta} = [2.46 \times 10^{-9}, -3.96 \times 10^{-8}, 1.12 \times 10^{-8}]
$$

$$
\hat{\boldsymbol\theta}_4 = \hat{\boldsymbol\theta}_3 + \Delta\hat{\boldsymbol\theta} = [0.03796883984503478,\,-0.6537043452916178,\,0.29657317310496684]
$$

**Verifikasi konvergensi** iterasi 4:

$$\left\|\hat{\boldsymbol\theta}_4-\hat{\boldsymbol\theta}_3\right\| = \|\Delta\hat{\boldsymbol\theta}\|_2 \approx 4.13 \times 10^{-8}$$

Hasil: $4.13 \times 10^{-8} < 10^{-6}$ → **KONVERGEN**. Iterasi berhenti pada iterasi ke-4:

$$
\hat{\boldsymbol\theta}_{MLE} = \hat{\boldsymbol\theta}_4 = [0.03796883984503478,\,-0.6537043452916178,\,0.29657317310496684]
$$

**Step 8: Hitung Standard Error (SE)** (teori [#3.1.1](#311-teori-standard-error-se-untuk-mle))

FIM dievaluasi pada estimasi **final** $\hat{\boldsymbol\theta}_{MLE}=[0.037969,-0.653704,0.296573]$ -
pola perhitungan sama seperti Step 1-6 Iterasi 1 di atas, hanya di titik konvergen:

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i=P_iQ_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.4304 | 0.6060 | 0.3940 | 0.2388 |
| m2p-v002 | 0.0931 | 0.5233 | 0.4767 | 0.2495 |
| m2p-n001 | -0.3120 | 0.4226 | 0.5774 | 0.2440 |
| m2p-n002 | -0.5466 | 0.3666 | 0.6334 | 0.2322 |
| m2p-r001 | 0.6506 | 0.6572 | 0.3428 | 0.2253 |
| m2p-r002 | 0.6519 | 0.6574 | 0.3426 | 0.2252 |
| m2p-r003 | 1.0529 | 0.7413 | 0.2587 | 0.1918 |

Menjumlahkan $\mathbf{I}_i=P'_i\,\mathbf{a}_i\mathbf{a}_i^\top$ atas ke-7 item (pola identik Step 5-6
Iterasi 1):

$$
\mathbf{I}_S(\hat{\boldsymbol\theta}_{MLE}) \approx \begin{bmatrix} 1.7330 & 0.5622 & 0.7698 \\ 0.5622 & 1.8502 & 1.0031 \\ 0.7698 & 1.0031 & 2.4432 \end{bmatrix}
$$

$$
\mathbf{I}_S^{-1} \approx \begin{bmatrix} 0.6904 & -0.1181 & -0.1690 \\ -0.1181 & 0.7154 & -0.2565 \\ -0.1690 & -0.2565 & 0.5679 \end{bmatrix}
$$

$$
SE(\hat{\boldsymbol\theta}_{MLE}) = \sqrt{\text{diag}(\mathbf{I}_S^{-1})} \approx [0.8309,\; 0.8458,\; 0.7536]
$$

**Interpretasi:** dimensi **reasoning** ($SE\approx0.754$) paling presisi diestimasi - konsisten
dengan item bank yang punya 3 item dominan-reasoning ($a_3\in\{2.0,1.9,1.8\}$) vs hanya sebaran lebih
kecil untuk verbal/numeric, sehingga informasi (FIM) terkumpul lebih banyak di dimensi reasoning.

#### 3.2.3 DEMO 3: Kasus Divergen (all-correct)

Menggunakan 3-item subset ($\texttt{m2p-v001}$, $\texttt{m2p-n001}$, $\texttt{m2p-r001}$) dengan
$\mathbf{u}=[1,1,1]$ (seluruhnya benar).

| Item | $\mathbf{a}$ | $d$ | $u$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | **1** |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | **1** |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | **1** |

Starting $\hat{\boldsymbol\theta}_0=[0,0,0]$.

**Iterasi 1** dengan $\boldsymbol\theta_0=[0,0,0]$:

$z$ values sama dengan DEMO 2 Iterasi 1 (karena dimulai dari [0,0,0]):
$z_1=0.40, z_2=0.80, z_3=0.30$ → $P_1=0.5987, P_2=0.6900, P_3=0.5744$

Residual: $(1-P_1)=0.4013, (1-P_2)=0.3100, (1-P_3)=0.4256$ (semua **positif** karena semua benar)

Gradien dan FIM dihitung seperti DEMO 2, tapi hanya 3 item dan semua respons benar:
$$\Delta\hat{\boldsymbol\theta}^{(1)} \approx [0.3245,\, 0.8942,\, 0.6531]$$

$$\hat{\boldsymbol\theta}_1 = [0.3245,\, 0.8942,\, 0.6531]$$

**Iterasi 2** dengan $\hat{\boldsymbol\theta}_1=[0.3245,\, 0.8942,\, 0.6531]$:

Hitung $z_i$ baru dengan norm yang lebih besar:
$$z_1 = [1.9,0.2,0.3] \cdot [0.3245,0.8942,0.6531] + 0.40 = 0.6166 + 0.1789 + 0.1959 + 0.40 = 1.3914$$
$$z_2 = [0.3,1.9,0.4] \cdot [0.3245,0.8942,0.6531] + 0.80 = 0.0974 + 1.6990 + 0.2612 + 0.80 = 2.8576$$
$$z_3 = [0.5,0.4,2.0] \cdot [0.3245,0.8942,0.6531] + 0.30 = 0.1623 + 0.3577 + 1.3062 + 0.30 = 2.1262$$

Kemudian: $P_1=\sigma(1.3914)\approx 0.8015, P_2\approx 0.9459, P_3\approx 0.8966$

Residual masih positif (semua benar): $(1-P_i)>0$ untuk semua item

Karena **semua residual positif** dan **tidak ada respons salah** untuk "menyeimbangkan", gradien terus mendorong $\hat{\boldsymbol\theta}$ ke arah yang memperbesar semua $P_i$ menuju 1.

$$\hat{\boldsymbol\theta}_2 \approx [1.847,\, 2.156,\, 1.843]$$

**Iterasi 3-4 (pola berlanjut):**

Norm terus meningkat: $\|\hat{\boldsymbol\theta}_3\| \approx 5.2$, $\|\hat{\boldsymbol\theta}_4\| \approx 8.9$, dst.

Sebab matematis: dengan $k=3$ item dan $k=3$ dimensi, matriks parameter $\mathbf{A}=[1.9,0.2,0.3; 0.3,1.9,0.4; 0.5,0.4,2.0]$ memiliki rank penuh. Ada arah $\mathbf{v}$ unik (eigenvector dominan dari $\mathbf{A}^\top\mathbf{A}$) sehingga $\mathbf{A}\mathbf{v}>0$ (semua komponen positif). Sepanjang $\boldsymbol\theta = t\mathbf{v}$ dengan $t\to\infty$, **semua** $P_i\to1$ serentak, sehingga likelihood terus naik tanpa mencapai maksimum interior - hanya asimtot pada $P_i=1$ untuk semua item.

Fungsi skor $\nabla\log f$ tidak pernah betul-betul mencapai nol, tapi mendekati nol dari arah positif:
$$\lim_{t\to\infty} \nabla\log f(t\mathbf{v}) = \mathbf{0}^+ \quad\text{(dari komponen positif)}$$

Iterasi berhenti setelah 100 loop dengan:

$$
\hat{\boldsymbol\theta}_{MLE} = [16.065,\; 14.291,\; 11.594], \qquad \|\hat{\boldsymbol\theta}\| = 24.43
$$

(Nilai besar tak-bermakna, bukan estimasi kemampuan yang interpretabel.)

**Mengapa DEMO 2 tidak divergen meskipun 7 item:**

DEMO 2 menggunakan 7 item dengan **pola respons campuran** - tidak semua benar, ada yang salah:
$\mathbf{u}=[1,0,0,1,1,0,1]$. Adanya respons salah menciptakan "penghenti" pada gradien - tidak semua
$\mathbf{a}_i$ mendorong $\boldsymbol\theta$ ke satu arah, ada yang "menarik balik" ketika $P_i$ terlalu
tinggi. Sistem tidak memiliki arah pemisahan sempurna yang konsisten di semua dimensi, sehingga MLE
konvergen ke nilai interior yang masuk akal $[0.038,\,-0.654,\,0.297]$.

Ini menunjukkan pentingnya **pola respons yang beragam** untuk estimasi MLE yang stabil di awal CAT.

### 3.3 Kelebihan dan Kekurangan

**Kelebihan:**
- Landasan teori paling matang & tertua - dasar dari seluruh literatur IRT sejak Lord (1980),
  dan Newton-Raphson/Fisher scoring-nya sudah didokumentasikan lengkap dengan contoh numerik
  ber-halaman oleh Baker (2001) [2, Eq.5-1, p.86-88].
- Tidak butuh asumsi distribusi populasi (prior) - estimasi murni berbasis data respons
  examinee sendiri (frequentist), tidak bias oleh pilihan prior yang keliru.
- Asimtotik efisien & normal [1, Eq.7, p.277] - untuk tes yang cukup panjang, varians estimasi
  mendekati batas bawah Cramér–Rao.

**Kekurangan:**
- **Divergen** bila pola respons dapat dipisahkan sempurna oleh arah linear tertentu dari
  $\mathbf{a}_i$ - dibuktikan langsung di [#3.2.3](#323-demo-3-kasus-divergen-all-correct), bukan hanya kasus trivial
  all-correct/all-incorrect. Risiko ini lebih tinggi di awal tes (item sedikit) - persis mengapa
  MCAT umumnya memakai MAP di round-round awal (lihat [#2](#4-maximum-a-posteriori-map-atau-bayes-modal)).
- Tidak ada mekanisme built-in untuk mencegah estimasi ekstrem, sehingga kasus divergen menghasilkan nilai besar tak-berguna (mis. $\|\hat\theta\|=24.43$ di [#3.2.3](#323-demo-3-kasus-divergen-all-correct)).
- Butuh minimal beberapa item dengan variasi respons (benar & salah) untuk estimasi yang stabil -
  tidak cocok dipakai sebagai estimator tunggal di 1-2 round pertama CAT.

---

## 4. Maximum A Posteriori (MAP atau Bayes Modal)

### 4.1 Teori

MAP (disebut juga *Bayes Modal*/BM) memaksimalkan **posterior**, bukan likelihood murni - Magis &
Raîche (2012), *"Random Generation of Response Patterns under Computerized Adaptive Testing with
the R Package catR"*, Journal of Statistical Software 48(8), **#2.2 "Ability estimation", p.4-5**
[3]:

$$
g(\theta) = f(\theta)\,L(\theta) \qquad\Rightarrow\qquad \log g(\theta) = \log f(\theta) + \log L(\theta)
\tag{5, p.5}
$$ 


$$
\hat\theta_{BM} = \arg\max_\theta g(\theta)
$$

dengan $f(\theta)$ *prior* dan $L(\theta)$ likelihood (identik $f(\mathbf u\mid\theta)$ di
[#2](#2-model-dan-notasi-dasar-dipakai-oleh-ketiga-metode)). Magis & Raîche [3, p.4]: *"The choice of a prior distribution is
usually driven by some prior belief of the ability distribution among the population of
examinees. The most common choice is the normal distribution with mean $\mu$ and variance
$\sigma^2$."* Catatan ini menggunakan **multivariate normal** $\pi(\boldsymbol\theta)=N(\boldsymbol\mu,\boldsymbol\Sigma)$
dengan $\boldsymbol\Sigma$ diagonal. Paper aslinya (BM
diformalkan oleh Mislevy 1986 [4], dirujuk di [3, p.4]) tidak dapat diakses gratis untuk
verifikasi halaman langsung - diverifikasi silang melalui Magis & Raîche [3] yang open access dan
mereproduksi definisi Eq.5 secara eksplisit dengan nomor persamaan.

#### 4.1.1 Apa Bedanya "Prior" dan "Variance"?

Dua istilah ini sering tertukar padahal levelnya berbeda: **variance bukan hal terpisah dari
prior - variance adalah salah satu parameter *di dalam* prior**, bukan konsep yang sejajar. Jadi kalau ditulis matematis, prior = N(μ, Σ).

- **Prior** = keseluruhan asumsi/keyakinan tentang distribusi $\theta$ sebelum ada data respons -
  mencakup bentuk distribusinya (di sini: normal), titik tengahnya ($\mu$), dan lebar
  sebarannya ($\Sigma$).
- **Mean ($\mu$)** = parameter di dalam prior yang menentukan estimasi ditarik ke arah mana
  (biasanya $0$, mewakili "peserta rata-rata").
- **Variance ($\Sigma$)** = parameter di dalam prior yang menentukan **seberapa kuat** tarikan
  itu. Variance kecil = prior "yakin"/sempit = tarikan kuat (*shrinkage* besar ke $\mu$). Variance
  besar = prior "longgar"/lebar = tarikan lemah, estimasi mendekati MLE murni.

Analogi: *"Saya percaya kemampuan peserta di populasi berbentuk lonceng (normal), berpusat di
0"* adalah **prior**-nya. *"Seberapa lebar/sempit lonceng itu"* adalah **variance**-nya - satu
angka di dalam prior tersebut.

**Dampak konkret ke perhitungan** (item bank & respons identik [#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i), Iterasi 1, start $\hat{\boldsymbol\theta}_0=\boldsymbol\mu=[0,0,0]$
- karena $\theta_0=\mu$, prior gradient di Step 4 selalu nol berapa pun $\Sigma$-nya, sehingga efek
$\Sigma$ pada iterasi ini murni lewat Hessian di Step 7-9):

| Prior | $\Sigma$ (variance) | $\boldsymbol\Sigma^{-1}$ (precision) | $\hat{\boldsymbol\theta}_1$ (numeric) | Interpretasi |
|---|---|---|---|---|
| Sangat kuat/yakin | $0.25$ | $4$ | $-0.1641$ | shrinkage besar, paling dekat ke $\mu=0$ |
| Existing di demo ini | $1$ | $1$ | $-0.3794$ | shrinkage sedang |
| Longgar | $4$ | $0.25$ | $-0.5769$ | shrinkage kecil |
| Tanpa prior (MLE) | $\infty$ | $0$ | $-0.7017$ | tidak ada shrinkage sama sekali |

Semakin kecil variance ($\Sigma\downarrow$), semakin besar precision-nya ($\Sigma^{-1}\uparrow$),
semakin besar "penalti" yang ditambahkan ke $\mathbf{H}_{MAP}=\mathbf{I}_S+\boldsymbol\Sigma^{-1}$,
semakin kecil step Newton-nya, dan estimasi makin tertarik ke $\mu$. Bila $\Sigma$ bukan diagonal
(dimensi berkorelasi di populasi), $\boldsymbol\Sigma^{-1}$ juga akan punya nilai off-diagonal,
sehingga prior di satu dimensi ikut menarik dimensi lainnya - kode produksi saat ini memakai
$\Sigma$ diagonal saja (`prior_cov_diag`), jadi kasus ini tidak dibahas lebih lanjut di sini.

**Pembuktian (bukan dikutip, diturunkan sendiri dari Eq.5 di atas):** untuk prior multivariate
normal $\pi(\boldsymbol\theta)=(2\pi)^{-k/2}|\boldsymbol\Sigma|^{-1/2}\exp\!\big(-\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)\big)$:

$$
\log f(\boldsymbol\theta) = -\tfrac12(\boldsymbol\theta-\boldsymbol\mu)^\top\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu) + \text{const}
$$

$$
\nabla\log f(\boldsymbol\theta) = -\boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu), \qquad
\frac{\partial^2\log f}{\partial\boldsymbol\theta\partial\boldsymbol\theta^\top} = -\boldsymbol\Sigma^{-1}
$$

(turunan standar bentuk kuadratik multivariat - eksak, bukan ekspektasi, karena $\log f$ memang
kuadratik murni). Menggabungkan dengan skor & FIM likelihood dari [#4.1](#21-pembuktian-skor-gradien-log-likelihood)/[#4.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring):

$$
\boxed{\nabla\log g(\boldsymbol\theta) = \nabla\log f(\boldsymbol\theta) - \boldsymbol\Sigma^{-1}(\boldsymbol\theta-\boldsymbol\mu)}
\qquad
\boxed{\mathbf{H}_{MAP} \approx \mathbf{I}_S(\boldsymbol\theta) + \boldsymbol\Sigma^{-1}}
$$

- Newton step: $\hat{\boldsymbol\theta}_{s+1}=\hat{\boldsymbol\theta}_s+\mathbf{H}_{MAP}^{-1}\nabla\log g$,
mulai dari $\hat{\boldsymbol\theta}_0=\boldsymbol\mu$, bukan $\mathbf 0$ seperti MLE.

Karena $\boldsymbol\Sigma^{-1}\succeq0$ selalu ditambahkan ke $\mathbf{I}_S(\boldsymbol\theta)\succeq0$,
$\mathbf{H}_{MAP}\succeq\mathbf{H}_{MLE}$ - informasi MAP selalu $\geq$ MLE, menjelaskan standard
error MAP yang lebih kecil, sesuai [3, Eq.6, p.5]:
$se(\hat\theta_{BM}) = 1/\sqrt{1/\sigma^2 + \sum_i I_i(\hat\theta_{BM})}$ (bentuk univariat).
Generalisasi multivariat dan pembuktian bahwa formula ini **memang** dihitung oleh kode produksi -
dijabarkan di [#4.1.2](#412-teori-standard-error-se-untuk-map).

**Keterangan variabel (tambahan untuk MAP):**

| Simbol | Arti |
|---|---|
| $g(\theta)$ | Posterior tak-ternormalisasi $=f(\theta)L(\theta)$ |
| $f(\theta)$, $\pi(\boldsymbol\theta)$ | Densitas prior (dipakai bergantian, notasi Magis & Raîche vs notasi umum) |
| $\boldsymbol\mu$, $\boldsymbol\Sigma$ | Mean & kovarians prior (produksi: `prior_mean`, `diag(prior_cov_diag)`) |
| $\boldsymbol\Sigma^{-1}$ | *Prior precision* - presisi/informasi prior, ditambahkan langsung ke FIM |
| $\mathbf{H}_{MAP}$ | Hessian (Fisher scoring) posterior $=\mathbf{I}_S(\theta)+\boldsymbol\Sigma^{-1}$ |

#### 4.1.2 Teori: Standard Error (SE) untuk MAP

Sama seperti MLE ([#3.1.1](#311-teori-standard-error-se-untuk-mle)), SE MAP berasal dari **pendekatan
Laplace**: di sekitar mode posterior $\hat\theta_{BM}$, posterior didekati normal dengan matriks
kovarians $=$ invers Hessian posterior yang sudah dibuktikan di [#4.1](#41-teori),
$\mathbf{H}_{MAP}=\mathbf{I}_S(\boldsymbol\theta)+\boldsymbol\Sigma^{-1}$ - Hessian yang **sama persis**
dengan yang dipakai Newton-Raphson untuk mencari $\hat\theta_{BM}$ itu sendiri (tidak ada perhitungan
tambahan terpisah). Generalisasi multivariat dari bentuk univariat Magis & Raîche [3, Eq.6, p.5]:

$$
\boxed{SE(\hat\theta_{BM,j}) = \sqrt{\left[\mathbf{H}_{MAP}(\hat{\boldsymbol\theta}_{BM})^{-1}\right]_{jj}}}, \qquad
\mathbf{H}_{MAP} = \mathbf{I}_S(\hat{\boldsymbol\theta}_{BM}) + \boldsymbol\Sigma^{-1}
$$

Untuk $k=1$ dengan $\boldsymbol\Sigma^{-1}=1/\sigma^2$, ini tereduksi tepat ke
$se(\hat\theta_{BM})=1/\sqrt{1/\sigma^2+\sum_iI_i(\hat\theta_{BM})}$ [3, Eq.6, p.5] yang dikutip di
[#4.1](#41-teori).

Karena $\mathbf{H}_{MAP}\succeq\mathbf{I}_S$ (dibuktikan [#4.1](#41-teori)),
$\mathbf{H}_{MAP}^{-1}\preceq\mathbf{I}_S^{-1}$ (invers matriks definit-positif membalik urutan
Loewner) - sehingga $SE_{MAP}\leq SE_{MLE}$ **untuk setiap dimensi**, secara aljabar menjelaskan
pengamatan "informasi MAP selalu $\geq$ MLE" di [#4.1](#41-teori): prior menambah informasi, sehingga
selalu memperkecil (atau menyamakan, jika $\boldsymbol\Sigma^{-1}\to0$) SE dibanding MLE murni pada
data identik.

**Identik dengan kode produksi:** `map::estimate()` sendiri tidak menghitung SE (`map.rs:51-53`:
`posterior_se: None`, sesuai komentar *"MAP's SE is derived afterward from cum_fim + prior_cov_inv
(McatEngine::compute_se), not produced here"*) - persis seperti MLE, satu langkah terpisah setelah
Newton-Raphson berhenti yang menghitung rumus di atas:

```rust
// engine.rs, closure map_style_se() dipakai untuk EstimationMethod::Map:
let prior_cov_inv = DMatrix::from_diagonal(&DVector::from_vec(
    settings.prior_cov_diag.0.iter().map(|v| 1.0 / v).collect(),
));
se_vector(&(cum_fim + prior_cov_inv), k)   // = sqrt(diag(inv(I_S + Σ⁻¹)))
```

`se_vector()` (`mirt.rs:62-71`) itu sendiri identik dengan definisi di
[#3.1.1](#311-teori-standard-error-se-untuk-mle): akar diagonal invers matriks yang diberikan - hanya
matriks yang diberikan berbeda (`cum_fim + prior_cov_inv` untuk MAP, vs `cum_fim` saja untuk MLE).
Konsekuensi praktis lain: karena $\boldsymbol\Sigma^{-1}$ selalu $\succ0$ untuk prior proper, SE MAP
tetap terdefinisi bahkan pada $n=0$ item (posterior $=$ prior, $\mathbf{H}_{MAP}=\boldsymbol\Sigma^{-1}$)
- berbeda dari SE MLE yang `None`/tak terdefinisi sebelum ada item terjawab sama sekali ($\mathbf{I}_S=\mathbf 0$
tidak invertible).

### 4.2 Perhitungan Manual

#### 4.2.1 DEMO 1: MAP vs MLE (k=1), prior N(0,1)

Item & respons identik [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1), prior $\mu=0,\sigma^2=1\Rightarrow\Sigma^{-1}=1.0$.
$\hat\theta_0=\mu=0$.

| Item | $a$ | $d$ | $c$ | $u$ |
|---|---|---|---|---|
| 1 | 1.0 | +1.0 | 0 | 1 |
| 2 | 1.2 | 0.0 | 0 | 0 |
| 3 | 0.8 | -0.8 | 0 | 1 |

**Iterasi 1** dengan $\hat\theta_0=0$:

**Step 1: Hitung $z_i = a\theta + d$ untuk setiap item:**

| Item | $a$ | $d$ | $z_i = a(0) + d$ |
|---|---|---|---|
| 1 | 1.0 | +1.0 | 1.0 |
| 2 | 1.2 | 0.0 | 0.0 |
| 3 | 0.8 | -0.8 | -0.8 |

**Step 2: Hitung $P_i = \sigma(z_i)$, $Q_i = 1-P_i$, $P'_i = P_iQ_i$:**

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| 1 | 1.0 | $\frac{1}{1+e^{-1.0}} = 0.7311$ | 0.2689 | 0.1966 |
| 2 | 0.0 | 0.5000 | 0.5000 | 0.2500 |
| 3 | -0.8 | $\frac{1}{1+e^{0.8}} = 0.3100$ | 0.6900 | 0.2139 |

**Step 3: Hitung likelihood gradient** $\nabla\log L = \sum_i a_i(u_i - P_i)$:

| Item | $u$ | $a_i(u_i-P_i)$ |
|---|---|---|
| 1 | 1 | $1.0(1-0.7311) = +0.2689$ |
| 2 | 0 | $1.2(0-0.5000) = -0.6000$ |
| 3 | 1 | $0.8(1-0.3100) = +0.5520$ |
| **sum** | | **+0.2209** |

Jadi: $\nabla\log L = 0.2209$

**Step 4: Hitung prior gradient term** $\nabla\log f = -\Sigma^{-1}(\theta - \mu) = -1.0(0 - 0) = 0$

**Step 5: Hitung posterior gradient:**

$$\nabla\log g = \nabla\log L + \nabla\log f = 0.2209 + 0 = 0.2209$$

**Step 6: Hitung Hessian likelihood** $H_L = \sum_i a_i^2 P_i Q_i$:

| Item | $a_i^2 P_i Q_i$ |
|---|---|
| 1 | $(1.0)^2(0.7311)(0.2689) = 0.1966$ |
| 2 | $(1.2)^2(0.5000)(0.5000) = 0.3600$ |
| 3 | $(0.8)^2(0.3100)(0.6900) = 0.1369$ |
| **sum** | **0.6935** |

**Step 7: Hitung Hessian MAP** (Fisher scoring + prior Hessian):

$$H_{MAP} = H_L + \Sigma^{-1} = 0.6935 + 1.0 = 1.6935$$

**Step 8: Hitung parameter update:**

$$\Delta\hat\theta = \frac{\nabla\log g}{H_{MAP}} = \frac{0.2209}{1.6935} \approx 0.130451$$

$$\hat\theta_1 = \hat\theta_0 + \Delta\hat\theta = 0 + 0.130451 = 0.130451$$

**Iterasi 2** dengan $\hat\theta_1 = 0.130451$:

**Step 1: Hitung $z_i$ baru:**

| Item | $z_i = a(0.130451) + d$ |
|---|---|
| 1 | $1.0(0.130451) + 1.0 = 1.130451$ |
| 2 | $1.2(0.130451) + 0.0 = 0.156541$ |
| 3 | $0.8(0.130451) - 0.8 = -0.695639$ |

**Step 2: Hitung $P_i, Q_i$:**

| Item | $z_i$ | $P_i$ | $Q_i$ |
|---|---|---|---|
| 1 | 1.130451 | 0.755922 | 0.244078 |
| 2 | 0.156541 | 0.539056 | 0.460944 |
| 3 | -0.695639 | 0.332780 | 0.667220 |

**Step 3: Hitung likelihood gradient:**

$$\nabla\log L = \sum_i a_i(u_i-P_i) = 1.0(1-0.755922) + 1.2(0-0.539056) + 0.8(1-0.332780) = 0.244078 - 0.646867 + 0.533776 = 0.130987$$

**Step 4: Hitung prior gradient:**

$$\nabla\log f = -1.0(0.130451 - 0) = -0.130451$$

**Step 5: Hitung posterior gradient:**

$$\nabla\log g = 0.130987 - 0.130451 = 0.000536$$

**Step 6: Hitung Hessian:**

$$H_L = (1.0)^2(0.755922)(0.244078) + (1.2)^2(0.539056)(0.460944) + (0.8)^2(0.332780)(0.667220) \approx 0.184496 + 0.357761 + 0.142154 = 0.684411$$

$$H_{MAP} = 0.684411 + 1.0 = 1.684411$$

**Step 7: Update parameter:**

$$\Delta\hat\theta = \frac{0.000536}{1.684411} \approx 0.000318 \quad\Rightarrow\quad \hat\theta_2 = 0.130451 + 0.000318 = 0.130769$$

**Verifikasi konvergensi** iterasi 2: $|\Delta\hat\theta| = 3.18\times10^{-4} \gg 10^{-6} \Rightarrow$ **BELUM KONVERGEN**, lanjut iterasi.

**Iterasi 3** dengan $\hat\theta_2 = 0.130769$:

Pada titik ini $\hat\theta_2$ sudah sangat dekat dengan mode posterior, sehingga $\nabla\log L$ dan $-\nabla\log f$ hampir saling meniadakan:

$$\nabla\log L \approx 0.130769, \quad \nabla\log f = -0.130769, \quad \nabla\log g \approx 0.00000000$$

$$H_{MAP} \approx 1.684383, \quad \Delta\hat\theta \approx 2.70\times10^{-9}$$

$$\hat\theta_3 = \hat\theta_2 + \Delta\hat\theta \approx 0.130769$$

**Verifikasi konvergensi:** $|\Delta\hat\theta| = 2.70\times10^{-9} < 10^{-6} \Rightarrow \textbf{KONVERGEN}$ - iterasi berhenti (persis pola yang sama seperti demo MLE [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1): perubahan harus benar-benar jatuh di bawah $10^{-6}$, bukan sekadar "terlihat kecil" setelah pembulatan tampilan).

**Perbandingan dengan MLE pada data identik:**

| Metode | Hasil | Start | Prior |
|---|---|---|---|
| MLE | 0.3248 | $\theta_0=1.0$ | tidak ada |
| MAP | 0.1308 | $\theta_0=0$ | $N(0,1)$ |

MAP tersusut (*shrinkage*) signifikan ke arah mean prior $\mu=0$ - dari 0.3248 menjadi 0.1308 (60% lebih dekat ke 0). Ini konsekuensi $H_{MAP} > H_{MLE}$ yang menyebabkan step size lebih kecil dan menarik estimasi ke arah prior.

**Step tambahan: Hitung Standard Error (SE)** (teori [#4.1.2](#412-teori-standard-error-se-untuk-map))

Karena kriteria konvergensi terpenuhi tepat di Iterasi 3, $H_{MAP}$ pada titik itu (dihitung di Step 6
Iterasi 3 sebelumnya, $H_{MAP}\approx1.684383$) langsung dipakai untuk SE - tidak perlu evaluasi ulang:

$$
SE(\hat\theta_{MAP}) = \frac{1}{\sqrt{H_{MAP}}} = \frac{1}{\sqrt{1.684383}} \approx 0.770512
$$

Dibandingkan MLE pada data identik ($SE(\hat\theta_{MLE})=1/\sqrt{0.661446}\approx1.229569$, dihitung
di [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1)) - SE MAP **37% lebih kecil** ($0.7705$ vs $1.2296$),
konsekuensi langsung $H_{MAP}=0.661446+1.0=1.684383 > \mathbf{I}_S(\hat\theta)_{MLE}=0.661446$ yang
dibuktikan di [#4.1.2](#412-teori-standard-error-se-untuk-map).

#### 4.2.2 DEMO 2: Multidimensional (k=3), 7 item, prior N(0,I)

Item & respons identik [#3.2.2](#322-demo-2-multidimensional-k3-7-item), prior $\boldsymbol\mu=[0,0,0]$, $\boldsymbol\Sigma=\mathbf{I}$ (diagonal) $\Rightarrow \boldsymbol\Sigma^{-1}=\mathbf{I}$ (prior presisi diagonal $[1,1,1]$).

MAP mulai dari $\hat{\boldsymbol\theta}_0=\boldsymbol\mu=[0,0,0]$ (kebetulan sama nilainya dengan start MLE di
[#3.2.2](#322-demo-2-multidimensional-k3-7-item) karena $\mu=\mathbf 0$, tapi secara konseptual berbeda sumber: start dari mean prior, bukan arbitrary zero).

Menggunakan seluruh 7-item bank yang sama seperti
[Item Bank Snapshot](/posts/cat/item-selection-criteria-mcat#item-bank-snapshot).

| Item | $\mathbf{a}$ | $d$ | $u$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | 1 |
| m2p-v002 | [1.7,0.2,0.2] | 0.10 | 0 |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | 0 |
| m2p-n002 | [0.3,1.8,0.4] | 0.50 | 1 |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | 1 |
| m2p-r002 | [0.3,0.8,1.9] | 0.60 | 0 |
| m2p-r003 | [0.4,0.3,1.8] | 0.70 | 1 |

**Iterasi 1** dengan $\hat{\boldsymbol\theta}_0=[0,0,0]$:

**Step 1: Hitung $z_i = \mathbf{a}_i \cdot \boldsymbol\theta_0 + d_i$** dengan $\boldsymbol\theta_0=[0,0,0]$ (dot product dengan vektor nol $=0$, sehingga $z_i=d_i$):

| Item | $\mathbf{a}_i$ | $d_i$ | $z_i$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | 0.40 |
| m2p-v002 | [1.7,0.2,0.2] | 0.10 | 0.10 |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | 0.80 |
| m2p-n002 | [0.3,1.8,0.4] | 0.50 | 0.50 |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | 0.30 |
| m2p-r002 | [0.3,0.8,1.9] | 0.60 | 0.60 |
| m2p-r003 | [0.4,0.3,1.8] | 0.70 | 0.70 |

**Step 2: Hitung $P_i=\sigma(z_i)$, $Q_i=1-P_i$, $P'_i=P_iQ_i$ (M2PL, $c=0$):**

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.40 | 0.5987 | 0.4013 | 0.2403 |
| m2p-v002 | 0.10 | 0.5250 | 0.4750 | 0.2494 |
| m2p-n001 | 0.80 | 0.6900 | 0.3100 | 0.2139 |
| m2p-n002 | 0.50 | 0.6225 | 0.3775 | 0.2350 |
| m2p-r001 | 0.30 | 0.5744 | 0.4256 | 0.2446 |
| m2p-r002 | 0.60 | 0.6456 | 0.3544 | 0.2290 |
| m2p-r003 | 0.70 | 0.6682 | 0.3318 | 0.2217 |

**Step 3: Hitung residual $(u_i-P_i)$ dan kontribusi ke gradien $\mathbf{a}_i(u_i-P_i)$:**

| Item | $u$ | $(u_i-P_i)$ | Kontribusi = $\mathbf{a}_i \times (u_i - P_i)$ |
|---|---|---|---|
| m2p-v001 | 1 | $1-0.5987=0.4013$ | $[0.7625,0.0803,0.1204]$ |
| m2p-v002 | 0 | $0-0.5250=-0.5250$ | $[-0.8925,-0.1050,-0.1050]$ |
| m2p-n001 | 0 | $0-0.6900=-0.6900$ | $[-0.2070,-1.3110,-0.2760]$ |
| m2p-n002 | 1 | $1-0.6225=0.3775$ | $[0.1133,0.6795,0.1510]$ |
| m2p-r001 | 1 | $1-0.5744=0.4256$ | $[0.2128,0.1702,0.8512]$ |
| m2p-r002 | 0 | $0-0.6456=-0.6456$ | $[-0.1937,-0.5165,-1.2266]$ |
| m2p-r003 | 1 | $1-0.6682=0.3318$ | $[0.1327,0.0995,0.5972]$ |

Menjumlahkan seluruh kontribusi (persis perhitungan yang sama dengan MLE Iterasi 1, lihat [#3.2.2](#322-demo-2-multidimensional-k3-7-item), karena likelihood gradient tidak bergantung pada prior):

$$\nabla\log L = [-0.0719,-0.9029,0.1121]$$

**Step 4: Prior gradient term:**

$$\nabla\log f = -\boldsymbol\Sigma^{-1}(\boldsymbol\theta - \boldsymbol\mu) = -\mathbf{I}([0,0,0] - [0,0,0]) = [0,0,0]$$

**Step 5: Posterior gradient:**

$$\nabla\log g = \nabla\log L + \nabla\log f = [-0.0719,-0.9029,0.1121] + [0,0,0] = [-0.0719,-0.9029,0.1121]$$

(Sama dengan likelihood gradient karena $\theta_0 = \mu$)

**Step 6: Hitung FIM per item** $\mathbf{I}_i=P'_i\,\mathbf{a}_i\mathbf{a}_i^\top$ menggunakan $P'_i$ dari Step 2, lalu jumlahkan atas ke-7 item (persis definisi [#2.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring)):

Sebagai contoh, untuk item m2p-v001 ($P'_1=0.2403$):

$$
\mathbf{I}_1 = 0.2403 \times \begin{bmatrix} 1.9 \\ 0.2 \\ 0.3 \end{bmatrix}\begin{bmatrix} 1.9 & 0.2 & 0.3 \end{bmatrix} = \begin{bmatrix} 0.8673 & 0.0913 & 0.1369 \\ 0.0913 & 0.0096 & 0.0144 \\ 0.1369 & 0.0144 & 0.0216 \end{bmatrix}
$$

Dilakukan sama untuk 6 item lainnya:

| Item | $P'_i$ | $\mathbf{I}_i$ (diagonal $[a_1^2,a_2^2,a_3^2]\times P'_i$) |
|---|---|---|
| m2p-v001 | 0.2403 | $[0.8673,\,0.0096,\,0.0216]$ |
| m2p-v002 | 0.2494 | $[0.7207,\,0.0100,\,0.0100]$ |
| m2p-n001 | 0.2139 | $[0.0193,\,0.7722,\,0.0342]$ |
| m2p-n002 | 0.2350 | $[0.0212,\,0.7614,\,0.0376]$ |
| m2p-r001 | 0.2446 | $[0.0611,\,0.0391,\,0.9778]$ |
| m2p-r002 | 0.2290 | $[0.0206,\,0.1464,\,0.8259]$ |
| m2p-r003 | 0.2217 | $[0.0355,\,0.0200,\,0.7183]$ |

Menjumlahkan seluruh $\mathbf{I}_i$ (matriks lengkap $3\times3$, bukan hanya diagonal) menghasilkan FIM total - persis sama dengan FIM MLE Iterasi 1 di [#3.2.2](#322-demo-2-multidimensional-k3-7-item), karena FIM likelihood juga tidak bergantung pada prior:

$$\mathbf{I}_S^{(L)} = \begin{bmatrix} 1.7456 & 0.5553 & 0.8101 \\ 0.5553 & 1.7587 & 1.0192 \\ 0.8101 & 1.0192 & 2.6255 \end{bmatrix}$$

**Step 7: Prior Hessian (eksak untuk prior kuadratik):**

$$-\frac{\partial^2\log f}{\partial\boldsymbol\theta\partial\boldsymbol\theta^\top} = \boldsymbol\Sigma^{-1} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Step 8: Posterior Hessian:**

$$\mathbf{H}_{MAP} = \mathbf{I}_S^{(L)} + \boldsymbol\Sigma^{-1} = \begin{bmatrix} 1.7456+1 & 0.5553 & 0.8101 \\ 0.5553 & 1.7587+1 & 1.0192 \\ 0.8101 & 1.0192 & 2.6255+1 \end{bmatrix}$$

$$= \begin{bmatrix} 2.7456 & 0.5553 & 0.8101 \\ 0.5553 & 2.7587 & 1.0192 \\ 0.8101 & 1.0192 & 3.6255 \end{bmatrix}$$

(Diagonal $[2.7456, 2.7587, 3.6255]$ seperti tercatat)

**Step 9: Hitung invers dan parameter update:**

$$\mathbf{H}_{MAP}^{-1} \approx \begin{bmatrix} 0.3966 & -0.0526 & -0.0739 \\ -0.0526 & 0.4115 & -0.1039 \\ -0.0739 & -0.1039 & 0.3215 \end{bmatrix}$$

$$\Delta\hat{\boldsymbol\theta} = \mathbf{H}_{MAP}^{-1} \nabla\log g = \begin{bmatrix} 0.3966 & -0.0526 & -0.0739 \\ -0.0526 & 0.4115 & -0.1039 \\ -0.0739 & -0.1039 & 0.3215 \end{bmatrix} \begin{bmatrix} -0.0719 \\ -0.9029 \\ 0.1121 \end{bmatrix}$$

$$= \begin{bmatrix} -0.0285+0.0475-0.0083 \\ 0.0038-0.3715-0.0116 \\ 0.0053+0.0938+0.0360 \end{bmatrix} = \begin{bmatrix} 0.0107 \\ -0.3794 \\ 0.1352 \end{bmatrix}$$

$$\hat{\boldsymbol\theta}_1 = [0,0,0] + [0.0107,-0.3794,0.1352] = [0.0107,-0.3794,0.1352]$$

**Iterasi 2** dengan $\hat{\boldsymbol\theta}_1 = [0.0107,-0.3794,0.1352]$:

**Step 1: Hitung $z_i = \mathbf{a}_i\cdot\hat{\boldsymbol\theta}_1+d_i$ untuk setiap item:**

| Item | $\mathbf{a}_i\cdot\hat{\boldsymbol\theta}_1$ | $d_i$ | $z_i$ |
|---|---|---|---|
| m2p-v001 | $1.9(0.0107)+0.2(-0.3794)+0.3(0.1352)=0.0203-0.0759+0.0406$ | +0.40 | 0.3850 |
| m2p-v002 | $1.7(0.0107)+0.2(-0.3794)+0.2(0.1352)=0.0182-0.0759+0.0270$ | +0.10 | 0.0693 |
| m2p-n001 | $0.3(0.0107)+1.9(-0.3794)+0.4(0.1352)=0.0032-0.7209+0.0541$ | +0.80 | 0.1364 |
| m2p-n002 | $0.3(0.0107)+1.8(-0.3794)+0.4(0.1352)=0.0032-0.6829+0.0541$ | +0.50 | -0.1256 |
| m2p-r001 | $0.5(0.0107)+0.4(-0.3794)+2.0(0.1352)=0.0054-0.1518+0.2704$ | +0.30 | 0.4240 |
| m2p-r002 | $0.3(0.0107)+0.8(-0.3794)+1.9(0.1352)=0.0032-0.3035+0.2569$ | +0.60 | 0.5566 |
| m2p-r003 | $0.4(0.0107)+0.3(-0.3794)+1.8(0.1352)=0.0043-0.1138+0.2434$ | +0.70 | 0.8338 |

**Step 2: Hitung $P_i$, $Q_i$, $P'_i$ berdasarkan $z_i$ baru** (P bergerak lebih sedikit dari titik start dibanding MLE Iterasi 2 karena step MAP lebih kecil - lihat [#3.2.2](#322-demo-2-multidimensional-k3-7-item) sebagai pembanding):

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.3850 | 0.5951 | 0.4049 | 0.2410 |
| m2p-v002 | 0.0693 | 0.5173 | 0.4827 | 0.2497 |
| m2p-n001 | 0.1364 | 0.5341 | 0.4659 | 0.2488 |
| m2p-n002 | -0.1256 | 0.4686 | 0.5314 | 0.2490 |
| m2p-r001 | 0.4240 | 0.6044 | 0.3956 | 0.2391 |
| m2p-r002 | 0.5566 | 0.6356 | 0.3644 | 0.2316 |
| m2p-r003 | 0.8338 | 0.6972 | 0.3028 | 0.2111 |

**Step 3: Hitung kontribusi ke likelihood gradient** $\mathbf{a}_i(u_i-P_i)$, lalu jumlahkan:

| Item | $u$ | $(u_i-P_i)$ | Kontribusi |
|---|---|---|---|
| m2p-v001 | 1 | 0.4049 | $[0.7694,0.0810,0.1215]$ |
| m2p-v002 | 0 | -0.5173 | $[-0.8794,-0.1035,-0.1035]$ |
| m2p-n001 | 0 | -0.5341 | $[-0.1602,-1.0147,-0.2136]$ |
| m2p-n002 | 1 | 0.5314 | $[0.1594,0.9564,0.2125]$ |
| m2p-r001 | 1 | 0.3956 | $[0.1978,0.1582,0.7912]$ |
| m2p-r002 | 0 | -0.6356 | $[-0.1907,-0.5085,-1.2077]$ |
| m2p-r003 | 1 | 0.3028 | $[0.1211,0.0909,0.5451]$ |

$$\nabla\log L = [0.0174,-0.3402,0.1455]$$

**Step 4: Prior gradient:**

$$\nabla\log f = -\mathbf{I}(\hat{\boldsymbol\theta}_1 - [0,0,0]) = -[0.0107,-0.3794,0.1352] = [-0.0107,0.3794,-0.1352]$$

**Step 5: Posterior gradient:**

$$\nabla\log g = [0.0174,-0.3402,0.1455] + [-0.0107,0.3794,-0.1352] = [0.0067,0.0392,0.0103]$$

(Jauh lebih kecil daripada likelihood gradient - prior "menarik balik" mendekati konvergensi)

**Step 6: Hitung FIM baru, Hessian, dan update** (FIM per item dihitung persis seperti Step 6 Iterasi 1, memakai $P'_i$ dari Step 2 di atas):

$$\mathbf{I}_S^{(L,2)} \approx \begin{bmatrix} 1.7507 & 0.5815 & 0.8051 \\ 0.5815 & 1.9302 & 1.0502 \\ 0.8051 & 1.0502 & 2.5879 \end{bmatrix}, \qquad \mathbf{H}_{MAP}^{(2)} = \mathbf{I}_S^{(L,2)}+\mathbf{I} \approx \begin{bmatrix} 2.7507 & 0.5815 & 0.8051 \\ 0.5815 & 2.9302 & 1.0502 \\ 0.8051 & 1.0502 & 3.5879 \end{bmatrix}$$

$$\Delta\hat{\boldsymbol\theta} = \big(\mathbf{H}_{MAP}^{(2)}\big)^{-1}\nabla\log g \approx [-0.0001,\,0.0138,\,-0.0011]$$

$$\hat{\boldsymbol\theta}_2 = [0.0107,-0.3794,0.1352] + [-0.0001,0.0138,-0.0011] \approx [0.0105,-0.3656,0.1340]$$

**Iterasi 3** dengan $\hat{\boldsymbol\theta}_2 = [0.010512,-0.365561,0.134039]$ (presisi penuh, bukan dibulatkan ke 4 desimal, supaya verifikasi konvergensi di bawah akurat):

**Step 1-2: Hitung $z_i$, $P_i$, $Q_i$, $P'_i$:**

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.387072 | 0.595578 | 0.404422 | 0.240865 |
| m2p-v002 | 0.071566 | 0.517884 | 0.482116 | 0.249680 |
| m2p-n001 | 0.162203 | 0.540462 | 0.459538 | 0.248363 |
| m2p-n002 | -0.101241 | 0.474711 | 0.525289 | 0.249360 |
| m2p-r001 | 0.427110 | 0.605183 | 0.394817 | 0.238936 |
| m2p-r002 | 0.565379 | 0.637696 | 0.362304 | 0.231040 |
| m2p-r003 | 0.835807 | 0.697581 | 0.302419 | 0.210962 |

**Step 3: Kontribusi ke likelihood gradient** $\mathbf{a}_i(u_i-P_i)$:

| Item | $u$ | $(u_i-P_i)$ | Kontribusi |
|---|---|---|---|
| m2p-v001 | 1 | 0.404422 | $[0.768402,0.080884,0.121327]$ |
| m2p-v002 | 0 | -0.517884 | $[-0.880403,-0.103577,-0.103577]$ |
| m2p-n001 | 0 | -0.540462 | $[-0.162139,-1.026878,-0.216185]$ |
| m2p-n002 | 1 | 0.525289 | $[0.157587,0.945519,0.210115]$ |
| m2p-r001 | 1 | 0.394817 | $[0.197408,0.157927,0.789633]$ |
| m2p-r002 | 0 | -0.637696 | $[-0.191309,-0.510157,-1.211623]$ |
| m2p-r003 | 1 | 0.302419 | $[0.120967,0.090726,0.544354]$ |

$$\nabla\log L \approx [0.010515,-0.365556,0.134045]$$

**Step 4: Prior gradient:**

$$\nabla\log f = -\mathbf{I}(\hat{\boldsymbol\theta}_2 - [0,0,0]) = [-0.010512,0.365561,-0.134039]$$

**Step 5: Posterior gradient** (likelihood dan prior nyaris saling meniadakan, tersisa residual orde $10^{-6}$):

$$\nabla\log g = [0.010515,-0.365556,0.134045] + [-0.010512,0.365561,-0.134039] \approx [2.7,\,5.3,\,5.8]\times10^{-6}$$

**Step 6: FIM & Hessian:**

$$\mathbf{I}_S^{(L,3)} \approx \begin{bmatrix} 1.7502 & 0.5812 & 0.8044 \\ 0.5812 & 1.9292 & 1.0490 \\ 0.8044 & 1.0490 & 2.5846 \end{bmatrix}, \qquad \mathbf{H}_{MAP}^{(3)} \approx \begin{bmatrix} 2.7502 & 0.5812 & 0.8044 \\ 0.5812 & 2.9292 & 1.0490 \\ 0.8044 & 1.0490 & 3.5846 \end{bmatrix}$$

**Step 7: Update parameter:**

$$\Delta\hat{\boldsymbol\theta} = \big(\mathbf{H}_{MAP}^{(3)}\big)^{-1}\nabla\log g \approx [3.5,\,13.4,\,11.5]\times10^{-7}$$

$$\hat{\boldsymbol\theta}_3 = [0.010512,-0.365561,0.134039] + [0.00000035,0.00000134,0.00000115] \approx [0.0105124,-0.3655597,0.1340402]$$

**Verifikasi konvergensi** iterasi 3, kriteria [#2.3](#23-metode-numerik-iterasi-dan-konvergensi): $\|\hat{\boldsymbol\theta}_3-\hat{\boldsymbol\theta}_2\|=\|\Delta\hat{\boldsymbol\theta}\|=\sqrt{(3.5\times10^{-7})^2+(13.4\times10^{-7})^2+(11.5\times10^{-7})^2}\approx1.80\times10^{-6}$

$$1.80\times10^{-6} \gg 10^{-6} \Rightarrow \textbf{BELUM KONVERGEN}\text{ (meski gradiennya kelihatan "sangat kecil"), lanjut ke iterasi 4.}$$

Ini koreksi terhadap versi sebelumnya di dokumen ini yang menyimpulkan "konvergen" langsung di iterasi 3 - kesimpulan itu keliru karena hanya menilai gradien secara kasar/dibulatkan, bukan mengecek $\|\Delta\hat{\boldsymbol\theta}\|$ terhadap ambang $10^{-6}$ seperti kriteria resmi yang dipakai `map.rs:40` (`if delta_norm < 1e-6`).

**Iterasi 4** dengan $\hat{\boldsymbol\theta}_3 = [0.0105124,-0.3655597,0.1340402]$:

Sama seperti pola konvergensi kuadratik Newton-Raphson pada demo MLE [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1)/[#3.2.2](#322-demo-2-multidimensional-k3-7-item) - begitu dekat dengan akar, error mengecil drastis tiap iterasi:

$$\nabla\log g \approx [4.3,\,5.6,\,16.7]\times10^{-13}, \qquad \Delta\hat{\boldsymbol\theta} \approx [1.7\times10^{-14},\,2.3\times10^{-14},\,4.6\times10^{-13}]$$

$$\hat{\boldsymbol\theta}_4 = \hat{\boldsymbol\theta}_3 + \Delta\hat{\boldsymbol\theta} \approx [0.0105124,-0.3655597,0.1340402]$$

**Verifikasi konvergensi** iterasi 4:

$$\|\hat{\boldsymbol\theta}_4-\hat{\boldsymbol\theta}_3\| = \|\Delta\hat{\boldsymbol\theta}\| \approx 4.57\times10^{-13} < 10^{-6} \Rightarrow \textbf{KONVERGEN} \text{ - iterasi berhenti pada iterasi ke-4.}$$

**Hasil final:** $\hat{\boldsymbol\theta}_{MAP} = [0.010512, -0.365560, 0.134040]$ (konvergen di iterasi 4, bukan 3 - dibulatkan $[0.0105, -0.3656, 0.1340]$ untuk tabel perbandingan di bawah)

**Perbandingan langsung**:

| Metode | $\hat\theta_{verbal}$ | $\hat\theta_{numeric}$ | $\hat\theta_{reasoning}$ |
|---|---|---|---|
| MLE (no prior) | 0.0380 | -0.6537 | 0.2966 |
| MAP ($\Sigma=I$) | 0.0105 | -0.3656 | 0.1340 |

**Analisis shrinkage:**

- **Verbal:** $0.0380 \to 0.0105$ (72% lebih dekat ke 0) - shrinkage minimal
- **Numeric:** $-0.6537 \to -0.3656$ (44% lebih dekat ke 0) - shrinkage signifikan
- **Reasoning:** $0.2966 \to 0.1340$ (55% lebih dekat ke 0) - shrinkage sedang

MAP tersusut (*shrinkage*) ke arah $\mathbf 0$ di **ketiga** dimensi - konsekuensi langsung $\mathbf{H}_{MAP}=\mathbf{I}_S+\mathbf{I}\succ\mathbf{I}_S$ yang dibuktikan di [#4.1](#41-teori). Hessian yang lebih besar berarti curvature posterior lebih tajam, sehingga step-size lebih kecil dan penarik dari $\mu=0$ lebih kuat.

**Step tambahan: Hitung Standard Error (SE)** (teori [#4.1.2](#412-teori-standard-error-se-untuk-map))

Menggunakan $\mathbf{H}_{MAP}^{(3)}$ dari Iterasi 3 (titik konvergen praktis, karena
$\hat{\boldsymbol\theta}_4\approx\hat{\boldsymbol\theta}_3$ hingga presisi $10^{-6}$):

$$
\big(\mathbf{H}_{MAP}^{(3)}\big)^{-1} \approx \begin{bmatrix} 0.39619 & -0.05224 & -0.07362 \\ -0.05224 & 0.38824 & -0.10189 \\ -0.07362 & -0.10189 & 0.32531 \end{bmatrix}
$$

$$
SE(\hat{\boldsymbol\theta}_{MAP}) = \sqrt{\text{diag}\big(\mathbf{H}_{MAP}^{-1}\big)} \approx [0.6294,\,0.6231,\,0.5704]
$$

Dibandingkan MLE pada data identik ($SE(\hat{\boldsymbol\theta}_{MLE})\approx[0.8309,\,0.8458,\,0.7536]$,
dihitung di [#3.2.2](#322-demo-2-multidimensional-k3-7-item)) - SE MAP lebih kecil di **ketiga**
dimensi (24-32% penyusutan), pola shrinkage yang sama seperti pada titik estimasi $\hat\theta$ sendiri
(lihat "Analisis shrinkage" di atas): prior menyumbang informasi tambahan $\boldsymbol\Sigma^{-1}=\mathbf I$
yang langsung memperkecil variance posterior di semua dimensi.

#### 4.2.3 DEMO 3: MAP Meregularisasi Kasus Divergen

Data identik [#5.2.3](#323-demo-3-kasus-divergen-all-correct) (3 item, $\mathbf u=[1,1,1]$, prior $N(\mathbf 0,\mathbf I)$):

| Item | $\mathbf{a}$ | $d$ | $u$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | **1** |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | **1** |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | **1** |

**Hasil final:**

$$
\hat{\boldsymbol\theta}_{MLE} = [16.065,\,14.291,\,11.594],\; \|\hat\theta\|=24.43 \qquad\text{(divergen, lihat \S1.2.3)}
$$

$$
\hat{\boldsymbol\theta}_{MAP} = [0.4719,\,0.3688,\,0.4505],\; \|\hat\theta\|=0.75 \qquad\text{(finite, teregularisasi)}
$$

**Iterasi 1** dengan $\hat{\boldsymbol\theta}_0=[0,0,0]$:

**Step 1-2: Hitung $z_i$, $P_i$, $Q_i$, $P'_i$** (dot product dengan $[0,0,0]$ sehingga $z_i=d_i$, persis pola yang sama seperti [#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i) Iterasi 1):

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 0.400000 | 0.598688 | 0.401312 | 0.240261 |
| m2p-n001 | 0.800000 | 0.689974 | 0.310026 | 0.213910 |
| m2p-r001 | 0.300000 | 0.574443 | 0.425557 | 0.244458 |

**Step 3: Kontribusi ke likelihood gradient** $\mathbf{a}_i(u_i-P_i)$ - **semua residual positif** karena ketiga respons benar ($u_i=1$ untuk semua $i$), berbeda dari [#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i) yang polanya campuran:

| Item | $u$ | $(u_i-P_i)$ | Kontribusi |
|---|---|---|---|
| m2p-v001 | 1 | 0.401312 | $[0.762493,0.080262,0.120394]$ |
| m2p-n001 | 1 | 0.310026 | $[0.093008,0.589048,0.124010]$ |
| m2p-r001 | 1 | 0.425557 | $[0.212779,0.170223,0.851115]$ |

$$\nabla\log L = [1.068280,\,0.839534,\,1.095519]$$

**Step 4: Prior gradient** (pada $\hat{\boldsymbol\theta}_0=\boldsymbol\mu$, jadi nol seperti biasa di iterasi pertama):

$$\nabla\log f = -\mathbf{I}([0,0,0] - [0,0,0]) = [0,0,0]$$

**Step 5: Posterior gradient** $=\nabla\log L$ (karena $\theta_0=\mu$):

$$\nabla\log g = [1.068280,\,0.839534,\,1.095519]$$

**Step 6: FIM per item** $\mathbf{I}_i=P'_i\,\mathbf{a}_i\mathbf{a}_i^\top$, dijumlahkan atas 3 item:

$$\mathbf{I}_S^{(L)} = \begin{bmatrix} 0.947708 & 0.262119 & 0.407076 \\ 0.262119 & 0.820938 & 0.372554 \\ 0.407076 & 0.372554 & 1.033682 \end{bmatrix}$$

**Step 7: Hessian MAP:**

$$\mathbf{H}_{MAP}^{(1)} = \mathbf{I}_S^{(L)} + \mathbf{I} = \begin{bmatrix} 1.947708 & 0.262119 & 0.407076 \\ 0.262119 & 1.820938 & 0.372554 \\ 0.407076 & 0.372554 & 2.033682 \end{bmatrix}$$

**Step 8: Invers Hessian dan update parameter:**

$$\mathbf{H}_{MAP}^{-1} \approx \begin{bmatrix} 0.541666 & -0.057961 & -0.097806 \\ -0.057961 & 0.576754 & -0.094055 \\ -0.097806 & -0.094055 & 0.528526 \end{bmatrix}$$

$$\Delta\hat{\boldsymbol\theta}^{(1)} = \mathbf{H}_{MAP}^{-1} \nabla\log g \approx [0.422843,\,0.319247,\,0.395565]$$

$$\hat{\boldsymbol\theta}_1 = [0,0,0] + [0.422843,\,0.319247,\,0.395565] = [0.422843,\,0.319247,\,0.395565], \quad \|\hat{\boldsymbol\theta}_1\| \approx 0.6612$$

(Step sebesar ini wajar - belum ada resistansi prior sama sekali di iterasi pertama, sama seperti [#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i))

**Iterasi 2** dengan $\hat{\boldsymbol\theta}_1 = [0.422843,\,0.319247,\,0.395565]$:

**Step 1-2: Hitung $z_i$, $P_i$, $Q_i$, $P'_i$ baru** (semua $P_i$ meningkat tajam - efek "semua benar" mendorong $\theta$ ke atas):

| Item | $z_i$ | $P_i$ | $Q_i$ | $P'_i$ |
|---|---|---|---|---|
| m2p-v001 | 1.385920 | 0.799940 | 0.200060 | 0.160036 |
| m2p-n001 | 1.691649 | 0.844441 | 0.155559 | 0.131360 |
| m2p-r001 | 1.430250 | 0.806940 | 0.193060 | 0.155788 |

**Step 3: Kontribusi ke likelihood gradient** (residual sudah menyusut - $P_i$ dekat 1, tapi tetap positif, terus mendorong $\theta$ naik seperti pola MLE divergen):

| Item | $u$ | $(u_i-P_i)$ | Kontribusi |
|---|---|---|---|
| m2p-v001 | 1 | 0.200060 | $[0.380114,0.040012,0.060018]$ |
| m2p-n001 | 1 | 0.155559 | $[0.046668,0.295562,0.062224]$ |
| m2p-r001 | 1 | 0.193060 | $[0.096530,0.077224,0.386120]$ |

$$\nabla\log L = [0.523311,\,0.412798,\,0.508361]$$

**Step 4: Prior gradient** - **di sinilah perbedaan kunci dengan MLE mulai terasa**, prior mulai "menahan":

$$\nabla\log f = -\mathbf{I}([0.422843,0.319247,0.395565] - [0,0,0]) = [-0.422843,\,-0.319247,\,-0.395565]$$

**Step 5: Posterior gradient** - suku prior yang negatif memotong sebagian besar likelihood gradient:

$$\nabla\log g = [0.523311,\,0.412798,\,0.508361] + [-0.422843,\,-0.319247,\,-0.395565] = [0.100469,\,0.093551,\,0.112796]$$

(Turun drastis dari $\approx[1.07,0.84,1.10]$ di Iterasi 1 ke $\approx[0.10,0.09,0.11]$ - inilah "rem" yang tidak dimiliki MLE)

**Step 6-8: FIM, Hessian, dan update:**

$$\mathbf{I}_S^{(L,2)} \approx \begin{bmatrix} 0.628499 & 0.166847 & 0.262771 \\ 0.166847 & 0.505539 & 0.234066 \\ 0.262771 & 0.234066 & 0.658572 \end{bmatrix}, \qquad \mathbf{H}_{MAP}^{(2)} \approx \begin{bmatrix} 1.628499 & 0.166847 & 0.262771 \\ 0.166847 & 1.505539 & 0.234066 \\ 0.262771 & 0.234066 & 1.658572 \end{bmatrix}$$

$$\Delta\hat{\boldsymbol\theta}^{(2)} \approx [0.048086,\,0.048484,\,0.053547]$$

$$\hat{\boldsymbol\theta}_2 = [0.422843,0.319247,0.395565] + [0.048086,0.048484,0.053547] \approx [0.470929,\,0.367731,\,0.449112], \quad \|\hat{\boldsymbol\theta}_2\| \approx 0.7475$$

(Step sudah jauh lebih kecil dari Iterasi 1 $\to$ 2 - resistansi prior semakin kuat seiring $\theta$ menjauh dari $\mu=0$)

**Iterasi 3** dengan $\hat{\boldsymbol\theta}_2 \approx [0.470929,\,0.367731,\,0.449112]$:

Pola yang sama berulang - likelihood gradient dan prior gradient makin dekat saling meniadakan:

$$\nabla\log L \approx [0.472933,\,0.369681,\,0.451753], \qquad \nabla\log f = -[0.470929,\,0.367731,\,0.449112]$$

$$\nabla\log g \approx [0.002004,\,0.001950,\,0.002641]$$

$$\Delta\hat{\boldsymbol\theta}^{(3)} \approx [0.000957,\,0.001033,\,0.001370], \qquad \hat{\boldsymbol\theta}_3 \approx [0.471886,\,0.368765,\,0.450482], \quad \|\Delta\hat{\boldsymbol\theta}^{(3)}\| \approx 1.96\times10^{-3}$$

**Verifikasi konvergensi:** $1.96\times10^{-3} \gg 10^{-6} \Rightarrow$ **BELUM KONVERGEN**, lanjut iterasi.

**Iterasi 4** dengan $\hat{\boldsymbol\theta}_3 \approx [0.471886,\,0.368765,\,0.450482]$:

$$\nabla\log g \approx [9.4,\,9.5,\,14.5]\times10^{-7}, \qquad \Delta\hat{\boldsymbol\theta}^{(4)} \approx [4.3,\,4.9,\,7.8]\times10^{-7}$$

$$\hat{\boldsymbol\theta}_4 \approx [0.471887,\,0.368765,\,0.450483], \quad \|\Delta\hat{\boldsymbol\theta}^{(4)}\| \approx 1.01\times10^{-6}$$

**Verifikasi konvergensi:** $1.01\times10^{-6}$ masih (tipis) di atas $10^{-6} \Rightarrow$ **BELUM KONVERGEN** (persis di ambang batas), satu iterasi lagi.

**Iterasi 5** dengan $\hat{\boldsymbol\theta}_4 \approx [0.471887,\,0.368765,\,0.450483]$:

$$\Delta\hat{\boldsymbol\theta}^{(5)} \approx 2.75\times10^{-13}, \qquad \hat{\boldsymbol\theta}_5 \approx [0.471887,\,0.368765,\,0.450483]$$

**Verifikasi konvergensi:** $2.75\times10^{-13} < 10^{-6} \Rightarrow \textbf{KONVERGEN}$ - iterasi berhenti pada iterasi ke-5.

$$\hat{\boldsymbol\theta}_{MAP} = [0.4719,\,0.3688,\,0.4505], \quad \|\hat{\boldsymbol\theta}_{MAP}\| \approx 0.7494$$

**Ringkasan mekanisme dari 5 iterasi di atas:** posterior gradient menyusut jauh lebih cepat daripada likelihood gradient saja - dari $\approx[1.07,0.84,1.10]$ (Iterasi 1, murni likelihood karena $\theta_0=\mu$) menjadi $\approx[0.10,0.09,0.11]$ (Iterasi 2) lalu $\approx[0.002,0.002,0.003]$ (Iterasi 3), karena **selisih** antara likelihood gradient (yang terus positif, mendorong ke luar) dan prior gradient (yang tumbuh negatif sebanding jarak dari $\mu$, menarik ke dalam) mengecil drastis begitu keduanya saling mendekati - inilah yang membuat MAP berhenti di titik finite $\|\theta\|\approx0.75$, alih-alih terus naik tanpa batas seperti MLE ($\|\theta\|=24.43$ setelah 100 iterasi, lihat [#3.2.3](#323-demo-3-kasus-divergen-all-correct)).

**MLE vs MAP pada divergen case (ringkasan kualitatif):**

| Aspek | MLE | MAP | Mekanisme |
|---|---|---|---|
| **Iterasi 1** | $\theta_1\approx[0.3245,0.8942,0.6531]$, norm $\approx 1.15$ (lihat [#3.2.3](#323-demo-3-kasus-divergen-all-correct)) | $\theta_1=[0.4228,0.3192,0.3956]$, norm $\approx 0.66$ (dihitung di atas) | Kedua metode punya prior gradient nol di iterasi 1 ($\theta_0=\mu=0$), tapi FIM MAP sudah $+\mathbf I$ sejak awal sehingga arah step-nya berbeda |
| **Iterasi 2-4** | $\theta$ terus naik tanpa henti, norm $\to$ puluhan | $\theta$ melambat cepat, norm $\to 0.75$ dan **konvergen di iterasi ke-5** (dihitung di atas) | Prior resistance $-\Sigma^{-1}(\theta-\mu)$ tumbuh sebanding jarak dari $\mu$, memotong likelihood gradient sampai nyaris nol |
| **Iterasi 100 (batas maksimum)** | berhenti paksa oleh iteration cap, norm $=24.43$, **tidak konvergen** (lihat [#3.2.3](#323-demo-3-kasus-divergen-all-correct)) | sudah konvergen jauh sebelumnya (iterasi ke-5), norm $=0.75$ | MLE tidak pernah mencapai kriteria $\|\Delta\theta\|<10^{-6}$; MAP mencapainya 20× lebih cepat |

(Catatan koreksi: versi sebelumnya di dokumen ini menyatakan MAP baru konvergen di sekitar "iterasi 50-100" dengan norm bertahap $0.57\to0.72\to0.75$ - klaim itu tidak akurat. Perhitungan presisi di atas menunjukkan MAP pada kasus ini **konvergen jauh lebih cepat, di iterasi ke-5**, norm-nya sudah $0.6612\to0.7475\to0.7494$ dan stabil sejak iterasi ke-3.)

**Mekanisme regularisasi (formulasi teknis):**

Pada iterasi besar di MLE dengan ${\boldsymbol\theta}$ besar:

$$\nabla\log L(\boldsymbol\theta) \to \mathbf 0^+ \quad \text{(asymptotik positif, tidak pernah melewati nol)}$$

Tidak ada suku lawan, jadi Newton step kecil tapi selalu "naik":

$$\hat{\boldsymbol\theta}_{s+1} = \hat{\boldsymbol\theta}_s + \mathbf{I}_S^{-1}\nabla\log L \quad\text{(terus naik)}$$

Dengan MAP:

$$\nabla\log g(\boldsymbol\theta) = \nabla\log L(\boldsymbol\theta) - \boldsymbol\Sigma^{-1}({\boldsymbol\theta}-\boldsymbol\mu)$$

Suku kedua **selalu negatif dan tumbuh linier** seiring $\|{\boldsymbol\theta}\|$ jauh dari ${\boldsymbol\mu}$:

$$\nabla\log g(\boldsymbol\theta) \to \nabla\log L^+ - (\text{linear term growing}) \to \text{may cross zero}$$

Pada suatu ${\boldsymbol\theta}$ finite, dua suku SALING MENIADAKAN dan terbentuk root interior. Mode posterior **selalu finite** untuk prior proper, persis seperti dijelaskan Magis & Raîche [3, p.4-5]: *"The posterior density is proper"* (untuk prior proper + likelihood).

### 4.3 Kelebihan dan Kekurangan

**Kelebihan:**
- **Tidak pernah divergen** untuk prior proper - dibuktikan langsung pada kasus yang membuat MLE
  divergen di [#4.2.3](#423-demo-3-map-meregularisasi-kasus-divergen).
- Landasan teori kuat (Bayes modal, Mislevy 1986 [4]; diverifikasi silang via Magis & Raîche [3,
  Eq.5-6, p.5]), sekaligus tetap murah komputasi - Newton-Raphson dengan FIM, sama seperti MLE,
  hanya menambah $\boldsymbol\Sigma^{-1}$ ke Hessian dan suku prior ke gradien.
- Efektif dipakai sejak round pertama CAT (start dari $\boldsymbol\mu$, bukan butuh estimasi awal
  arbitrer seperti MLE) - cocok untuk re-estimasi di awal tes ketika jumlah item masih sedikit.

**Kekurangan:**
- **Bias ke arah prior** - jika $\boldsymbol\mu$ tidak mencerminkan kemampuan examinee sebenarnya
  (mis. populasi prior salah untuk sub-grup tertentu), estimasi MAP secara sistematis tertarik ke
  $\boldsymbol\mu$, terbukti pada [#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i) (MAP $\neq$ MLE meski data sama).
- Memerlukan spesifikasi prior ($\boldsymbol\mu$, $\boldsymbol\Sigma$).

---

## 5. Expected A Posteriori (EAP)

### 5.1 Teori

EAP menghitung **rata-rata posterior** - Magis & Raîche (2012), **#2.2,
p.5-6, Eq.10-11** [3]:

$$
\hat\theta_{EAP} = \frac{\displaystyle\int_{-\infty}^{+\infty}\theta\,f(\theta)\,L(\theta)\,d\theta}{\displaystyle\int_{-\infty}^{+\infty} f(\theta)\,L(\theta)\,d\theta}
\tag{10, p.5}
$$

$$
se(\hat\theta_{EAP}) = \left[\frac{\int_{-\infty}^{+\infty}(\theta-\hat\theta_{EAP})^2f(\theta)L(\theta)d\theta}{\int_{-\infty}^{+\infty}f(\theta)L(\theta)d\theta}\right]^{1/2}
\tag{11, p.6}
$$

Sumber asli metode ini, Bock & Mislevy (1982), *"Adaptive EAP estimation of ability in a
microcomputer environment"*, Applied Psychological Measurement 6(4):431-444 [5], tidak dapat
diakses gratis untuk verifikasi halaman langsung - diverifikasi silang melalui Magis & Raîche [3]
yang secara eksplisit mengaitkan Eq.10 dengan Bock & Mislevy (1982) [3, p.5]: *"The third estimator
is the expected a posteriori (EAP) estimator (Bock and Mislevy 1982)."*

Magis & Raîche [3, p.6] menyatakan integral pada Eq.10-11 **"are approximated, for instance by
adaptive quadrature or numerical integration"** - tanpa memberi resep pasti. Kode produksi
mendekati integral dengan **kuadratur Gauss-Hermite klasik** (Bock & Mislevy 1982 [5]): titik grid
adalah akar polinomial Hermite (fisikawan) $H_{pts}$, dihitung via algoritma Golub-Welsch (1969)
[7] - nilai eigen dari matriks Jacobi tridiagonal simetris dengan diagonal nol dan off-diagonal
$\sqrt{i/2}$ - lalu digeser & diskalakan **per dimensi** ke prior $N(\mu_d,\sigma_d^2)$, dengan
$\mu_d$ dan $\sigma_d^2$ diambil dari `prior_mean`/`prior_cov_diag` yang sama dipakai MAP
([#4.1](#41-teori)) - EAP dan MAP dengan demikian selalu mengintegralkan/memaksimalkan posterior
yang identik, hanya beda cara meringkasnya (mean vs modus):

$$
\theta_{q,d} = \mu_d + \sqrt{2}\,\sigma_d\, x_q, \qquad A_q = \frac{w_q}{\sqrt\pi}, \qquad q=0,1,\dots,pts-1
$$

dengan $x_q$ = node Gauss-Hermite standar (akar $H_{pts}$) dan $w_q$ = bobot Gauss-Hermite
standarnya (dari Golub-Welsch: $w_q=\sqrt\pi\cdot v_q[0]^2$, $v_q$ = eigenvector ternormalisasi
ke-$q$). Substitusi perubahan variabel $\theta=\mu+\sqrt2\sigma x$ mengubah integral kontinu menjadi:

$$
\int f(\theta)N(\theta;\mu,\sigma^2)\,d\theta = \frac{1}{\sqrt\pi}\int f(\mu+\sqrt2\sigma x)e^{-x^2}dx \approx \sum_q A_q\, f(\theta_q)
$$

yang eksak untuk $f$ polinomial hingga derajat $2\cdot pts-1$ - bobot $A_q$ sudah mengintegralkan
densitas prior Gaussian secara analitik, sehingga $\sum_q A_q = 1$ tepat (tidak perlu evaluasi
$N(\theta_q;\mu,\sigma)$ terpisah, dan invarian terhadap $\mu,\sigma$ karena murni hasil substitusi
variabel). Menjumlahkan atas seluruh kombinasi grid $k$-dimensi ($pts^k$
titik total):

$$
\hat{\boldsymbol\theta}_{EAP} \approx \frac{\sum_{\text{grid}} \boldsymbol\theta_q \cdot L(\boldsymbol\theta_q)\cdot A(\boldsymbol\theta_q)}{\sum_{\text{grid}} L(\boldsymbol\theta_q)\cdot A(\boldsymbol\theta_q)}, \qquad A(\boldsymbol\theta_q)=\prod_{d=1}^k A_{q,d}
$$

Pola penjumlahan multi-indeks atas grid ini sama seperti teknik kuadratur Gauss-Hermite
$k$-dimensi pada Chalmers (2012), *"mirt: A Multidimensional Item Response Theory Package for the
R Environment"*, JSS 48(6), **Eq.6, p.5** [6]:
$\tilde P_\ell=\sum_{q_m}\cdots\sum_{q_1}L_\ell(\mathbf x_\ell\mid\boldsymbol\Psi,\mathbf K)\,g(K_{q1})g(K_{q2})\cdots g(K_{qm})$
- meski Eq.6 [6] dipakai untuk mengintegralkan $\theta$ sebagai *nuisance parameter* pada estimasi
parameter item (EM), bukan untuk EAP examinee individual, teknik diskretisasi grid multi-indeksnya
identik.

#### 5.1.1 Algoritma Golub-Welsch: cara menentukan grid

Grid di EAP adalah kumpulan titik-titik diskrit sepanjang sumbu θ yang dipakai untuk mengaproksimasi integral di rumus EAP (yang tadi kita bahas) secara numerik, karena integral kontinu $\int_{-\infty}^{+\infty}$ tidak bisa dihitung langsung secara komputasi, harus didekati dengan penjumlahan berhingga.

Landasan teorinya adalah **teorema kuadratur Gauss** klasik: untuk kuadratur $n$-titik terhadap fungsi
bobot $w(x)$ pada suatu domain, node yang membuat kuadratur eksak untuk polinomial berderajat setinggi
mungkin ($2n-1$) **adalah akar-akar polinomial ortogonal derajat-$n$ terhadap $w(x)$** pada domain itu.
Untuk domain $(-\infty,\infty)$ dengan $w(x)=e^{-x^2}$, polinomial ortogonalnya adalah polinomial
Hermite (fisikawan) $H_n$ - sehingga node $x_q$ = akar $H_{pts}$, persis definisi di atas. Golub &
Welsch (1969) [7] menunjukkan akar-akar ini (dan bobotnya) bisa dihitung tanpa menyelesaikan polinomial
secara aljabar - hanya lewat **dekomposisi eigen** matriks Jacobi tridiagonal simetris yang dibangun
dari relasi rekursi tiga-suku (*three-term recurrence*) milik polinomial ortogonal itu sendiri. Contoh
numerik di tiap langkah di bawah memakai $n=pts=5$, kasus yang sama dipakai di
[#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5).

**Langkah 1 - relasi rekursi menjadi matriks Jacobi.** Polinomial Hermite fisikawan memenuhi rekursi
$H_{i+1}(x)=2xH_i(x)-2iH_{i-1}(x)$. Setelah dinormalisasi menjadi fungsi ortonormal $\varphi_i$ (agar
$\int\varphi_i\varphi_j\,e^{-x^2}dx=\delta_{ij}$), rekursi ini menjadi simetris:

$$x\,\varphi_i(x) = \sqrt{\tfrac{i+1}{2}}\,\varphi_{i+1}(x) + \sqrt{\tfrac{i}{2}}\,\varphi_{i-1}(x), \qquad i=0,1,\dots,n-1$$

artinya "kalikan dengan $x$" adalah operator linear yang direpresentasikan pada basis
$\{\varphi_0,\dots,\varphi_{n-1}\}$ oleh matriks tridiagonal simetris $J_n$: diagonal nol (karena
$e^{-x^2}$ simetris di sekitar 0, tak ada suku $a_i$ yang tersisa) dan off-diagonal
$J_{i,i+1}=J_{i+1,i}=\sqrt{(i+1)/2}$ untuk $i=0,\dots,n-2$.

> **Contoh ($n=5$):** off-diagonal $J_5$ adalah $\sqrt{0.5},\sqrt1,\sqrt{1.5},\sqrt2\approx
> 0.7071,1.0,1.2247,1.4142$ (untuk $i=0,1,2,3$), sehingga
> $$J_5=\begin{pmatrix}0&0.7071&0&0&0\\0.7071&0&1&0&0\\0&1&0&1.2247&0\\0&0&1.2247&0&1.4142\\0&0&0&1.4142&0\end{pmatrix}$$
> - persis matriks `jacobi` yang dibangun di `gauss_hermite()` (`eap.rs:12-30`) sebelum didekomposisi eigen.

**Langkah 2 - eigenvalue $J_n$ = akar $H_n$ = node kuadratur.** Karena $J_n$ adalah matriks operator
"kalikan dengan $x$" pada basis $\{\varphi_i\}$, polinomial karakteristik $\det(xI-J_n)$ sebanding
dengan $\varphi_n(x)$ (dan karenanya dengan $H_n(x)$) - sehingga eigenvalue $J_n$ **adalah** akar-akar
$H_n$, tanpa perlu memfaktorkan polinomial derajat-$n$ secara eksplisit. Ini krusial untuk $n$ besar,
yang tidak lagi punya bentuk tertutup praktis seperti kasus $n=5$ di bawah.

> **Contoh ($n=5$):** dekomposisi eigen $J_5$ di atas menghasilkan node $x_q=\{0,\ \pm0.958572,\
> \pm2.020183\}$. Kebetulan untuk $n=5$ ini bisa diverifikasi tanpa alat numerik, karena
> $H_5(x)=8x(4x^4-20x^2+15)$ tereduksi ke persamaan kuadrat pada $u=x^2$:
> $$4u^2-20u+15=0 \;\Rightarrow\; u=\frac{20\pm\sqrt{160}}{8} \;\Rightarrow\; x_q=\{0,\ \pm0.958572,\ \pm2.020183\}$$
> - identik dengan eigenvalue $J_5$. Setelah diskalakan $\theta_q=\sqrt2\sigma\,x_q$ ($\sigma=1$):
> $\{0,\pm1.355626,\pm2.856970\}$, node yang dipakai di [#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5).
> Untuk $n$ berapa pun selain kasus kecil seperti ini, tidak ada bentuk tertutup lagi dan dekomposisi
> eigen numerik (`SymmetricEigen`) menjadi satu-satunya cara praktis.

**Langkah 3 - eigenvector menentukan bobot.** Teorema Golub-Welsch [7] menyatakan bobot kuadratur
$w_q=\mu_0\cdot v_{q,0}^2$, dengan $\mu_0=\int e^{-x^2}dx=\sqrt\pi$ (momen ke-0 fungsi bobot) dan
$v_{q,0}$ komponen pertama eigenvector ternormalisasi ke-$q$.

> **Contoh ($n=5$):** eigenvector ternormalisasi untuk tiap eigenvalue di Langkah 2 punya komponen
> pertama $v_{q,0}$ berikut:
>
> | $x_q$ | $v_{q,0}$ | $w_q=\sqrt\pi\cdot v_{q,0}^2$ | $A_q=w_q/\sqrt\pi=v_{q,0}^2$ |
> |---|---|---|---|
> | $0$ | $0.730297$ | $0.945309$ | $0.533333$ |
> | $\pm0.958572$ | $0.471251$ | $0.393619$ | $0.222076$ |
> | $\pm2.020183$ | $0.106098$ | $0.019953$ | $0.011257$ |
>
> ($\sum w_q=\sqrt\pi\approx1.772454$, $\sum A_q=1$ tepat) - $A_q$ inilah yang dipakai sebagai bobot
> kuadratur di [#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5).

**Keterangan variabel (tambahan untuk EAP):**

| Simbol | Arti |
|---|---|
| $\hat\theta_{EAP}$ | Estimasi = rata-rata (mean) posterior, bukan modus |
| $\theta_q$, $\boldsymbol\theta_q$ | Titik grid kuadratur Gauss-Hermite ke-$q$ (skalar/vektor), akar $H_{pts}$ digeser $\mu_d$ & diskalakan $\sqrt2\sigma_d$ per dimensi ($\mu_d,\sigma_d$ dari `prior_mean`/`prior_cov_diag`, sama seperti MAP) |
| $pts$ | Jumlah titik grid per dimensi (`eap_quad_pts`, default 21 di `engine.rs:123`) |
| $A_q$, $A(\boldsymbol\theta_q)$ | Bobot kuadratur Gauss-Hermite (sudah termasuk densitas prior), $=\prod_d A_{q,d}$, $\sum_q A_q=1$ |
| $L(\boldsymbol\theta_q)$ | Likelihood seluruh respons pada titik grid $=\prod_i P_i(\boldsymbol\theta_q)^{u_i}Q_i(\boldsymbol\theta_q)^{1-u_i}$ |

#### 5.1.2 Teori & Pembuktian: Standard Error (SE) untuk EAP

Berbeda dari MLE/MAP ([#3.1.1](#311-teori-standard-error-se-untuk-mle)/[#4.1.2](#412-teori-standard-error-se-untuk-map))
yang mengandalkan aproksimasi asimtotik-normal (invers FIM/Hessian di satu titik), SE EAP dihitung
**langsung** dari bentuk posterior aktual - definisinya sendiri adalah **standard deviation posterior**
[3, Eq.11, p.6] (dikutip di [#5.1](#51-teori)):

$$
se(\hat\theta_{EAP}) = \sqrt{\frac{\int(\theta-\hat\theta_{EAP})^2f(\theta)L(\theta)\,d\theta}{\int f(\theta)L(\theta)\,d\theta}} = \sqrt{\mathrm{Var}[\theta\mid\mathbf u]}
$$

- tidak butuh pendekatan Laplace/Gaussian di sekitar mode karena EAP mengintegralkan seluruh bentuk
posterior, bukan hanya kelengkungannya di satu titik - salah satu alasan EAP "tidak pernah divergen"
dan tetap stabil pada posterior yang skewed (lihat [#5.3](#53-kelebihan-dan-kekurangan)).

**Pembuktian aljabar (identitas momen, bukan dikutip):** variance bisa dihitung dua cara yang identik
secara aljabar - bentuk **tersentral** (Eq.11 di atas, dipakai manual di
[#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5) Step 6) dan bentuk **momen mentah**:

$$
\mathrm{Var}[\theta] = E\big[(\theta-E[\theta])^2\big] = E[\theta^2]-2\,\theta\,E[\theta]\big|_{E[\cdot]}+E[\theta]^2 = E[\theta^2]-2E[\theta]^2+E[\theta]^2
$$

$$
\boxed{\mathrm{Var}[\theta\mid\mathbf u] = E[\theta^2\mid\mathbf u] - \big(E[\theta\mid\mathbf u]\big)^2}
$$

Dalam notasi kuadratur grid (sama seperti Eq.10 di [#5.1](#51-teori)):

$$
E[\theta\mid\mathbf u] \approx \frac{\sum_q\theta_qL(\theta_q)A_q}{\sum_qL(\theta_q)A_q} = \hat\theta_{EAP}, \qquad
E[\theta^2\mid\mathbf u] \approx \frac{\sum_q\theta_q^2L(\theta_q)A_q}{\sum_qL(\theta_q)A_q}
$$

**Verifikasi numerik identitas ini** memakai angka Step 3-6 di
[#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5) DEMO 1 ($pts=5$, $den=\sum_qw_q=0.500000$,
$\hat\theta_{EAP}=0.525310$):

| $\theta_q$ | $w_q=L\cdot A_q$ | $\theta_q^2\,w_q$ |
|---|---|---|
| $-2.856970$ | 0.000153 | $8.1623\times0.000153=0.001249$ |
| $-1.355626$ | 0.025702 | $1.8377\times0.025702=0.047233$ |
| $0.000000$ | 0.266667 | $0.0000\times0.266667=0.000000$ |
| $+1.355626$ | 0.196374 | $1.8377\times0.196374=0.360934$ |
| $+2.856970$ | 0.011105 | $8.1623\times0.011105=0.090642$ |
| **sum** | 0.500000 | **0.500058** |

$$
E[\theta^2] = \frac{0.500058}{0.500000} = 1.000116, \qquad E[\theta]^2 = 0.525310^2 = 0.275951
$$

$$
\mathrm{Var}[\theta] = E[\theta^2]-E[\theta]^2 = 1.000116 - 0.275951 = 0.724165 \qquad\Rightarrow\qquad se(\hat\theta_{EAP}) = \sqrt{0.724165} \approx 0.850979
$$

- identik (selisih $6.8\times10^{-5}$ murni akibat pembulatan tampilan 6-desimal pada $w_q$, bukan
kesalahan aljabar) dengan hasil bentuk tersentral di [#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5)
Step 6: $se(\hat\theta_{EAP})=0.850911$ - **membuktikan kedua bentuk identik secara aljabar**.

**Identik dengan kode produksi:** `eap.rs:50-110` mengakumulasi **tiga** penjumlah sekaligus dalam
satu pass atas grid ($num=\sum\theta_qw_q$, $num2=\sum\theta_q^2w_q$, $den=\sum w_q$), lalu menghitung
`variance = num2[d]/den - mean*mean` (`eap.rs:103`) - persis identitas momen-mentah yang dibuktikan
di atas, dipilih **karena efisiensi**: satu pass kuadratur $pts^k$-titik, bukan dua (pass pertama untuk
$\hat\theta_{EAP}$, pass kedua untuk $(\theta-\hat\theta_{EAP})^2$ seperti bentuk tersentral manual).
`posterior_se` yang dihasilkan (`None` hanya bila `den` terlalu kecil untuk dinormalisasi, mis. seluruh
titik grid punya likelihood yang collapse ke nol) inilah yang dipakai `McatEngine::compute_se`
(`engine.rs:190-196`, cabang `EstimationMethod::Eap`) sebagai SE EAP - **tanpa** jatuh ke formula
gaya-MAP kecuali `eap_posterior_se` benar-benar `None` (lihat komentar `engine.rs:167-172`).

### 5.2 Perhitungan Manual

#### 5.2.1 DEMO 1: Reproduksi 1 Dimensi (pts=5)

Item & prior:
- Item: $a=1.5, d=0, c=0$ (M2PL, diskriminasi 1.5, tanpa difficulty/guessing)
- Respons: benar ($u=1$)
- Prior: $\pi(\theta) = N(0,1)$ (normal standard, $\sigma=1$)
- Grid Gauss-Hermite: $pts=5$ node $\theta_q=\sqrt2\cdot x_q = [-2.856970, -1.355626, 0.000000, +1.355626, +2.856970]$
  (akar $H_5$ diskalakan $\sqrt2\sigma$, dihitung via Golub-Welsch - lihat [#5.1.1](#511-algoritma-golub-welsch-cara-menentukan-grid))
- Bobot kuadratur $A_q = w_q/\sqrt\pi = [0.011257, 0.222076, 0.533333, 0.222076, 0.011257]$
  (dari eigenvector $J_5$ - lihat contoh Langkah 3 di [#5.1.1](#511-algoritma-golub-welsch-cara-menentukan-grid));
  $\sum_q A_q = 1.000000$ tepat - bobot ini sudah mengintegralkan $N(\theta;0,1)$ secara analitik

**Step 0: Turunkan grid & bobot kuadratur ($n=pts=5$, Golub-Welsch - detail teori di [#5.1.1](#511-algoritma-golub-welsch-cara-menentukan-grid))**

*0a. Matriks Jacobi $J_5$* - diagonal 0, off-diagonal $\sqrt{(i+1)/2}$ untuk $i=0,1,2,3$
($\sqrt{0.5},\sqrt1,\sqrt{1.5},\sqrt2\approx0.7071,1.0,1.2247,1.4142$):

$$J_5=\begin{pmatrix}0&0.7071&0&0&0\\0.7071&0&1&0&0\\0&1&0&1.2247&0\\0&0&1.2247&0&1.4142\\0&0&0&1.4142&0\end{pmatrix}$$

*0b. Eigenvalue $J_5$ = node $x_q$* - dekomposisi eigen matriks di atas (untuk $n=5$ bisa diverifikasi
lewat reduksi $H_5(x)=8x(4x^4-20x^2+15)$ ke persamaan kuadrat $4u^2-20u+15=0$ pada $u=x^2$) memberi

$$x_q=\Big\{0,\ \pm0.958572,\ \pm2.020183\Big\}$$

lalu diskalakan ke domain $\theta$ lewat $\theta_q=\sqrt2\,\sigma\,x_q$ ($\sigma=1$ dari prior $N(0,1)$):

$$\theta_q=[-2.856970,\ -1.355626,\ 0.000000,\ +1.355626,\ +2.856970]$$

*0c. Eigenvector $J_5$ = bobot $w_q$, lalu normalisasi jadi $A_q$* - komponen pertama $v_{q,0}$ dari
eigenvector ternormalisasi tiap eigenvalue di atas:

| $x_q$ | $v_{q,0}$ | $w_q=\sqrt\pi\cdot v_{q,0}^2$ | $A_q=w_q/\sqrt\pi=v_{q,0}^2$ |
|---|---|---|---|
| $0$ | $0.730297$ | $0.945309$ | $0.533333$ |
| $\pm0.958572$ | $0.471251$ | $0.393619$ | $0.222076$ |
| $\pm2.020183$ | $0.106098$ | $0.019953$ | $0.011257$ |

($\sum w_q=\sqrt\pi\approx1.772454$; $\sum A_q=1$ tepat - lolos cek normalisasi)

Menggabungkan 0b & 0c menurut urutan $\theta_q$, inilah grid & bobot yang dipakai di Step 1-4:

| $\theta_q$ | $-2.856970$ | $-1.355626$ | $0.000000$ | $+1.355626$ | $+2.856970$ |
|---|---|---|---|---|---|
| $A_q$ | $0.011257$ | $0.222076$ | $0.533333$ | $0.222076$ | $0.011257$ |

**Step 1: Hitung likelihood $L(\theta_q) = P(\theta_q)^u \cdot Q(\theta_q)^{1-u}$ untuk setiap titik grid**

Dengan $z_q = a\theta_q + d = 1.5\theta_q + 0 = 1.5\theta_q$ dan $P(\theta_q) = \sigma(z_q) = \frac{1}{1+e^{-1.5\theta_q}}$:

| $\theta_q$ | $z_q = 1.5\theta_q$ | $P(\theta_q) = \sigma(z_q)$ | $L(\theta_q) = P^1 = P$ |
|---|---|---|---|
| $-2.856970$ | $-4.285455$ | $0.013580$ | **0.013580** |
| $-1.355626$ | $-2.033439$ | $0.115736$ | **0.115736** |
| $0.000000$ | $0.000000$ | $0.500000$ | **0.500000** |
| $+1.355626$ | $+2.033439$ | $0.884264$ | **0.884264** |
| $+2.856970$ | $+4.285455$ | $0.986420$ | **0.986420** |

**Step 2: Bobot kuadratur $A_q$ (sudah menggantikan evaluasi densitas prior terpisah)**

Karena node/bobot Gauss-Hermite dibangun tepat untuk mengintegralkan terhadap $e^{-x^2}$ - yang
setelah substitusi $\theta=\sqrt2\sigma x$ menjadi $N(\theta;0,\sigma^2)$ - $A_q$ **sudah**
merepresentasikan "densitas prior $\times$ lebar kuadratur" tanpa perlu evaluasi $N(\theta_q;0,1)$
secara terpisah (lihat [#5.1](#51-teori)):

| $\theta_q$ | $A_q = w_q/\sqrt\pi$ |
|---|---|
| $-2.856970$ | **0.011257** |
| $-1.355626$ | **0.222076** |
| $0.000000$ | **0.533333** |
| $+1.355626$ | **0.222076** |
| $+2.856970$ | **0.011257** |

**Step 3: Hitung weight $w_q = L(\theta_q) \times A_q$ untuk setiap titik**

| $\theta_q$ | $L(\theta_q)$ | $A_q$ | $w_q = L \times A$ |
|---|---|---|---|
| $-2.856970$ | 0.013580 | 0.011257 | $0.013580 \times 0.011257 =$ **0.000153** |
| $-1.355626$ | 0.115736 | 0.222076 | $0.115736 \times 0.222076 =$ **0.025702** |
| $0.000000$ | 0.500000 | 0.533333 | $0.500000 \times 0.533333 =$ **0.266667** |
| $+1.355626$ | 0.884264 | 0.222076 | $0.884264 \times 0.222076 =$ **0.196374** |
| $+2.856970$ | 0.986420 | 0.011257 | $0.986420 \times 0.011257 =$ **0.011105** |
| **sum** | | | **0.500000** |

(Total tepat $0.500000$ - bukan kebetulan: dengan node/bobot Gauss-Hermite simetris dan
$P(-\theta)=1-P(\theta)=Q(\theta)$ (simetri fungsi logistik), $\sum_q A_q P(\theta_q)$ selalu sama
persis dengan $\sum_q A_q Q(\theta_q)$, dan keduanya berjumlah $\sum_q A_q=1$ - jadi masing-masing
tepat $0.5$, terlepas dari nilai $a$. Ini konsekuensi struktural dari kuadratur yang genuinely
ternormalisasi, berbeda dari grid berjarak-sama yang totalnya sembarang.)

**Step 4: Hitung numerator $\sum \theta_q w_q$ (momen posterior pertama)**

| $\theta_q$ | $w_q$ | $\theta_q \times w_q$ |
|---|---|---|
| $-2.856970$ | 0.000153 | $-2.856970 \times 0.000153 = -0.000437$ |
| $-1.355626$ | 0.025702 | $-1.355626 \times 0.025702 = -0.034843$ |
| $0.000000$ | 0.266667 | $0.000000 \times 0.266667 = 0.000000$ |
| $+1.355626$ | 0.196374 | $+1.355626 \times 0.196374 = +0.266209$ |
| $+2.856970$ | 0.011105 | $+2.856970 \times 0.011105 = +0.031725$ |
| **sum** | | **0.262655** |

**Step 5: Hitung EAP (rata-rata posterior)**

$$\hat\theta_{EAP} = \frac{\sum_q \theta_q w_q}{\sum_q w_q} = \frac{0.262655}{0.500000} = 0.525310$$

**Step 6: Hitung standard error (opsional, dari Eq.11 [#3.1](#31-teori))**

Hitung momen kedua:

| $\theta_q$ | $w_q$ | $(\theta_q - \hat\theta_{EAP})^2 \times w_q$ |
|---|---|---|
| $-2.856970$ | 0.000153 | $(-2.856970 - 0.5253)^2 \times 0.000153 = 11.4398 \times 0.000153 = 0.001749$ |
| $-1.355626$ | 0.025702 | $(-1.355626 - 0.5253)^2 \times 0.025702 = 3.5379 \times 0.025702 = 0.090933$ |
| $0.000000$ | 0.266667 | $(0.000000 - 0.5253)^2 \times 0.266667 = 0.2760 \times 0.266667 = 0.073587$ |
| $+1.355626$ | 0.196374 | $(+1.355626 - 0.5253)^2 \times 0.196374 = 0.6894 \times 0.196374 = 0.135385$ |
| $+2.856970$ | 0.011105 | $(+2.856970 - 0.5253)^2 \times 0.011105 = 5.4366 \times 0.011105 = 0.060371$ |
| **sum** | | **0.362025** |

$$se(\hat\theta_{EAP}) = \sqrt{\frac{0.362025}{0.500000}} = \sqrt{0.724049} = 0.850911$$

Simetri mean prior (0) ditambah likelihood yang lebih terkonsentrasi di dekat 0 menghasilkan EAP
yang moderat ($0.525$) dibanding MLE yang divergen atau MAP yang shrink lebih dalam untuk kasus
serupa.

#### 5.2.2 DEMO 2: Multidimensional (k=3), 7 item, grid 5^3=125 titik

Bank item & respons identik [#5.2.2](#322-demo-2-multidimensional-k3-7-item)/[#4.2.2](#422-demo-2-multidimensional-k3-7-item-prior-n0i). Prior $\pi(\boldsymbol\theta)=N(\mathbf 0,\mathbf I)$ (multivariate normal dengan mean $[0,0,0]$ dan variance $[1,1,1]$).

Grid Gauss-Hermite per dimensi: $pts=5$ node $= [-2.856970, -1.355626, 0.000000, +1.355626, +2.856970]$,
bobot $A_q = [0.011257, 0.222076, 0.533333, 0.222076, 0.011257]$, total kombinasi $5^3=125$ titik.

**Step 1: Diskripsi grid & struktur perhitungan**

Integrasi EAP atas posterior multidimensi dilakukan via grid rectangular (Cartesian product) node
Gauss-Hermite:

$$\hat{\boldsymbol\theta}_{EAP} = \frac{\sum_{q_1=1}^{5}\sum_{q_2=1}^{5}\sum_{q_3=1}^{5} \boldsymbol\theta_q L(\boldsymbol\theta_q)A(\boldsymbol\theta_q)}{\sum_{q_1=1}^{5}\sum_{q_2=1}^{5}\sum_{q_3=1}^{5} L(\boldsymbol\theta_q)A(\boldsymbol\theta_q)}$$

Karena prior multivariate normal dengan $\boldsymbol\Sigma=\mathbf{I}$ (diagonal & independen), bobot
kuadratur juga terfaktorisasi per dimensi:

$$A(\boldsymbol\theta_q) = A(\theta_{q,1})A(\theta_{q,2})A(\theta_{q,3})$$

Bobot $A_q$ per dimensi dapat di-*cache* sebelum grid kombinasi - menghemat perhitungan.

**Kenapa hasilnya 125?** Tiap dimensi ($\theta_1,\theta_2,\theta_3$) punya 5 pilihan titik yang sama
persis (5 node dari [#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5)). Untuk bikin 1 titik grid
3-dimensi, kita ambil **1 pilihan dari dimensi-1, 1 pilihan dari dimensi-2, dan 1 pilihan dari
dimensi-3 - dan ketiganya bebas, tidak saling bergantung**.

Analoginya: bayangkan memilih baju (5 pilihan warna), celana (5 pilihan model), dan sepatu (5
pilihan jenis) untuk bikin 1 "outfit". Tiap kombinasi baju+celana+sepatu adalah outfit yang berbeda.
Karena tiap potongan pakaian dipilih bebas dari yang lain, total outfit = $5\times5\times5=125$ -
**bukan** $5+5+5=15$ (itu kalau cuma pilih salah satu jenis pakaian saja, bukan gabungan ketiganya).

Sama persis logikanya di sini: 5 pilihan untuk $\theta_1$, dikali 5 pilihan untuk $\theta_2$, dikali
5 pilihan untuk $\theta_3$ = 125 titik grid total. Ketiga dimensi memakai daftar 5 node yang identik
(node index $1,2,3,4,5$):
$$\text{nodes} = [-2.856970,\ -1.355626,\ 0.000000,\ +1.355626,\ +2.856970]$$
Contoh sebagian kecil (bukan semua 125) untuk menunjukkan pola indeks
$(q_1,q_2,q_3)\to[\text{nodes}[q_1],\text{nodes}[q_2],\text{nodes}[q_3]]$:

| $(q_1,q_2,q_3)$ | $\boldsymbol\theta_q$ | Posisi |
|---|---|---|
| $(1,1,1)$ | $[-2.857,-2.857,-2.857]$ | pojok kubus (node terkecil di ketiga dimensi) |
| $(5,5,5)$ | $[+2.857,+2.857,+2.857]$ | pojok kubus berlawanan |
| $(3,3,3)$ | $[0,0,0]$ | pusat grid (node tengah di ketiga dimensi) |
| $(2,3,3)$ | $[-1.356,0,0]$ | dekat pusat, bergeser hanya di dimensi-1 |
| $(5,1,5)$ | $[+2.857,-2.857,+2.857]$ | salah satu pojok campuran tanda |

Untuk melihat/mengecek keseluruhan 125 kombinasi, cukup jalankan 3 loop bersarang `for q1 in 0..5 { for
q2 in 0..5 { for q3 in 0..5 { ... } } }` - inilah yang dilakukan kode produksi di `eap.rs` (loop atas
$pts^k$ titik). Tabel Step 2 di bawah hanya mengambil 12 dari 125 titik ini sebagai sampel ilustrasi.

**Step 2: Sampel titik grid (12 dari 125 titik, untuk ilustrasi)**

| $\boldsymbol\theta_q = [\theta_{q,1}, \theta_{q,2}, \theta_{q,3}]$ | $L(\boldsymbol\theta_q)$ | $A(\boldsymbol\theta_q) = \prod_k A_{q,k}$ | $w_q = L \times A$ | Keterangan |
|---|---|---|---|---|
| $[-2.857,-2.857,-2.857]$ | $\approx 0$ | $0.000001$ | $\approx 0$ | Sudut ekstrem |
| $[-2.857,-1.356,0]$ | $0.000010$ | $0.001333$ | $\approx 0$ | Edge |
| $[-1.356,-1.356,-1.356]$ | $0.000004$ | $0.010952$ | $\approx 0$ | Sudut sedang |
| $[-1.356,0,0]$ | $0.001919$ | $0.063168$ | 0.00012120 | |
| $[0,0,0]$ | 0.007464 | 0.151704 | **0.00113237** | **Pusat grid (mode-like)** |
| $[0,0,+1.356]$ | 0.001583 | 0.063168 | 0.00010001 | |
| $[+1.356,0,0]$ | 0.001951 | 0.063168 | 0.00012323 | |
| $[+1.356,+1.356,+1.356]$ | 0.000006 | 0.010952 | $\approx 0$ | Sudut sedang |
| $[+1.356,+1.356,0]$ | 0.000101 | 0.026303 | 0.00000265 | |
| $[+2.857,-2.857,+2.857]$ | $\approx 0$ | $0.000001$ | $\approx 0$ | Sudut ekstrem |
| $[+2.857,+2.857,-2.857]$ | $\approx 0$ | $0.000001$ | $\approx 0$ | Sudut ekstrem |
| $[+2.857,+2.857,+2.857]$ | $\approx 0$ | $0.000001$ | $\approx 0$ | Sudut ekstrem |
| *113 titik lain* | *..* | *..* | *..* | *Dikerjakan via loop* |

**Step 3: Agregasi keseluruhan 125 titik** (dihitung via loop, hasil final):

Denominator (integral posterior):
$$\sum_{\text{125 titik}} w_q = \sum_{q_1}\sum_{q_2}\sum_{q_3} L(\boldsymbol\theta_q)A(\boldsymbol\theta_q) = 0.00219048$$

Numerator (weighted mean):
$$\sum_{\text{125 titik}} \boldsymbol\theta_q w_q = \begin{bmatrix} 0.0000667 \\ -0.0006605 \\ 0.0002937 \end{bmatrix}$$

(Pusat grid di [0,0,0] mendominasi bobot karena likelihood terkuat di dekat sana, dan prior simetris)

**Step 4: Hitung EAP dengan grid pts=5:**

$$\hat{\boldsymbol\theta}_{EAP}^{(pts=5)} = \frac{[0.0000667, -0.0006605, 0.0002937]}{0.00219048} = [0.03047, -0.30155, 0.13408]$$

Variance per dimensi (untuk SE):

$$\text{Var}_k = \frac{\sum_q (\theta_{q,k} - \hat\theta_{EAP,k})^2 w_q}{\sum_q w_q}$$

Dilakukan per dimensi dengan tabel momen kedua, hasil (dari API):
$$se(\hat{\boldsymbol\theta}_{EAP}^{(pts=5)}) \approx [0.6106, 0.6635, 0.5311]$$

**Step 5: Ulangi dengan grid lebih halus pts=21**

Grid Gauss-Hermite per dimensi: 21 node (akar $H_{21}$, diskalakan $\sqrt2\sigma$) - tidak
berjarak sama seperti grid linear, node lebih rapat di dekat 0 dan merenggang menuju ekor.

Total kombinasi: $21^3 = 9261$ titik.

Integrasi dilakukan dengan prosedur identik (tetapi 9261 kali lebih banyak perhitungan):

$$\sum_{\text{9261 titik}} w_q \approx 0.00204330$$

$$\sum_{\text{9261 titik}} \boldsymbol\theta_q w_q \approx [0.0000660, -0.0007440, 0.0003728]$$

$$\hat{\boldsymbol\theta}_{EAP}^{(pts=21)} = \frac{[0.0000660, -0.0007440, 0.0003728]}{0.00204330} = [0.03229, -0.36412, 0.18244]$$

SE per dimensi (dari API):
$$se(\hat{\boldsymbol\theta}_{EAP}^{(pts=21)}) \approx [0.6751, 0.6679, 0.6099]$$

**Perbandingan 3 metode pada data identik:**

| Metode | $\hat\theta_{verbal}$ | $\hat\theta_{numeric}$ | $\hat\theta_{reasoning}$ |
|---|---|---|---|
| **MLE** (no prior) | 0.0380 | -0.6537 | 0.2966 |
| **MAP** ($\Sigma=I$) | 0.0105 | -0.3656 | 0.1340 |
| **EAP** ($pts=5$, grid kasar) | 0.0305 | -0.3016 | 0.1341 |
| **EAP** ($pts=21$) | 0.0323 | -0.3641 | 0.1824 |

**Analisis konvergensi EAP → MAP:**

- Grid $pts=5$ (125 titik): numeric $-0.3016$ mendekati MAP $-0.3656$ (selisih 17.5%)
- Grid $pts=21$ (9261 titik): numeric $-0.3641$ sangat dekat ke MAP $-0.3656$ (selisih 0.4%)

Sesuai teori: EAP dan MAP mengintegralkan/memaksimalkan **posterior yang sama** $g(\boldsymbol\theta) = f(\boldsymbol\theta)L(\boldsymbol\theta)$, dan estimasi EAP **konvergen ke MAP** seiring resolusi grid $pts\to\infty$ (integral numerik $\to$ integral kontinyu). Karena kuadratur Gauss-Hermite eksak untuk fungsi polinomial hingga derajat $2\cdot pts-1$, konvergensinya jauh lebih cepat per titik dibanding grid linear naif - selisih $pts=5$ terhadap MAP di sini (17.5%) sudah jauh lebih kecil daripada yang didapat grid berjarak-sama pada resolusi setara.

> **Versi sederhana:** MAP mencari **puncak** kurva posterior, EAP menghitung **rata-rata** kurva posterior yang sama. Kalau kurvanya berbentuk lonceng simetris, puncak dan rata-rata letaknya hampir sama - makanya EAP ≈ MAP. EAP sendiri dihitung pakai grid (sampel titik), jadi bukan integral asli yang mulus - makin banyak titik grid ($pts$ makin besar), makin dekat hasil EAP ke integral "sebenarnya", dan makin dekat pula ke MAP. Buktinya di angka di atas: $pts=5$ bedanya 17.5% dari MAP, $pts=21$ tinggal 0.4%. Gauss-Hermite juga irit - beda dari grid biasa (jarak sama rata), sedikit titik saja sudah cukup akurat.

Grid $pts=5$ sengaja dibuat kasar di atas supaya semua 125 titik bisa ditampilkan & dipahami secara manual; produksi menggunakan pts=21 untuk presisi yang wajar.

#### 5.2.3 DEMO 3: EAP Tidak Pernah Divergen

Data identik [#5.2.3](#323-demo-3-kasus-divergen-all-correct)/[#4.2.3](#423-demo-3-map-meregularisasi-kasus-divergen) (3 item, $\mathbf u=[1,1,1]$, prior $N(\mathbf 0,\mathbf I)$, $pts=21$):

| Item | $\mathbf{a}$ | $d$ | $u$ |
|---|---|---|---|
| m2p-v001 | [1.9,0.2,0.3] | 0.40 | **1** |
| m2p-n001 | [0.3,1.9,0.4] | 0.80 | **1** |
| m2p-r001 | [0.5,0.4,2.0] | 0.30 | **1** |

**Hasil final:**

| Metode | $\hat{\boldsymbol\theta}$ | $\|\hat\theta\|$ | Status |
|---|---|---|---|
| MLE | $[16.065,\,14.291,\,11.594]$ | 24.43 | **Divergen** (stop after 100 iter) |
| MAP | $[0.472,\,0.369,\,0.450]$ | 0.75 | **Finite** (regularized by prior) |
| EAP | $[0.584,\,0.480,\,0.564]$ | 0.94 | **Finite by construction** |

**Penjelasan mengapa EAP tetap finite:**

Berbeda dengan MAP yang mengatasi divergen melalui **mekanisme regularisasi dinamis** (prior gradient menarik balik), EAP tetap finite untuk **alasan struktural**:

1. **Integral atas domain terbatas**: Eq.10 [#3.1](#31-teori) dihitung hanya atas grid Gauss-Hermite
   $pts$ titik per dimensi - domainnya adalah rentang akar $H_{pts}$ (diskalakan $\sqrt2\sigma$), yang
   untuk $pts=21$ berarti $|\theta_q| \leq 7.849$ per dimensi (bukan $\pm3\sigma$ tetap - rentang node
   terluar Gauss-Hermite melebar seiring $pts$ membesar, tapi tetap **terbatas** untuk $pts$ berapa pun)

2. **Pembilang & penyebut selalu finite**: 
   - Penyebut: $\sum_q w_q = \sum_q L(\boldsymbol\theta_q)A(\boldsymbol\theta_q)$ adalah jumlah terbatas nilai-nilai finite
   - Pembilang: $\sum_q \boldsymbol\theta_q w_q$ juga terbatas karena $|\boldsymbol\theta_q| \leq 7.849$ di grid ($pts=21$), dan bobot $w_q$ terbatas

3. **Tidak ada iterasi divergen**: Tidak seperti MLE/MAP yang involve Newton-Raphson iteratif dengan potensi loop tak-terbatas, EAP adalah **komputasi satu-pass** (sekali jalan grid, langsung dapat hasil)

**Step-by-step komputasi EAP untuk kasus all-correct:**

**Step 1: Evaluasi likelihood di titik ilustrasi (bukan node Gauss-Hermite literal, dipilih sebagai
angka bulat untuk menunjukkan pola secara jelas)**

| $\boldsymbol\theta_q$ | $z_1=\mathbf{a}_1\cdot\boldsymbol\theta_q+d_1$ | $z_2=\mathbf{a}_2\cdot\boldsymbol\theta_q+d_2$ | $z_3=\mathbf{a}_3\cdot\boldsymbol\theta_q+d_3$ | $L(\boldsymbol\theta_q)=\prod P_i^{u_i}Q_i^{1-u_i}$ |
|---|---|---|---|---|
| $[0,0,0]$ | $0+0.40=0.40$ | $0+0.80=0.80$ | $0+0.30=0.30$ | $0.5987 \times 0.6900 \times 0.5744 = 0.2374$ |
| $[1,1,1]$ | $1.9+0.2+0.3+0.40=2.80$ | $0.3+1.9+0.4+0.80=3.40$ | $0.5+0.4+2.0+0.30=3.20$ | $\sigma(2.80) \times \sigma(3.40) \times \sigma(3.20) = 0.9436 \times 0.9669 \times 0.9608 = 0.8786$ |
| $[2,2,2]$ | $3.8+0.4+0.6+0.40=5.20$ | $0.6+3.8+0.8+0.80=6.00$ | $1.0+0.8+4.0+0.30=6.10$ | $0.9945 \times 0.9975 \times 0.9978 = 0.9898$ |
| $[3,3,3]$ | $5.7+0.6+0.9+0.40=7.60$ | $0.9+5.7+1.2+0.80=8.60$ | $1.5+1.2+6.0+0.30=9.00$ | $0.9995 \times 0.9998 \times 0.9999 = 0.9992$ |

Perhatian: **Semua likelihood positif dan terbatas** - tidak ada yang eksplosi menuju infinity

**Step 2: Evaluasi prior di grid (multivariate normal $N(\mathbf 0,\mathbf I)$)**

Prior presisi (independent per dimensi):

| $\boldsymbol\theta_q$ | $\pi(\boldsymbol\theta_q) = \prod_k \pi(\theta_{q,k})$ | Bobot |
|---|---|---|
| $[0,0,0]$ | $0.3989^3 = 0.0635$ | **Tertinggi** (di mean prior) |
| $[1,1,1]$ | $(0.3989 \times e^{-0.5})^3 = (0.2420)^3 = 0.0142$ | Sedang |
| $[2,2,2]$ | $(0.3989 \times e^{-2})^3 = (0.0540)^3 = 0.000157$ | Kecil |
| $[3,3,3]$ | $(0.3989 \times e^{-4.5})^3 = (0.0066)^3 = 0.000000287$ | Sangat kecil |

**Step 3: Hitung weight untuk setiap titik $w_q = L(\boldsymbol\theta_q) \times \pi(\boldsymbol\theta_q)$**

| $\boldsymbol\theta_q$ | $L(\boldsymbol\theta_q)$ | $\pi(\boldsymbol\theta_q)$ | $w_q$ |
|---|---|---|---|
| $[0,0,0]$ | 0.2374 | 0.0635 | $0.01507$ |
| $[1,1,1]$ | 0.8786 | 0.0142 | $0.01247$ |
| $[2,2,2]$ | 0.9898 | 0.000157 | $0.000155$ |
| $[3,3,3]$ | 0.9992 | 0.000000287 | $0.000000287$ |
| *116 titik lain* (kombinasi campuran di grid 21×21×21) | | | *..* |

**Pola penting:** Meskipun likelihood meningkat dengan $\|\boldsymbol\theta\|$ (semua benar → push ke infinity di MLE), **prior bobot menurun eksponensial**. Hasil: **produk keduanya (posterior weight) mencapai peak di titik intermediate**, bukan di infinity.

**Step 4: Agregasi integral untuk semua 21³=9261 titik grid**

Denominator (normalisasi posterior):
$$Z = \sum_{q=1}^{9261} w_q = 0.244901 \quad \text{(terbatas dan well-defined)}$$

Numerator (weighted mean):
$$\sum_{q=1}^{9261} \boldsymbol\theta_q w_q = [0.142995, 0.117633, 0.138192]$$

(Lebih rendah dari MAP karena likelihood penuh, tapi prior "tarik balik" juga kuat)

**Step 5: Hitung EAP**

$$\hat{\boldsymbol\theta}_{EAP} = \frac{[0.142995, 0.117633, 0.138192]}{0.244901} = [0.5839, 0.4803, 0.5643]$$

SE per dimensi (variance posterior):

$$\text{Var}_k = \frac{\sum_q (\theta_{q,k} - \hat\theta_{EAP,k})^2 w_q}{Z}$$

Dilakukan dengan tabel momen kedua pada 9261 titik, hasil: $se \approx [0.8320, 0.8521, 0.8331]$

**Perbandingan tiga metode pada all-correct:**

| Aspek | MLE | MAP | EAP |
|---|---|---|---|
| **Hasil** | $[16.07, 14.29, 11.59]$ | $[0.472, 0.369, 0.450]$ | $[0.584, 0.480, 0.564]$ |
| **Norm** | 24.43 | 0.75 | 0.94 |
| **Mekanisme finite** | **DIVERGEN** | Regularisasi dinamis (prior gradient) | Struktur integral (domain terbatas) |
| **SE** | N/A (divergen) | $\approx [0.4, 0.4, 0.4]$ | $[0.83, 0.85, 0.83]$ |
| **Interpretasi** | Tidak berguna | Over-regularized? | **Moderat, interpretabel** |

**Kesimpulan structural:**

EAP finite **bukan karena prior memberi penalti** (seperti MAP), melainkan **karena integral numerik atas domain terbatas** adalah operasi yang fundamentally terbatas. Posterior dihitung sebagai:

$$g(\boldsymbol\theta) = L(\boldsymbol\theta) \times \pi(\boldsymbol\theta)$$

Walaupun $L$ bisa naik monoton menuju 1 (pola semua-benar), **prior $\pi$ menurun eksponensial** menjauh dari mean $\mu$. Hasil perkalian adalah **distribusi yang terkonsentrasi**. Integrasi atas grid terbatas $[-3\sigma,3\sigma]^k$ otomatis menghasilkan integral yang finite dan well-defined untuk **pola respons apa pun** - tidak ada kasus patologi seperti divergen MLE atau over-shrinkage MAP.

### 5.3 Kelebihan dan Kekurangan

**Kelebihan:**
- **Tidak pernah divergen**, untuk alasan yang lebih fundamental dari MAP: bukan hasil regularisasi
  optimasi, melainkan sifat integral pada domain terbatas - dibuktikan di [#5.2.3](#523-demo-3-eap-tidak-pernah-divergen).
- Tidak butuh titik awal/iterasi Newton-Raphson sama sekali (tidak ada risiko konvergen ke
  maksimum lokal yang salah, tidak seperti MLE/MAP) - estimasi dihitung langsung dari satu kali
  penjumlahan grid.
- **Kuadratur Gauss-Hermite eksak untuk fungsi polinomial hingga derajat $2\cdot pts-1$** (Golub &
  Welsch 1969 [7], detail algoritma di [#5.1](#51-teori)), sehingga bobotnya $A_q$ sudah mengintegralkan densitas prior Gaussian secara
  analitik dan otomatis ternormalisasi ($\sum_q A_q=1$ tepat - dibuktikan di [#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5) Step 3).

  > **Versi paling sederhana:** bayangkan mau menimbang berapa "berat" tiap titik grid dalam kurva lonceng (prior). Cara naif: pakai grid jarak-sama rata terus dikira-kira beratnya - butuh banyak sekali titik biar hasilnya halus dan akurat. Gauss-Hermite lebih pintar: posisi & berat titiknya sudah dihitung lewat rumus matematis khusus yang "cocok" dengan bentuk kurva lonceng itu, jadi:
  > - **Otomatis pas** - total semua berat pasti tepat 1 (100%), tanpa perlu dicek/dikoreksi lagi.
  > - **Hemat titik** - 5 titik saja lewat Gauss-Hermite bisa seakurat puluhan/ratusan titik lewat grid biasa.
  >
  > Grid biasa  itu seperti menimbang badan dengan banyak titik sample sembarang lalu dirata-rata (butuh banyak sample biar akurat). Gauss-Hermite itu seperti sudah tahu persis di titik mana harus menimbang dan berapa bobot masing-masing, supaya hasilnya presisi walau sample-nya sedikit.
  >
  > "Grid biasa" (atau disebut juga grid linear/naif) itu cara paling sederhana bikin titik-titik sampel: titik-titiknya berjarak sama rata, seperti mistar/penggaris. Contoh: kalau mau bikin 5 titik grid di rentang $[-3, 3]$, grid biasa tinggal bagi rata: $$-3, -1.5, 0, 1.5, 3$$ Jaraknya sama semua (1.5), dan bobot tiap titik juga biasanya disamakan

**Kekurangan:**
- **Akurasi tetap bergantung pada resolusi grid** $pts$ (meski konvergensinya lebih cepat per titik
  dari grid linear) - dibuktikan di [#5.2.2](#522-demo-2-multidimensional-k3-7-item-grid-53125-titik): $pts=5$ vs
  $pts=21$ menghasilkan estimasi yang masih berbeda pada dimensi reasoning ($0.134$ vs
  $0.182$). Biaya komputasi tumbuh $pts^k$ - untuk $k=3$, $pts=21$ berarti $9261$ evaluasi
  likelihood per estimasi, jauh lebih mahal dari MLE/MAP (~3-5 iterasi Newton).
- **Node/bobot tidak berbentuk tertutup** - tidak ada rumus aljabar sederhana yang langsung
  menghasilkan posisi titik grid dan bobotnya (beda dengan grid berjarak sama, yang tinggal dihitung
  dari pembagian rentang secara langsung). Untuk $pts$ berapa pun selain kasus kecil tertentu,
  satu-satunya cara praktis menghitungnya adalah lewat **dekomposisi nilai eigen** (algoritma
  Golub-Welsch [7]) - proses numerik yang lebih rumit dan sedikit lebih mahal dibanding sekadar
  membuat titik-titik berjarak sama. Perhitungan ini diulang setiap kali estimasi EAP dijalankan
  (meski biayanya kecil untuk $pts$ yang tidak terlalu besar, dan hasilnya sebenarnya bisa disimpan
  untuk dipakai ulang selama $pts$ dan bentuk prior tidak berubah).

  > Kalau grid biasa (jarak sama rata), posisi titiknya gampang dihitung sendiri pakai kalkulator - tinggal bagi rentang jadi beberapa bagian sama besar. Grid Gauss-Hermite **tidak bisa** dihitung sesimpel itu - tidak ada rumus langsung "masukkan angka, keluar posisi titik". Satu-satunya cara adalah lewat proses numerik yang cukup rumit (algoritma Golub-Welsch, sejenis proses aljabar linear/eigenvalue) yang dikerjakan komputer. Ini bukan masalah besar karena prosesnya cepat & murah, tapi maksudnya: setiap kali mau menjalankan EAP, komputer harus "menghitung ulang" posisi & bobot titik grid ini lebih dulu (bukan tinggal comot dari rumus) - kecuali hasilnya disimpan/di-cache dari sebelumnya untuk dipakai lagi selama pengaturannya ($pts$, bentuk prior) tidak berubah.


### 5.4 Cara Menentukan Jumlah Titik Grid ($pts$) yang Sesuai

Karena akurasi EAP bergantung pada $pts$ sementara biaya komputasi tumbuh $pts^k$ (poin Kekurangan
di atas), memilih $pts$ adalah trade-off eksplisit presisi vs waktu komputasi. Tidak ada rumus
tunggal untuk $pts$ optimal, tapi
ada beberapa heuristik praktis:

**1. Uji konvergensi empiris (paling andal).** Naikkan $pts$ bertahap (mis. $5\to11\to21\to41$) pada
data representatif, berhenti begitu $\hat\theta_{EAP}$ tidak lagi berubah berarti (mis. selisih $<
se/10$). Ini persis yang dilakukan [#5.2.2](#522-demo-2-multidimensional-k3-7-item-grid-53125-titik):
$pts=5\to21$ mengubah $\hat\theta_{reasoning}$ dari $0.134$ ke $0.182$ - selisih masih signifikan,
artinya $pts=5$ terlalu kasar untuk dataset itu dan $pts=21$ lebih aman dipakai sebagai default.

**2. Pertimbangkan derajat eksak kuadratur.** Gauss-Hermite $pts$ titik eksak untuk polinomial
hingga derajat $2\cdot pts-1$ ([#5.1](#51-teori)). Likelihood IRT bukan polinomial, tapi berbentuk
lonceng halus di sekitar mode - secara empiris $pts$ di kisaran 15-21 biasanya sudah cukup untuk
kasus unidimensional/dimensi-rendah dengan diskriminasi item $a_i$ yang tidak ekstrem (likelihood
tidak terlalu tajam/sempit).

**3. Pertimbangkan biaya $pts^k$ terhadap jumlah dimensi $k$** - biaya tumbuh eksponensial dengan
$k$, bukan cuma $pts$:

| $k$ | $pts=11$ | $pts=21$ | $pts=41$ |
|---|---|---|---|
| 1 | 11 | 21 | 41 |
| 2 | 121 | 441 | 1,681 |
| 3 | 1,331 | 9,261 | 68,921 |

Untuk $k\geq4$, $pts=21$ berarti $\geq194{,}481$ titik per estimasi - kuadratur grid rectangular
jadi impraktis (*curse of dimensionality*) dan makin mahal walau kodenya sendiri sudah generik
untuk $k$ berapa pun; pada titik ini MAP/MLE (biayanya tidak bergantung $pts$ sama sekali) jadi
pilihan lebih realistis secara komputasi.


---

## 6. Ringkasan Perbandingan

| Metode | Formula Inti | Butuh Prior? | Titik Awal | Bisa Divergen? |
|---|---|---|---|---|
| MLE | $\arg\max_\theta f(\mathbf u\mid\theta)$ | Tidak | $\mathbf 0$ | Ya ([#5.2.3](#323-demo-3-kasus-divergen-all-correct)) |
| MAP | $\arg\max_\theta f(\theta)L(\theta)$ | Ya | $\boldsymbol\mu$ (prior mean) | Tidak (prior proper) |
| EAP | $\dfrac{\int\theta f(\theta)L(\theta)d\theta}{\int f(\theta)L(\theta)d\theta}$ | Ya ($\boldsymbol\mu,\boldsymbol\Sigma$ sama seperti MAP - `prior_mean`/`prior_cov_diag`, lihat [#5.1](#51-teori)) | N/A (bukan iteratif) | Tidak |

**Hasil numerik pada dataset identik** (7 item, pola respons campuran - lihat
[#5.2.2](#322-demo-2-multidimensional-k3-7-item)):

| Metode | $\hat\theta_{verbal}$ | $\hat\theta_{numeric}$ | $\hat\theta_{reasoning}$ |
|---|---|---|---|
| MLE | 0.0380 | -0.6537 | 0.2966 |
| MAP ($\Sigma=I$) | 0.0105 | -0.3656 | 0.1340 |
| EAP ($pts=21$) | 0.0323 | -0.3641 | 0.1824 |

**SE pada dataset identik yang sama** - teori masing-masing di
[#3.1.1](#311-teori-standard-error-se-untuk-mle)/[#4.1.2](#412-teori-standard-error-se-untuk-map)/[#5.1.2](#512-teori--pembuktian-standard-error-se-untuk-eap):

| Metode | $SE_{verbal}$ | $SE_{numeric}$ | $SE_{reasoning}$ |
|---|---|---|---|
| MLE | 0.8309 | 0.8458 | 0.7536 |
| MAP ($\Sigma=I$) | 0.6294 | 0.6231 | 0.5704 |
| EAP ($pts=21$) | 0.6751 | 0.6679 | 0.6099 |

Pola $SE_{MAP}<SE_{EAP}<SE_{MLE}$ konsisten di semua dimensi: MAP paling kecil karena mode posterior +
kelengkungan Hessian penuh (termasuk prior) selalu memberi variance Laplace terkecil; MLE paling besar
karena tanpa informasi prior sama sekali; EAP di antaranya karena mengukur *spread* aktual posterior
(bisa lebih lebar dari sekadar kelengkungan di mode jika posterior tidak simetris sempurna).

**Hasil numerik pada dataset all-correct (kasus divergen MLE)** - lihat [#5.2.3](#323-demo-3-kasus-divergen-all-correct):

| Metode | $\hat\theta$ | $\|\hat\theta\|$ |
|---|---|---|
| MLE | $[16.065,14.291,11.594]$ | 24.43 |
| MAP | $[0.472,0.369,0.450]$ | 0.75 |
| EAP | $[0.584,0.480,0.564]$ | 0.94 |

| Aspek | MLE | MAP | EAP |
|---|---|---|---|
| Basis teori | Likelihood murni (frequentist) | Posterior mode (Bayesian) | Posterior mean (Bayesian) |
| Algoritma | Newton-Raphson / Fisher scoring | Newton-Raphson / Fisher scoring + prior | Kuadratur Gauss-Hermite (bukan iteratif) |
| Cocok untuk tahap tes | Menengah–akhir (butuh $\geq$ beberapa item non-separable) | Awal–akhir (aman sejak round 1) | Awal–akhir (aman sejak round 1, tapi mahal) |
| Risiko utama | Divergensi pada pola respons separable | Bias ke prior jika $\mu$ keliru | Akurasi bergantung $pts$ |
| Biaya komputasi | Rendah (~3-5 iterasi $k\times k$ inverse) | Rendah (sama seperti MLE) | Tinggi ($pts^k$ evaluasi likelihood + 1 dekomposisi eigen $pts\times pts$) |

---

## 7. Catatan: Cara Menentukan Prior (MAP & EAP)

Berlaku sama untuk kedua metode karena MAP dan EAP memakai prior normal multivariat
$\mathcal N(\boldsymbol\mu,\boldsymbol\Sigma)$ yang identik ($\boldsymbol\mu$=`prior_mean`,
$\boldsymbol\Sigma$=`prior_cov_diag` - lihat [#4.1](#41-teori)/[#5.1](#51-teori)); EAP hanya beda
cara memakainya (integral penuh, bukan penalti pada mode).

### 7.1 Cara Menentukan Mean ($\mu$) dan Variance ($\Sigma$) yang Tepat

Properness ([#7.2](#72-cara-mengecek-properness-simetris--positive-definite) di bawah) hanya
menjamin prior **valid secara matematis** - tidak menjamin prior itu **masuk akal secara
psikometrik**. Bagian ini membahas cara memilih nilai $\boldsymbol\mu,\boldsymbol\Sigma$ yang
tepat, bukan sekadar sah. Magis & Raîche [3, p.4] menegaskan pemilihan ini murni soal keyakinan
tentang populasi: *"The choice of a prior distribution is usually driven by some prior belief of
the ability distribution among the population of examinees."* - lihat juga [#4.1](#41-teori) untuk
kutipan lengkap dan [#4.1.1](#411-apa-bedanya-prior-dan-variance) untuk peran $\Sigma$ sebagai
kekuatan *shrinkage*.

**1. Default weakly-informative: $\boldsymbol\mu=\mathbf 0,\ \boldsymbol\Sigma=\mathbf I$.** 

Skala
$\theta$ pada IRT tidak punya satuan natural (arbitrary scale) - konvensi standar adalah
menjangkarkan skala itu ke populasi rujukan berdistribusi normal baku, sehingga $\theta=0$ berarti
"kemampuan rata-rata populasi" dan $\theta=\pm1$ berarti "satu deviasi standar dari rata-rata". Cocok dipakai selama
tidak ada informasi tambahan tentang populasi examinee.

**2. $\mu$ mewakili populasi, bukan tebakan tentang examinee itu sendiri.** $\mu$ 

adalah keyakinan
*sebelum* melihat respons examinee yang sedang dites - kalau nilainya digeser dari $0$, pergeseran
itu harus berasal dari informasi **agregat populasi/subgroup** (mis. norma kelas, riwayat skor
tes lain yang berkorelasi, level pendidikan), bukan dari dugaan tentang kemampuan examinee
individu itu sendiri. Memilih $\mu$ berdasarkan examinee yang sama yang sedang diestimasi adalah
*circular* dan membuat estimasi bias secara sistematis ke arah tebakan itu.

**3. Empirical Bayes: estimasi $\mu,\Sigma$ dari histori kalibrasi.** 

Kalau tersedia data
$\hat\theta_{MLE}$ dari batch examinee sebelumnya pada populasi yang sama (mis. hasil administrasi
tes periode lalu), $\boldsymbol\mu$ dan $\boldsymbol\Sigma$ untuk periode berikutnya dapat
diestimasi langsung sebagai mean dan kovarians sampel dari histori itu:

$$
\hat{\boldsymbol\mu} = \frac{1}{N}\sum_{j=1}^N \hat{\boldsymbol\theta}_j, \qquad
\hat{\boldsymbol\Sigma} = \frac{1}{N-1}\sum_{j=1}^N (\hat{\boldsymbol\theta}_j-\hat{\boldsymbol\mu})(\hat{\boldsymbol\theta}_j-\hat{\boldsymbol\mu})^\top
$$

dengan $N$ jumlah examinee historis dan $\hat{\boldsymbol\theta}_j$ estimasi kemampuan examinee
ke-$j$. Ini pendekatan *empirical Bayes* standar: populasi yang sama cenderung punya sebaran
kemampuan yang mirip antar periode, sehingga prior makin representatif dibanding default
$N(\mathbf 0,\mathbf I)$ yang generik.

**Histori harus dari MLE, bukan dari MAP/EAP periode sebelumnya.** $\hat{\boldsymbol\theta}_j$ di
atas idealnya adalah estimasi **MLE**, bukan hasil MAP/EAP periode lalu. Alasannya: MAP/EAP sudah
menarik ($\text{shrink}$) tiap $\hat\theta_j$ ke arah $\mu$ prior yang dipakai saat itu
([#4.1.1](#411-apa-bedanya-prior-dan-variance)), sehingga sebaran histori MAP/EAP sudah dipersempit
secara artifisial dibanding sebaran kemampuan populasi yang sesungguhnya. Kalau
$\hat{\boldsymbol\Sigma}$ dihitung dari histori yang sudah di-shrink itu, hasilnya akan
**under-estimate** variance populasi asli - dan kalau prior periode lalu memang kurang tepat, bias
itu ikut terbawa ke prior baru (efeknya makin besar kalau prior lama itu kuat/$\Sigma$ kecil).
Estimasi MLE tidak punya masalah ini karena murni dari data tanpa pengaruh prior sama sekali
(asalkan tidak divergen - lihat [#3.1](#31-teori)), sehingga mean/variance sampelnya representasi
paling jujur dari sebaran kemampuan populasi. Kalau yang tersimpan cuma histori MAP/EAP (tidak ada
MLE), tetap bisa dipakai sebagai pendekatan kasar, tapi dengan kesadaran $\hat{\boldsymbol\Sigma}$
hasilnya kemungkinan lebih kecil dari variance populasi yang sebenarnya.

**Contoh perhitungan (multidimensional, $k=3$).** Misalkan tersedia $\hat{\boldsymbol\theta}_{MLE}$
dari $N=5$ examinee periode sebelumnya, pada dimensi verbal/numeric/reasoning yang sama seperti
[Item Bank Snapshot](/posts/cat/item-selection-criteria-mcat#item-bank-snapshot). Karena kode
produksi memakai $\Sigma$ diagonal saja (`prior_cov_diag`, lihat
[#4.1.1](#411-apa-bedanya-prior-dan-variance)), mean dan variance dihitung **per dimensi secara
independen** - tidak ada kovarians antar dimensi yang dihitung/dipakai:

| Examinee ke-$j$ | $\hat\theta_{verbal}$ | $\hat\theta_{numeric}$ | $\hat\theta_{reasoning}$ |
|---|---|---|---|
| 1 | 0.8 | -0.3 | 1.1 |
| 2 | -0.5 | 0.6 | 0.2 |
| 3 | 1.2 | -0.8 | 0.4 |
| 4 | -0.3 | 0.9 | -0.1 |
| 5 | 0.3 | 0.1 | 0.9 |

**Step 1: Hitung mean tiap dimensi ($\hat\mu_d$)** - rata-rata kolom, terpisah untuk tiap dimensi:

$$
\hat\mu_{verbal} = \frac{0.8-0.5+1.2-0.3+0.3}{5} = \frac{1.5}{5} = 0.3, \qquad
\hat\mu_{numeric} = \frac{-0.3+0.6-0.8+0.9+0.1}{5} = \frac{0.5}{5} = 0.1, \qquad
\hat\mu_{reasoning} = \frac{1.1+0.2+0.4-0.1+0.9}{5} = \frac{2.5}{5} = 0.5
$$

**Step 2: Hitung deviasi tiap examinee dari mean kolomnya, lalu kuadratkan** (tiap dimensi
memakai $\hat\mu_d$ miliknya sendiri dari Step 1):

| $j$ | $\hat\theta_{verbal}-\hat\mu_{verbal}$ | kuadrat | $\hat\theta_{numeric}-\hat\mu_{numeric}$ | kuadrat | $\hat\theta_{reasoning}-\hat\mu_{reasoning}$ | kuadrat |
|---|---|---|---|---|---|---|
| 1 | $0.8-0.3=0.5$ | 0.25 | $-0.3-0.1=-0.4$ | 0.16 | $1.1-0.5=0.6$ | 0.36 |
| 2 | $-0.5-0.3=-0.8$ | 0.64 | $0.6-0.1=0.5$ | 0.25 | $0.2-0.5=-0.3$ | 0.09 |
| 3 | $1.2-0.3=0.9$ | 0.81 | $-0.8-0.1=-0.9$ | 0.81 | $0.4-0.5=-0.1$ | 0.01 |
| 4 | $-0.3-0.3=-0.6$ | 0.36 | $0.9-0.1=0.8$ | 0.64 | $-0.1-0.5=-0.6$ | 0.36 |
| 5 | $0.3-0.3=0.0$ | 0.00 | $0.1-0.1=0.0$ | 0.00 | $0.9-0.5=0.4$ | 0.16 |
| **sum** | | **2.06** | | **1.86** | | **0.98** |

**Step 3: Hitung variance tiap dimensi ($\hat\sigma_d^2$)** - jumlah kuadrat deviasi kolom dibagi
$N-1$ (bukan $N$, supaya estimatornya *unbiased*):

$$
\hat\sigma^2_{verbal} = \frac{2.06}{5-1} = 0.515, \qquad
\hat\sigma^2_{numeric} = \frac{1.86}{5-1} = 0.465, \qquad
\hat\sigma^2_{reasoning} = \frac{0.98}{5-1} = 0.245
$$

→ Prior untuk periode berikutnya: `prior_mean = [0.3, 0.1, 0.5]`,
`prior_cov_diag = [0.515, 0.465, 0.245]` (dibanding default `[0,0,0]`/`[1,1,1]`). Interpretasi per
dimensi: populasi historis rata-rata sedikit di atas 0 pada ketiga dimensi (paling menonjol di
reasoning, $\mu=0.5$), dan sebarannya di ketiga dimensi lebih sempit dari default ($\Sigma_d<1$
semua) - dimensi reasoning paling sempit ($\Sigma=0.245$) sehingga prior di dimensi itu menarik
paling kuat, dimensi verbal paling longgar ($\Sigma=0.515$) sehingga tarikannya paling lemah di
antara ketiganya.

**4. Efek $\mu$ yang keliru terhadap estimasi.**

Intinya: **$\mu$ yang salah paling berbahaya di awal tes** (item masih sedikit), dan efeknya makin
hilang sendiri seiring examinee menjawab lebih banyak item.

Kenapa begitu? Estimasi MAP ditarik oleh dua kekuatan sekaligus - "bukti dari jawaban examinee"
($\mathbf{I}_S(\theta)$, makin besar kalau makin banyak item dijawab) dan "tarikan ke $\mu$"
($\boldsymbol\Sigma^{-1}$, besarnya tetap, tidak berubah walau item bertambah), karena
$\mathbf{H}_{MAP}=\mathbf{I}_S(\theta)+\boldsymbol\Sigma^{-1}$ ([#4.1](#41-teori)). Di awal tes,
$\mathbf{I}_S(\theta)$ masih kecil (baru sedikit/belum ada bukti dari jawaban), jadi tarikan ke
$\mu$ mendominasi - kalau $\mu$ ternyata jauh dari kemampuan asli examinee, estimasi awal akan
condong salah ke arah $\mu$ itu (lihat tabel shrinkage di
[#4.1.1](#411-apa-bedanya-prior-dan-variance)). Begitu makin banyak item dijawab,
$\mathbf{I}_S(\theta)$ membesar dan lama-lama jauh mengalahkan $\boldsymbol\Sigma^{-1}$, sehingga
pengaruh $\mu$ yang keliru itu makin pudar dengan sendirinya.

**5. Kalau $\mu$ tidak yakin, jangan pakai $\Sigma$ kecil.** 

$\Sigma$ kecil berarti prior "yakin"
dan menarik kuat ke $\mu$ ([#4.1.1](#411-apa-bedanya-prior-dan-variance)) - kombinasi $\mu$ yang
mungkin keliru dengan $\Sigma$ kecil adalah kondisi bias terburuk (tarikan kuat ke titik yang
salah). Kalau tidak ada dasar empiris kuat untuk $\mu\neq0$, lebih aman memakai $\Sigma=\mathbf I$
(atau lebih besar) supaya likelihood cepat mendominasi begitu beberapa item pertama dijawab,
alih-alih $\Sigma$ kecil yang mengunci estimasi ke $\mu$ yang belum tentu benar.

**Ringkasan praktis:**

| Situasi | $\mu$ | $\Sigma$ |
|---|---|---|
| Tidak ada informasi populasi | $\mathbf 0$ | $\mathbf I$ (weakly-informative) |
| Ada histori kalibrasi dari populasi sama | mean sampel histori | kovarians sampel histori |
| Ada info subgroup (mis. norma kelas) tapi bukan dari examinee ini | mean subgroup | $\geq$ variance populasi subgroup (jangan lebih kecil dari sebaran asli) |
| $\mu$ tidak yakin/berisiko keliru | tetap $\mathbf 0$ atau estimasi terbaik | besar (longgar), bukan kecil |

### 7.2 Cara Mengecek Properness (Simetris & Positive-Definite)

Prior $f(\boldsymbol\theta)$ disebut **proper** kalau memenuhi dua syarat: 
1. $f(\boldsymbol\theta)
\geq 0$ untuk semua $\boldsymbol\theta$, dan 
2. $\int f(\boldsymbol\theta)\,d\boldsymbol\theta$
terintegrasi ke nilai finite (dapat dinormalisasi menjadi 1). 

Sebaliknya, **improper prior** - mis.
flat prior di seluruh $(-\infty,\infty)$ atau prior Jeffreys $1/\theta$ di $(0,\infty)$ - integralnya
divergen sehingga bukan distribusi probabilitas yang valid.

Untuk prior normal multivariat $\mathcal N(\boldsymbol\mu, \boldsymbol\Sigma)$ yang dipakai di MAP
dan EAP pada dokumen ini, properness cukup dicek lewat $\boldsymbol\Sigma$ saja:

- $\boldsymbol\Sigma$ harus **simetris** ($\boldsymbol\Sigma^\top = \boldsymbol\Sigma$),
- $\boldsymbol\Sigma$ harus **positive-definite** (semua eigenvalue $> 0$, sehingga
  $\boldsymbol\Sigma^{-1}$ ada dan finite).

Kalau kedua syarat itu terpenuhi, konstanta normalisasi distribusi normal multivariat otomatis
membuat integral totalnya $=1$ - properness terjamin tanpa perlu menghitung integral manual. Secara
praktis, pengecekan positive-definite bisa dilakukan lewat **dekomposisi Cholesky**: jika dekomposisi
berhasil, $\boldsymbol\Sigma$ positive-definite (proper); jika gagal, $\boldsymbol\Sigma$ singular atau
punya eigenvalue $\leq 0$ (bukan kovarians valid, prior tidak proper).

Kasus khusus $\mathcal N(\mathbf 0, \mathbf I)$ - yang dipakai sebagai default di beberapa demo -
**selalu proper** di dimensi berapa pun, karena matriks identitas $\mathbf I$ otomatis simetris dan
semua eigenvalue-nya $=1>0$, tanpa bergantung pada input apa pun.

**Contoh $\boldsymbol\Sigma$ proper selain $\mathbf I$ (K=3):**

$$\boldsymbol\Sigma = \begin{bmatrix} 1.5 & 0.3 & 0 \\ 0.3 & 1.2 & 0.2 \\ 0 & 0.2 & 1.0 \end{bmatrix}$$

Varians tiap dimensi tidak seragam (1.5, 1.2, 1.0) dan ada korelasi prior antar $\theta_1$-$\theta_2$
(0.3) serta $\theta_2$-$\theta_3$ (0.2), sementara $\theta_1$-$\theta_3$ diasumsikan independen (0).

- Simetris: $\Sigma_{12}=\Sigma_{21}=0.3$, $\Sigma_{23}=\Sigma_{32}=0.2$, $\Sigma_{13}=\Sigma_{31}=0$ ✓
- Positive-definite lewat **Sylvester's criterion** (semua leading principal minor $>0$, alternatif
  praktis dari menghitung eigenvalue satu-satu):
  - $M_1 = 1.5 > 0$
  - $M_2 = 1.5(1.2) - 0.3^2 = 1.71 > 0$
  - $M_3 = \det(\boldsymbol\Sigma) = 1.5(1.2\cdot1.0-0.2^2) - 0.3(0.3\cdot1.0-0.2\cdot0) = 1.65 > 0$

Semua minor positif → $\boldsymbol\Sigma$ positive-definite → $\boldsymbol\Sigma^{-1}$ ada dan finite →
prior proper.

**Contoh gagal (kontras):** $\boldsymbol\Sigma_{\text{invalid}} = \begin{bmatrix} 1 & 2 \\ 2 & 1 \end{bmatrix}$
simetris, tapi $M_2 = 1(1)-2^2=-3<0$ → bukan positive-definite (korelasi 2 melebihi batas valid untuk
varians 1 & 1, seharusnya korelasi ternormalisasi $\in[-1,1]$). $\Sigma^{-1}$ tetap bisa dihitung secara
aljabar, tapi bukan kovarians valid, sehingga bukan prior proper meski syarat simetri terpenuhi.

Penting: properness murni sifat $f(\boldsymbol\theta)$ (yakni $\boldsymbol\mu$ dan $\boldsymbol\Sigma$)
- **tidak bergantung pada bank soal**. Parameter item ($a$, $d$, $c$) dan respons examinee hanya
masuk ke likelihood $L(\boldsymbol\theta)$, bukan ke prior; bank soal memengaruhi akurasi/bias hasil
estimasi, bukan properness prior itu sendiri.

---

## Referensi

**[1]** Mulder, J., & van der Linden, W. J. (2009). Multidimensional Adaptive Testing with Optimal
Design Criteria for Item Selection. *Psychometrika*, 74(2), 273–296.
https://doi.org/10.1007/s11336-008-9097-5 - Full text gratis (PubMed Central, open access):
https://pmc.ncbi.nlm.nih.gov/articles/PMC2813188/ (mirror PDF jurnal dengan nomor halaman asli:
https://www.cambridge.org/core/services/aop-cambridge-core/content/view/A3BFF7744EDCE563819C31270D9C7E7D/S0033312300021608a.pdf/multidimensional-adaptive-testing-with-optimal-design-criteria-for-item-selection.pdf).
Sumber untuk: model M3PL (Eq.1, p.275), definisi MLE & fungsi likelihood (Eq.2-3, p.276),
pernyataan Newton-Raphson & catatan non-eksistensi maksimum (p.276), Fisher Information Matrix
(Eq.4, p.276), aditivitas FIM (Eq.6, p.277), dan normalitas asimtotik/Cramér–Rao (Eq.7, p.277).
Sama seperti [1] pada
[item_selection_summary.md](/posts/cat/item-selection-criteria-mcat#referensi), bagian
berbeda (#2-3 alih-alih #3-4).

**[2]** Baker, F. B. (2001). *The Basics of Item Response Theory* (2nd ed.). ERIC Clearinghouse on
Assessment and Evaluation, University of Maryland. Full text gratis (ERIC ED458219):
https://files.eric.ed.gov/fulltext/ED458219.pdf (mirror: https://www.ime.unicamp.br/~cnaber/Baker_Book.pdf).
Sumber untuk Bab 5 "Estimating an Examinee's Ability" (p.85-90): formula iteratif MLE univariat
Eq.[5-1] (p.86), contoh tiga-item lengkap dengan nilai *a priori* (p.87), dan tabel iterasi
1-2 yang direproduksi persis di [#3.2.1](#321-demo-1-reproduksi-baker-2001-k1) (p.88). Sama seperti [2] pada
[item_selection_summary.md](/posts/cat/item-selection-criteria-mcat#referensi) (di sana dipakai untuk Bab 6
"The Information Function", di sini untuk Bab 5).

**[3]** Magis, D., & Raîche, G. (2012). Random Generation of Response Patterns under Computerized
Adaptive Testing with the R Package catR. *Journal of Statistical Software*, 48(8), 1–31.
https://doi.org/10.18637/jss.v048.i08 - Open access (JSS). PDF:
https://www.jstatsoft.org/index.php/jss/article/view/v048i08/600 (landing page:
https://www.jstatsoft.org/v48/i08/). Sumber utama #2.2 "Ability estimation" (p.4-6): definisi ML
(Eq.2-4, p.4), Bayes Modal/MAP (Eq.5-6, p.4-5), Jeffreys' prior (Eq.7-9, p.5, tidak dipakai kode
produksi), EAP (Eq.10-11, p.5-6), dan Weighted Likelihood/Warm estimator (Eq.12-14, p.6, tidak
diimplementasikan produksi - dicatat sebagai pembanding di [#4](#6-ringkasan-perbandingan)).

**[4]** Mislevy, R. J. (1986). Bayes modal estimation in item response models. *Psychometrika*,
51(2), 177–195. https://doi.org/10.1007/BF02293979 - Sumber asli/historis estimasi Bayes
Modal (MAP). Tidak berhasil diakses gratis (Springer/Psychometrika berbayar) - klaim yang berasal
dari Mislevy (1986) pada dokumen ini hanya diverifikasi secara tidak langsung lewat definisi &
nomor persamaan yang direproduksi eksplisit di [3, Eq.5-6, p.4-5].

**[5]** Bock, R. D., & Mislevy, R. J. (1982). Adaptive EAP estimation of ability in a
microcomputer environment. *Applied Psychological Measurement*, 6(4), 431–444.
https://doi.org/10.1177/014662168200600405 - Sumber asli/historis estimasi EAP. Tidak berhasil
diakses gratis (SAGE berbayar) - klaim yang berasal dari Bock & Mislevy (1982) pada dokumen ini
hanya diverifikasi secara tidak langsung lewat definisi & nomor persamaan yang direproduksi
eksplisit di [3, Eq.10-11, p.5-6], yang juga menyitasi Bock & Mislevy (1982) secara langsung
sebagai sumber EAP (p.5).

**[6]** Chalmers, R. P. (2012). mirt: A Multidimensional Item Response Theory Package for the R
Environment. *Journal of Statistical Software*, 48(6), 1–29. https://doi.org/10.18637/jss.v048.i06
- Open access (JSS). PDF: https://www.jstatsoft.org/index.php/jss/article/view/v048i06/598
(landing page: https://www.jstatsoft.org/article/view/v048i06). Sumber untuk model M3PL
multidimensional dengan skala $D$ (Eq.1, p.3 - produksi tidak memakai skala $D$, konsisten dengan
[1, Eq.1]) dan pola diskretisasi grid kuadratur multi-indeks $k$-dimensi (Eq.6, p.5), dipakai
sebagai pembanding teknik untuk grid EAP di [#3.1](#31-teori) (catatan: Eq.6 [6] pada paper aslinya
mengintegralkan $\theta$ sebagai *nuisance parameter* pada estimasi parameter item/EM, bukan pada
estimasi EAP examinee individual - hanya teknik diskretisasinya yang dipakai sebagai pembanding,
bukan rumusnya secara langsung).

**[7]** Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss Quadrature Rules. *Mathematics of
Computation*, 23(106), 221–230. https://doi.org/10.1090/S0025-5718-69-99647-1 - Sumber asli algoritma
Golub-Welsch (nilai eigen matriks Jacobi tridiagonal → node/bobot kuadratur Gauss) dipakai persis oleh
`gauss_hermite()` di `eap.rs:12-30`. PDF di halaman jurnal (`ams.org`) mengembalikan HTTP 403 saat
diambil langsung - tidak berhasil diverifikasi *full text* gratis. Klaim yang berasal dari paper ini
pada dokumen ini (konstruksi matriks Jacobi, rumus bobot $w_q=\mu_0v_{q,0}^2$) diverifikasi secara
tidak langsung: (a) turunan aljabar dari rekursi tiga-suku polinomial Hermite di [#5.1](#51-teori)
menghasilkan matriks off-diagonal yang identik dengan kode produksi, dan (b) untuk $pts=5$, akar $H_5$
yang diturunkan analitik (faktorisasi $H_5(x)=8x(4x^4-20x^2+15)$, tanpa dekomposisi eigen sama sekali)
cocok tepat dengan node numerik yang dihasilkan `gauss_hermite()` dan dipakai di
[#5.2.1](#521-demo-1-reproduksi-1-dimensi-pts5).

### Peta Sitasi per Formula

| Formula | Dipakai di metode | Sumber |
|---|---|---|
| $f(\mathbf u\mid\theta)=\prod P_i^{u_i}Q_i^{1-u_i}$, $\hat\theta=\arg\max f$ | MLE (dasar ketiganya) | [1] Eq.2–3, p.276 |
| $\nabla\log f(\theta)=\sum a_i(u_i-P_i)P'_i/(P_iQ_i)$ | MLE, MAP (bagian likelihood) | Diturunkan sendiri di [#4.1](#21-pembuktian-skor-gradien-log-likelihood) dari [1] Eq.2-3 |
| $\mathbf I_i(\theta)=w\cdot\mathbf a_i\mathbf a_i^\top$ (Fisher scoring) | MLE, MAP (Hessian) | [1] Eq.4, p.276; dibuktikan identik di [#4.2](#22-fisher-information-matrix-sebagai-pengganti-hessian-fisher-scoring) |
| $\hat\theta_{s+1}=\hat\theta_s+[\Sigma a(u-P)]/[\Sigma a^2PQ]$ | MLE (univariat) | [2] Eq.[5-1], p.86 |
| $\log g(\theta)=\log f(\theta)+\log L(\theta)$, $\hat\theta_{BM}=\arg\max g$ | MAP | [3] Eq.5, p.5; asal-usul [4] |
| $\nabla\log g=\nabla\log f-\Sigma^{-1}(\theta-\mu)$, $H_{MAP}=I_S+\Sigma^{-1}$ | MAP | Diturunkan sendiri di [#4.1](#41-teori) dari [3] Eq.5 + kalkulus Gaussian multivariat |
| $\hat\theta_{EAP}=\int\theta f L\,d\theta/\int fL\,d\theta$ | EAP | [3] Eq.10, p.5; asal-usul [5] |
| Grid kuadratur multi-indeks $k$-dimensi | EAP | Teknik dibandingkan dengan [6] Eq.6, p.5 |
| Node $x_q$/bobot $w_q$ via eigen matriks Jacobi (Golub-Welsch) | EAP (`gauss_hermite()`) | [7]; diverifikasi analitik untuk $pts=5$ di [#5.1](#51-teori) |
| $SE(\hat\theta_j)=\sqrt{[\mathbf I_S(\hat\theta)^{-1}]_{jj}}$ | MLE | Diturunkan sendiri di [#3.1.1](#311-teori-standard-error-se-untuk-mle) dari [1] Eq.7, p.277 |
| $SE(\hat\theta_{BM,j})=\sqrt{[H_{MAP}(\hat\theta)^{-1}]_{jj}}$ | MAP | [3] Eq.6, p.5 (bentuk univariat); generalisasi multivariat diturunkan sendiri di [#4.1.2](#412-teori-standard-error-se-untuk-map) |
| $se(\hat\theta_{EAP})=\sqrt{E[\theta^2\mid\mathbf u]-E[\theta\mid\mathbf u]^2}$ | EAP | [3] Eq.11, p.6; identitas momen-mentah $\equiv$ bentuk tersentral dibuktikan sendiri di [#5.1.2](#512-teori--pembuktian-standard-error-se-untuk-eap) |
