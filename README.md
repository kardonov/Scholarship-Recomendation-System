# 🎓 Sistem Rekomendasi Beasiswa — LightGBM & XGBoost

> Sistem cerdas berbasis Machine Learning untuk merekomendasikan penerima beasiswa secara otomatis dan berbasis data, menggunakan pendekatan metodologi **CRISP-DM** dengan algoritma **LightGBM** dan **XGBoost**.

---

## 📌 Deskripsi Proyek

Proyek ini merupakan sistem rekomendasi beasiswa yang menganalisis data mahasiswa — mencakup skor akademik dan latar belakang sosial-ekonomi — untuk memprediksi kelayakan penerimaan beasiswa. Sistem membandingkan dua algoritma boosting terkemuka (LightGBM dan XGBoost) dengan berbagai varian hyperparameter untuk menemukan pendekatan paling akurat dan efisien dalam pengambilan keputusan beasiswa yang adil dan berbasis data.

Proyek ini dibangun dalam dua tahap:

- **`notebook.ipynb`** — Eksperimen CRISP-DM lengkap: EDA, preprocessing, training, tuning, evaluasi.
- **`app.py`** — Aplikasi web interaktif menggunakan Streamlit sebagai kelanjutan dari notebook.

---

## 🗂️ Struktur Proyek

```
scholarship-recommendation/
│
├── CRISP_DM_with_LGBM_and_XGB_7_Feature.ipynb   # Notebook eksperimen utama
├── app.py                                         # Aplikasi Streamlit
├── README.md                                      # Dokumentasi proyek (file ini)
│
├── dataset/
│   └── Mahasiswa_ditetapkan_merged_rev5.csv       # Dataset mahasiswa (diperlukan)
│
└── models/                                        # (opsional) Model yang disimpan
    ├── lgbm_scholarship.pkl
    └── xgb_scholarship.pkl
```

---

## 🔬 Metodologi: CRISP-DM

Proyek ini mengikuti kerangka kerja **Cross-Industry Standard Process for Data Mining (CRISP-DM)** secara menyeluruh.

```
1. Business Understanding
       ↓
2. Data Understanding
       ↓
3. Data Preparation
       ↓
4. Modeling
       ↓
5. Evaluation
       ↓
6. Deployment
```

### 1. Business Understanding

Perguruan tinggi membutuhkan sistem seleksi beasiswa yang **objektif, konsisten, dan terukur**. Sistem ini dirancang untuk:

- Mengklasifikasikan mahasiswa sebagai **Layak** atau **Tidak Layak** menerima beasiswa.
- Mempertimbangkan faktor akademik dan sosial-ekonomi secara bersamaan.
- Membandingkan performa LightGBM vs XGBoost untuk menemukan model terbaik.

### 2. Data Understanding

Dataset berisi data mahasiswa pendaftar beasiswa dengan berbagai atribut sosial-ekonomi dan akademik. Eksplorasi data meliputi:

- Pemeriksaan distribusi kelas target (`Output_Class`)
- Analisis missing values dan duplikat
- Visualisasi distribusi fitur kategorikal (penghasilan, kepemilikan rumah, dll.)
- Analisis distribusi fitur numerik (skor kuesioner, nilai tes)

### 3. Data Preparation

| Tahap | Detail |
|---|---|
| **Drop kolom** | Menghapus 18 kolom tidak relevan (nama, NIK, NISN, dll.) |
| **Label Encoding** | Mengkodekan 7 kolom kategorikal menjadi numerik |
| **Standard Scaling** | Normalisasi fitur numerik kontinu |
| **Train-Test Split** | 70% training / 30% testing, `random_state=42` |

**Kolom yang di-drop:**

```
Nama_Mahasiswa(2), No_Pendaftaran(1), NIK(3), NISN(6),
Jumlah_Tanggungan(28), Tanggal_Lahir(15), Jenis_Kelamin(16),
Pendidikan Ayah, Pendidikan Ibu, Pekerjaan_Ibu(25),
Pekerjaan_Ayah(21), Status_Ayah(23), Status_Ibu(27),
Program_Studi_Pilihan 1(41), Asal_Sekolah(11),
No_Handphone(18), Alamat_Email(19), Program_Studi_Pilihan 2(42)
```

### 4. Modeling

#### 🟢 LightGBM Models

| Model | Boosting Type | Keterangan |
|---|---|---|
| LGBM Standard | `gbdt` (default) | Tanpa hyperparameter tuning |
| LGBM DART | `dart` | Dengan GridSearchCV tuning |
| LGBM GOSS | `goss` | Dengan GridSearchCV tuning |
| LGBM GBDT | `gbdt` | Dengan GridSearchCV tuning |

**Best Parameters — LGBM GOSS (Model Terpilih):**

```python
lgb.LGBMClassifier(
    boosting_type='goss',
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    num_leaves=20,
    feature_fraction=0.7,
    bagging_fraction=0.7,
    reg_alpha=0,
    verbose=-1,
    random_state=40
)
```

#### 🟡 XGBoost Models

| Model | Booster | Keterangan |
|---|---|---|
| XGB Standard | `gbtree` | Tanpa hyperparameter tuning |
| XGB GBtree | `gbtree` | Dengan GridSearchCV tuning |
| XGB DART | `dart` | Dengan GridSearchCV tuning |

**Best Parameters — XGB GBtree (Model Terpilih):**

```python
xgb.XGBClassifier(
    booster='gbtree',
    objective='binary:logistic',
    n_estimators=100,
    learning_rate=0.01,
    max_depth=3,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=1,
    reg_alpha=0,
    reg_lambda=0,
    random_state=40,
    eval_metric='logloss'
)
```

### 5. Evaluation

Metrik evaluasi yang digunakan:

| Metrik | Formula |
|---|---|
| **Accuracy** | (TP + TN) / Total |
| **Precision** | TP / (TP + FP) |
| **Recall** | TP / (TP + FN) |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **AUC-ROC** | Area Under the ROC Curve |

Visualisasi evaluasi:
- Confusion Matrix (heatmap)
- ROC Curve per model
- ROC Curve perbandingan LGBM GOSS vs XGB GBtree
- Feature Importance (gain-based)

### 6. Deployment

Model di-deploy sebagai **aplikasi web interaktif** menggunakan Streamlit (`app.py`), dengan fitur:

- Upload dataset CSV
- Training otomatis semua model
- Prediksi real-time untuk data mahasiswa baru
- Download model terlatih (`.pkl` via joblib)

---

## 📊 Fitur Dataset (9 Fitur Utama)

| No | Fitur | Tipe | Keterangan |
|---|---|---|---|
| 1 | `Status_DTKS(7)` | Kategorikal | Status Data Terpadu Kesejahteraan Sosial |
| 2 | `Status_P3KE(8)` | Kategorikal | Status Penghapusan Kemiskinan Ekstrem |
| 3 | `Penghasilan_Ayah(22)` | Kategorikal | Kategori penghasilan ayah |
| 4 | `Penghasilan_Ibu(26)` | Kategorikal | Kategori penghasilan ibu |
| 5 | `Kepemilikan_Rumah(29)` | Kategorikal | Status kepemilikan rumah |
| 6 | `Sumber_Daya_Listrik(31)` | Kategorikal | Sumber daya listrik yang digunakan |
| 7 | `Prestasi(50)` | Kategorikal | Prestasi akademik/non-akademik |
| 8 | `Rata-rata_Skor_Kuesioner` | Numerik | Rata-rata skor kuesioner sosial-ekonomi |
| 9 | `Nilai_Tes(51)` | Numerik | Nilai tes seleksi masuk |

**Target Variable:**

| Nilai | Label | Keterangan |
|---|---|---|
| `0` | Layak | Mahasiswa direkomendasikan menerima beasiswa |
| `1` | Tidak Layak | Mahasiswa tidak direkomendasikan |

---

## 🖥️ Aplikasi Streamlit (`app.py`)

### Halaman yang Tersedia

| Halaman | Deskripsi |
|---|---|
| 🏠 **Dashboard** | Ringkasan dataset, distribusi kelas, statistik deskriptif |
| 📊 **Data Exploration** | Info dataset, missing values, distribusi fitur, correlation matrix |
| 🤖 **LightGBM Models** | Training & evaluasi 4 varian LGBM (Standard, GOSS, DART, GBDT) |
| ⚡ **XGBoost Models** | Training & evaluasi 3 varian XGB (Standard, GBtree, DART) |
| 📈 **Model Comparison** | Tabel perbandingan semua model + ROC curve gabungan |
| 🔮 **Prediksi** | Input manual data mahasiswa → prediksi real-time + download model |

### Fitur Utama Aplikasi

- **Upload CSV** langsung dari browser
- **Preprocessing otomatis** (drop, encode, scale, split)
- **Training 7 model** dengan caching (`@st.cache_resource`)
- **Confusion Matrix interaktif** (Plotly)
- **ROC AUC Curve** per model & perbandingan
- **Feature Importance** bar chart
- **Gauge chart** probabilitas prediksi
- **Download model** `.pkl` (LightGBM & XGBoost)
- **Dark theme** custom dengan Syne + DM Sans font

---

## ⚙️ Instalasi & Menjalankan

### Prasyarat

- Python **3.8+**
- pip

### 1. Clone / Download Proyek

```bash
git clone https://github.com/username/scholarship-recommendation.git
cd scholarship-recommendation
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Atau install manual:

```bash
pip install streamlit lightgbm xgboost scikit-learn pandas numpy \
            plotly joblib imbalanced-learn matplotlib seaborn
```

### 3. Jalankan Notebook (Opsional — untuk eksperimen)

```bash
jupyter notebook "CRISP_DM_with_LGBM_and_XGB_7_Feature.ipynb"
```

### 4. Jalankan Aplikasi Streamlit

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser: `http://localhost:8501`

---

## 📦 Requirements

```
streamlit>=1.28.0
lightgbm>=4.0.0
xgboost>=2.0.0
scikit-learn>=1.3.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
joblib>=1.3.0
imbalanced-learn>=0.11.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

> Simpan sebagai `requirements.txt` di root folder proyek.

---

## 🗃️ Format Dataset CSV

Dataset harus berupa file `.csv` dengan kolom-kolom berikut (minimal):

```
Status_DTKS(7), Status_P3KE(8), Penghasilan_Ayah(22),
Penghasilan_Ibu(26), Kepemilikan_Rumah(29), Sumber_Daya_Listrik(31),
Prestasi(50), Rata-rata_Skor_Kuesioner, Nilai_Tes(51), Output_Class
```

Kolom tambahan (identitas mahasiswa, dll.) akan otomatis di-drop saat preprocessing.

---

## 🧠 Arsitektur Sistem

```
CSV Dataset
     │
     ▼
┌─────────────────────────────────────┐
│         Data Preprocessing          │
│  Drop Columns → Label Encode →      │
│  Standard Scale → Train-Test Split  │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  LightGBM         XGBoost
  ─────────         ───────
  Standard          Standard
  GOSS   ──────►   GBtree  ──────►  Evaluation
  DART              DART             (Acc, Prec,
  GBDT                               Recall, F1,
                                     AUC-ROC)
               │
               ▼
        Best Model Selection
               │
               ▼
     Streamlit Deployment
     (Prediksi Real-time)
```

---

## 📈 Cara Menggunakan Aplikasi

### Prediksi Mahasiswa Baru

1. Buka aplikasi → `streamlit run app.py`
2. Upload file CSV dataset di sidebar
3. Navigasi ke halaman **🔮 Prediksi**
4. Isi form data mahasiswa:
   - Status DTKS, P3KE, penghasilan orang tua
   - Kepemilikan rumah, sumber listrik
   - Prestasi, skor kuesioner, nilai tes
5. Pilih model (LGBM GOSS / XGB GBtree / Keduanya)
6. Klik **"🔮 Prediksi Sekarang"**
7. Hasil: **✅ Layak** atau **❌ Tidak Layak** beserta probabilitas
8. (Opsional) Download model `.pkl` untuk digunakan di sistem lain

### Contoh Input Data

```python
{
    'Status_DTKS(7)':          1,    # Terdaftar DTKS
    'Status_P3KE(8)':          5,    # Desil 5
    'Penghasilan_Ayah(22)':   12,    # Kategori penghasilan rendah
    'Penghasilan_Ibu(26)':     3,    # Kategori penghasilan rendah
    'Kepemilikan_Rumah(29)':   1,    # Sewa/kontrak
    'Sumber_Daya_Listrik(31)': 1,    # PLN
    'Prestasi(50)':             1,    # Prestasi tingkat sekolah
    'Rata-rata_Skor_Kuesioner': 7.5, # Skor kuesioner
    'Nilai_Tes(51)':           75    # Nilai tes
}
```

---

## 🔍 Penjelasan GridSearchCV (Notebook)

Hyperparameter tuning dilakukan menggunakan `GridSearchCV` dengan konfigurasi:

```python
GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,           # 5-fold cross validation
    n_jobs=-1,      # Gunakan semua CPU core
    verbose=2
)
```

**Parameter Grid yang Dieksplor:**

| Parameter | Nilai yang Dicoba |
|---|---|
| `n_estimators` | 50, 100, 200 |
| `learning_rate` | 0.01, 0.1, 0.5 |
| `max_depth` | 3, 5, 7 |
| `num_leaves` (LGBM) | 20, 31, 50 |
| `subsample` (XGB) | 0.7, 0.8, 1.0 |
| `colsample_bytree` | 0.7, 0.8, 1.0 |
| `reg_alpha` | 0, 0.1, 1 |
| `reg_lambda` | 0, 0.1, 1 |

---

## 📁 File Referensi

| File | Deskripsi |
|---|---|
| `CRISP_DM_with_LGBM_and_XGB_7_Feature.ipynb` | Notebook eksperimen CRISP-DM lengkap |
| `app.py` | Aplikasi Streamlit deployment |
| `README.md` | Dokumentasi proyek ini |

---

## 👥 Kontribusi

Proyek ini terbuka untuk pengembangan lebih lanjut. Beberapa area yang dapat dikembangkan:

- Penambahan algoritma lain (Random Forest, CatBoost, dll.)
- Implementasi SMOTE untuk menangani imbalanced data
- Integrasi database untuk menyimpan hasil prediksi
- Export hasil prediksi ke format Excel/PDF
- API endpoint menggunakan FastAPI

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademik dan penelitian. Silakan gunakan dan modifikasi sesuai kebutuhan dengan mencantumkan atribusi yang sesuai.

---

<div align="center">

**🎓 Sistem Rekomendasi Beasiswa**

*Dibangun dengan CRISP-DM · LightGBM · XGBoost · Streamlit*

</div>
