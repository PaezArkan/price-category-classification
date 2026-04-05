# Price Category Classification — Bengkel Las

Pipeline machine learning end-to-end untuk mengklasifikasikan harga jasa bengkel las ke dalam 3 kategori: **Rendah**, **Sedang**, dan **Tinggi** — berdasarkan karakteristik order seperti jenis material, ukuran, metode hitung, finishing, dan kerumitan desain.

---

## Hasil Model

| Metric | Score |
|--------|-------|
| Test Accuracy | **90.83%** |
| Weighted F1-Score | **90.79%** |
| CV Accuracy (5-Fold StratifiedKFold) | **86.88% ± 7.29%** |

---

## Dataset

- **600 baris** order bengkel las tahun 2020–2024
- **20 kolom** fitur: material, ukuran, ketebalan, metode hitung, finishing, kerumitan desain, harga final
- Produk: Pintu Handerson, Railing, Pagar, Pintu Gerbang, Teralis
- Metode hitung: PER-M2, PER-M, PER-LUBANG
- Lokasi: area Tegal, Slawi, Banyumas, Purwokerto (Jawa Tengah)

---

## Workflow

```
1. Load & Exploratory Data Analysis (EDA)
2. Feature Engineering
   - total_area = ukuran × jumlah_unit
   - total_lubang = jumlah_lubang × jumlah_unit
   - is_per_m2, is_per_lubang (binary flags dari metode_hitung)
   - material_thickness_area = ketebalan × ukuran
3. Label Creation — KMeans Clustering + manual threshold
   (Silhouette Score digunakan untuk validasi jumlah cluster)
4. Preprocessing Pipeline
   - Numerical: StandardScaler + SimpleImputer
   - Categorical: OneHotEncoder (handle_unknown='ignore')
   - ColumnTransformer untuk apply per tipe fitur
5. Model Training — Random Forest Classifier
   - n_estimators: 200, max_depth: 15
   - StratifiedKFold (5-fold) Cross Validation
6. Model Evaluation
   - Classification Report, Confusion Matrix
   - Feature Importance Analysis
7. Inference Function & Robustness Testing
   - build_input_df_safe(): handle missing/null input
   - 3 test case: PER-M2, PER-LUBANG, random rows
8. Save Model (joblib)
```

---

## Kenapa Desain Ini?

**KMeans untuk labeling** — daripada manual threshold harga yang subjektif, KMeans menentukan batas kategori secara data-driven berdasarkan distribusi aktual.

**StratifiedKFold** — label tidak perfectly balanced (Rendah/Sedang/Tinggi tidak sama jumlahnya), jadi StratifiedKFold memastikan tiap fold representatif.

**ColumnTransformer** — fitur numerical (ukuran, harga per m2, dll) butuh StandardScaler, sedangkan categorical (jenis_material, finishing, kerumitan_desain) butuh OneHotEncoder. ColumnTransformer handle keduanya dalam satu pipeline yang bersih.

**SimpleImputer di pipeline** — form input nyata sering punya field kosong (misalnya `jumlah_lubang` kosong untuk order PER-M2). Imputer memastikan model tidak crash dan tetap prediksi konsisten.

---

## Tools & Library

```
Python 3.10 (Google Colab)
pandas, numpy
scikit-learn (RandomForestClassifier, KMeans, Pipeline, ColumnTransformer, StratifiedKFold)
matplotlib, seaborn
joblib
```

---

## Cara Menjalankan

1. Buka `ml_classification_pipeline63.ipynb` di [Google Colab](https://colab.research.google.com)
2. Jalankan cell pertama (import libraries)
3. Upload file CSV dataset saat diminta di cell **Load Dataset**
4. Jalankan semua cell secara berurutan (**Runtime → Run all**)

---

## Struktur Repo

```
price-category-classification/
├── ml_classification_pipeline63.ipynb   ← main notebook (Google Colab)
├── README.md                            ← dokumentasi ini
└── requirements.txt                     ← library yang dibutuhkan
```

---

## Requirements

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
joblib>=1.2.0
```
