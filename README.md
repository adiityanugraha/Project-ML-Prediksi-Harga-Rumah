# Prediksi Harga Rumah Jabodetabek — Clustering & Klasifikasi

Proyek submission akhir kelas **Belajar Machine Learning untuk Pemula (BMLP) — Dicoding**, oleh Anak Agung Aryadipa Aditya Nugraha.

Proyek ini mengelompokkan rumah di wilayah Jabodetabek ke dalam segmen-segmen harga menggunakan **K-Means Clustering**, lalu membangun model **klasifikasi** yang mampu memprediksi segmen tersebut dengan akurasi **97,26%**.

## Alur Kerja

Proyek terdiri dari dua tahap yang saling berhubungan:

```
jabodetabek_house_price.csv
        │
        ▼
[1] Notebook Clustering ──► data_clustering_inverse.csv (data + label cluster)
        │                            │
        ▼                            ▼
  4 segmen rumah            [2] Notebook Klasifikasi ──► model prediksi segmen
```

### Tahap 1 — Clustering (Unsupervised Learning)

Notebook: `[Clustering]_Submission_Akhir_BMLP_....ipynb`

1. **EDA** — statistik deskriptif, matriks korelasi, histogram fitur numerik & kategorikal, boxplot untuk deteksi outlier.
2. **Preprocessing** — hapus missing value & duplikat, drop kolom tidak relevan (url, ads_id, address, lat, long), Label Encoding untuk fitur kategorikal, penanganan outlier dengan metode IQR, StandardScaler untuk fitur numerik, serta binning pada `price_in_rp` dan `building_size_m2`.
3. **Modeling** — penentuan jumlah cluster optimal dengan **Elbow Method** (hasil: **K = 4**), lalu K-Means Clustering. Dicoba juga versi dengan **PCA** (8 komponen, 95% variance).
4. **Evaluasi** — Silhouette Score: **0.2629** (tanpa PCA) dan **0.2812** (dengan PCA).
5. **Interpretasi** — analisis karakteristik tiap cluster (mean/min/max per fitur), termasuk setelah data dikembalikan ke skala asli dengan `inverse_transform()`.

### Tahap 2 — Klasifikasi (Supervised Learning)

Notebook: `[Klasifikasi]_Submission_Akhir_BMLP_....ipynb`

Label cluster hasil tahap 1 dijadikan target klasifikasi. Data dibagi 80:20 (train:test, stratified), lalu dilatih dan dibandingkan tiga model:

| Model | Akurasi | Presisi | Recall | F1-Score |
|---|---|---|---|---|
| Decision Tree | 96,58% | 96,59% | 96,58% | 96,57% |
| Random Forest | **97,26%** | **97,26%** | **97,26%** | **97,26%** |
| Tuned Random Forest (GridSearchCV) | 97,26% | 97,26% | 97,26% | 97,26% |

Hyperparameter tuning dilakukan dengan **GridSearchCV** (3-fold CV, 18 kombinasi parameter). Parameter terbaik: `n_estimators=50, max_depth=None, min_samples_split=2`.

## Dataset

- **Sumber:** `jabodetabek_house_price.csv` — data harga rumah wilayah Jakarta, Bogor, Depok, Tangerang, Bekasi.
- **Fitur utama:** harga (`price_in_rp`), jumlah kamar tidur/mandi, luas tanah & bangunan, carport, jumlah lantai, umur bangunan, kota/kecamatan, tipe properti, sertifikat, kondisi bangunan, dll.

## Struktur File

| File | Keterangan |
|---|---|
| `[Clustering]_...ipynb` | Notebook tahap 1: EDA, preprocessing, K-Means |
| `[Klasifikasi]_...ipynb` | Notebook tahap 2: Decision Tree, Random Forest, tuning |
| `jabodetabek_house_price.csv` | Dataset mentah |
| `data_clustering.csv` | Data hasil preprocessing (skala standar) + label cluster |
| `data_clustering_inverse.csv` | Data skala asli + label cluster (input tahap 2) |
| `model_clustering.h5` | Model K-Means |
| `PCA_model_clustering.h5` | Model K-Means dengan PCA |
| `decision_tree_model.h5` | Model Decision Tree |
| `explore_RandomForest_classification.h5` | Model Random Forest |
| `tuning_classification.h5` | Model Random Forest hasil tuning (terbaik) |

## Cara Menjalankan

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yellowbrick joblib
```

Jalankan notebook secara berurutan (clustering dulu, karena outputnya menjadi input klasifikasi):

1. `[Clustering]_Submission_Akhir_BMLP_Anak_Agung_Aryadipa_Aditya_Nugraha.ipynb`
2. `[Klasifikasi]_Submission_Akhir_BMLP_Anak_Agung_Aryadipa_Aditya_Nugraha.ipynb`

## Teknologi

Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn · Yellowbrick · joblib
