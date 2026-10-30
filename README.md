# Fraud Detection Dashboard

Dashboard interaktif untuk mengeksplorasi pola transaksi fraud menggunakan Python dan Streamlit. Proyek ini bertujuan untuk menganalisis karakteristik transaksi berdasarkan kategori merchant, nominal transaksi, dan waktu transaksi guna membantu mengidentifikasi pola yang berpotensi memerlukan perhatian lebih lanjut.

Dataset yang digunakan merupakan **dataset sintetis dari Kaggle**, sehingga hasil analisis digunakan sebagai simulasi eksplorasi data dan bukan representasi langsung dari aktivitas fraud pada institusi keuangan nyata.

## Project Overview

Fraud merupakan salah satu permasalahan yang dapat menimbulkan risiko finansial dalam aktivitas transaksi. Melalui proyek ini, dilakukan eksplorasi data untuk memahami distribusi transaksi fraud dan mengidentifikasi karakteristik yang dapat digunakan sebagai dasar penentuan prioritas analisis.

Dashboard dikembangkan menggunakan Streamlit agar hasil eksplorasi data dapat disajikan secara interaktif, informatif, dan mudah dipahami.

## Dataset

- **Source:** [Kaggle](https://www.kaggle.com/datasets/samayashar/fraud-detection-transactions-dataset/data)
- **Type:** Synthetic Transaction Dataset
- **Total Transactions:** 50,000
- **Fraud Transactions:** 16,067 (32.13%)
- **Non-Fraud Transactions:** 33,933 (67.87%)

## Fitur Dashboard

Dashboard terdiri dari beberapa bagian analisis:

### 1. Dataset Overview
Menampilkan gambaran umum dataset, termasuk jumlah data, karakteristik fitur, dan informasi dasar transaksi.

### 2. Algorithm, Model & Evaluation
Bagian untuk menyajikan tahapan pemodelan dan evaluasi yang tersedia dalam proyek, sebagai bagian dari proses analisis fraud.

### 3. Fraud Analysis

Bagian utama dashboard yang berfokus pada eksplorasi karakteristik transaksi fraud.

**A. Transaction Label Distribution**

Menganalisis distribusi transaksi Fraud dan Non-Fraud untuk memahami proporsi masing-masing kelas.

**B. Fraud by Merchant Category**

Membandingkan jumlah transaksi fraud dan fraud rate pada setiap kategori merchant untuk mengidentifikasi kategori yang layak ditinjau lebih lanjut.

**C. Transaction Amount Analysis**

Menganalisis hubungan antara nominal transaksi dan label fraud melalui:
- Distribusi nominal transaksi.
- Perbandingan karakteristik nominal transaksi Fraud dan Non-Fraud.
- Analisis fraud rate berdasarkan interval nominal transaksi.

**D. Transaction Time Analysis**

Mengeksplorasi pola transaksi fraud berdasarkan jam transaksi selama 24 jam untuk melihat variasi jumlah transaksi dan fraud rate pada setiap jam.

### 4. Fokus Analisis

Sebagai pendalaman dari analisis sebelumnya, kategori Restaurants dipilih sebagai objek analisis lanjutan.

Analisis difokuskan pada:
- Jumlah transaksi dan fraud rate.
- Distribusi nominal transaksi berdasarkan label.
- Pola fraud pada rentang waktu pukul 03.00–09.00 dan 15.00–19.00.
- Total nominal transaksi yang terindikasi fraud berdasarkan jam transaksi.

Pendalaman ini bertujuan untuk melihat pola yang lebih spesifik serta menentukan kelompok transaksi yang layak mendapatkan perhatian lebih lanjut.

## Temuan Utama

Beberapa temuan dari eksplorasi dataset:

- Dataset memiliki proporsi transaksi fraud sebesar 32.13% dari total 50,000 transaksi.
- Restaurants memiliki fraud rate tertinggi di antara kategori merchant, yaitu 32.63%.
- Perbedaan fraud rate antar kategori merchant relatif kecil, sehingga kategori dengan nilai tertinggi tidak dapat langsung dianggap memiliki risiko yang jauh lebih besar.
- Analisis nominal transaksi menunjukkan bahwa fraud rate pada sebagian besar interval nominal relatif berdekatan.
- Analisis waktu transaksi digunakan untuk mengidentifikasi variasi pola fraud berdasarkan jam.
- Pendalaman Restaurants menggabungkan jumlah kasus, fraud rate, dan total nominal transaksi fraud untuk membantu menentukan prioritas analisis.

## Tools & Teknologi

- **Python** – Data processing and analysis
- **Pandas** – Data manipulation
- **NumPy** – Numerical computation
- **Matplotlib** – Data visualization
- **Seaborn** – Statistical data visualization
- **Streamlit** – Interactive dashboard development
- **Altair** - Modern and Responsive Data Visualization

## Struktur Folder

Contoh struktur direktori proyek:

```text
Fraud-Detection-Dashboard/
│
├── dataset/
│   └── synthetic_fraud_dataset.csv
│
├── notebook/
│   └── app.py
│   └── main.ipynb
|
├── app.py
├── *.pkl
├── README.md
```

*Sesuaikan struktur direktori di atas dengan file aktual pada repository.*

## Panduan Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/kumaragp/fraud-detection.git
```

### 2. Navigate to Project Directory

```bash
cd fraud-detection
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run Streamlit Application

```bash
streamlit run app.py
```

Dashboard akan berjalan secara lokal dan dapat diakses melalui browser.

## Batasan Project

- Dataset yang digunakan bersifat sintetis, sehingga pola yang ditemukan belum tentu mencerminkan kondisi transaksi fraud di dunia nyata.
- Fraud rate menunjukkan proporsi transaksi yang terindikasi fraud, bukan probabilitas kerugian finansial aktual.
- Total nominal transaksi fraud digunakan sebagai gambaran nilai transaksi yang terindikasi fraud dan tidak dapat langsung disamakan dengan kerugian aktual.
- Hasil analisis bersifat eksploratif dan tidak menunjukkan hubungan sebab-akibat.

## Objektif Project

Proyek ini dikembangkan sebagai bagian dari pembelajaran dan pengembangan kemampuan dalam bidang **Data Analyst dan Data Science**, khususnya dalam proses data exploration, data visualization, fraud pattern analysis, serta penyajian hasil analisis melalui dashboard interaktif.

---

**Author:** [@kumaragp](https://github.com/kumaragp)