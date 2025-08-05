# Sistem Penilaian Esai Otomatis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-brightgreen)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.2-orange)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-ff6f00)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

Proyek ini adalah sebuah API yang dibangun dengan FastAPI untuk melakukan penilaian esai secara otomatis. Sistem ini mendukung dua jenis model Machine Learning: **XGBoost** (cepat dan efisien) dan **Deep Learning** (potensi akurasi lebih tinggi), yang dapat dipilih melalui konfigurasi.

---

## Daftar Isi

- [Sistem Penilaian Esai Otomatis](#sistem-penilaian-esai-otomatis)
  - [Daftar Isi](#daftar-isi)
  - [Tentang Proyek](#tentang-proyek)
    - [Cara Kerja](#cara-kerja)
    - [Teknologi yang Digunakan](#teknologi-yang-digunakan)
  - [Memulai](#memulai)
    - [Prasyarat](#prasyarat)
    - [Instalasi](#instalasi)
  - [Penggunaan](#penggunaan)
    - [Menghasilkan Jawaban Baru (Opsional)](#menghasilkan-jawaban-baru-opsional)

---

## Tentang Proyek

Tujuan utama proyek ini adalah untuk mengotomatisasi proses penilaian esai, memberikan skor yang konsisten dan objektif. API ini dapat diintegrasikan ke dalam sistem e-learning atau platform ujian online, dengan fleksibilitas untuk memilih arsitektur model yang paling sesuai.

### Cara Kerja

1.  **Orkestrasi Pelatihan**: Skrip `training/train.py` bertindak sebagai orkestrator. Ia menggunakan `DataManager` untuk mengambil data dari database.
2.  **Rekayasa Fitur**: Untuk setiap grup data, `FeatureEngineer` digunakan untuk membuat fitur numerik berdasarkan kemiripan kosinus (cosine similarity) antara jawaban siswa dan jawaban referensi menggunakan TF-IDF.
3.  **Pelatihan & Evaluasi Model**: `ModelTrainer` yang sesuai (misalnya, `XGBoostTrainer` atau `DeepLearningTrainer`) dipanggil. Kelas ini menangani evaluasi (seperti GridSearchCV atau Cross-Validation) dan melatih model final.
4.  **Penyimpanan Artefak**: `FeatureEngineer` yang telah di-fit dan model yang telah dilatih disimpan sebagai artefak terpisah di dalam direktori `app/models/` yang sesuai.
5.  **Ekspor Data**: Selama proses pelatihan, data mentah yang diambil dari database untuk setiap grup diekspor ke file Excel di direktori `data_exports/` untuk tujuan inspeksi dan analisis manual.
6.  **Penilaian via API**: Saat startup, API (`app/main.py`) memuat artefak model yang relevan berdasarkan variabel lingkungan `MODEL_TYPE`. Endpoint `/score` menggunakan `FeatureEngineer` untuk memproses input baru dan model untuk menghasilkan prediksi skor.

### Teknologi yang Digunakan

- **Backend**: FastAPI
- **Machine Learning**: Scikit-learn, Pandas, NumPy, XGBoost, TensorFlow/Keras
- **Generative AI**: Google Gemini Pro
- **Database**: MySQL (diakses melalui SQLAlchemy)
- **Otomatisasi**: GNU Make
- **Kontainerisasi**: Docker, Docker Compose

---

## Memulai

Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal.

### Prasyarat

- Python 3.11+
- [Poetry](https://python-poetry.org/docs/#installation) untuk manajemen dependensi
- `make` (biasanya sudah terinstal di Linux/macOS)
- Docker & Docker Compose (opsional, untuk menjalankan via kontainer)

### Instalasi

1.  **Clone repository ini:**
    ```sh
    git clone <URL_REPOSITORY_ANDA>
    cd essay-scoring
    ```

2.  **Buat dan konfigurasikan file lingkungan:**
    Salin file contoh `.env.example` menjadi `.env` dan isi dengan kredensial database Anda.
    **Penting:**
    - Atur variabel `MODEL_TYPE` ke `xgboost` atau `deep-learning` untuk menentukan model mana yang akan digunakan oleh API.
    - Jika Anda ingin menggunakan fitur pembuatan jawaban, isi `GEMINI_API_KEY` dengan API key dari Google AI Studio.
	```sh
    cp .env.example .env
    ```
    Edit file `.env` dengan editor teks favorit Anda.

3.  **Instal dependensi:**
    Perintah ini akan secara otomatis membuat lingkungan virtual di dalam direktori proyek (`.venv/`) dan menginstal semua paket yang dibutuhkan menggunakan Poetry.
    ```sh
    make setup
    ```

---

## Penggunaan

### Menghasilkan Jawaban Baru (Opsional)

Anda dapat menggunakan AI untuk membuat jawaban referensi berkualitas tinggi untuk semua soal di database Anda.
```sh
make generate-answers
