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
    - [Melatih Model](#melatih-model)
      - [Model XGBoost](#model-xgboost)
    - [Menjalankan API](#menjalankan-api)
    - [Contoh Permintaan API](#contoh-permintaan-api)
  - [Struktur Proyek](#struktur-proyek)
  - [Daftar Perintah Makefile](#daftar-perintah-makefile)

---

## Tentang Proyek

Tujuan utama proyek ini adalah untuk mengotomatisasi proses penilaian esai, memberikan skor yang konsisten dan objektif. API ini dapat diintegrasikan ke dalam sistem e-learning atau platform ujian online, dengan fleksibilitas untuk memilih arsitektur model yang paling sesuai.

### Cara Kerja

1.  **Orkestrasi Pelatihan**: Skrip `training/train.py` bertindak sebagai orkestrator. Ia menggunakan `DataManager` untuk mengambil data dari database.
2.  **Rekayasa Fitur**: Untuk setiap grup data, `FeatureEngineer` digunakan untuk membuat fitur numerik berdasarkan kemiripan kosinus (cosine similarity) antara jawaban siswa dan jawaban referensi menggunakan TF-IDF.
3.  **Pelatihan & Evaluasi Model**: `ModelTrainer` yang sesuai (misalnya, `XGBoostTrainer` atau `DeepLearningTrainer`) dipanggil. Kelas ini menangani evaluasi (seperti GridSearchCV atau Cross-Validation) dan melatih model final.
4.  **Penyimpanan Artefak**: `FeatureEngineer` yang telah di-fit dan model yang telah dilatih disimpan sebagai artefak terpisah di dalam direktori `app/models/` yang sesuai.
5.  **Penilaian via API**: Saat startup, API (`app/main.py`) memuat artefak model yang relevan berdasarkan variabel lingkungan `MODEL_TYPE`. Endpoint `/score` menggunakan `FeatureEngineer` untuk memproses input baru dan model untuk menghasilkan prediksi skor.

### Teknologi yang Digunakan

- **Backend**: FastAPI
- **Machine Learning**: Scikit-learn, Pandas, NumPy, XGBoost, TensorFlow/Keras
- **Database**: MySQL (diakses melalui SQLAlchemy)
- **Otomatisasi**: GNU Make
- **Kontainerisasi**: Docker, Docker Compose

---

## Memulai

Ikuti langkah-langkah berikut untuk menjalankan proyek ini secara lokal.

### Prasyarat

- Python 3.10+
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
    **Penting:** Atur variabel `MODEL_TYPE` ke `xgboost` atau `deep-learning` untuk menentukan model mana yang akan digunakan oleh API.
	```sh
    cp .env.example .env
    ```
    Edit file `.env` dengan editor teks favorit Anda.

3.  **Setup lingkungan virtual dan instal dependensi:**
    Perintah ini akan membuat lingkungan virtual Python di dalam direktori `venv/` dan menginstal semua paket yang dibutuhkan dari `requirements.txt`.
    ```sh
    make setup
    ```

---

## Penggunaan

### Melatih Model

Anda dapat melatih model XGBoost atau Deep Learning secara terpisah.

#### Model XGBoost
- **Latih semua model XGBoost yang belum ada:**
  ```sh
  make train
  ```
- **Latih satu model berikutnya secara manual:**
  Perintah ini akan melatih satu model yang belum ada, lalu berhenti. Jalankan berulang kali untuk melatih satu per satu.
  ```sh
  make train-next
  ```
- **Paksa pelatihan ulang semua model:**
  ```sh
  make update-models
  ```

### Menjalankan API

- **Mode Pengembangan (Lokal):**
  Menjalankan server menggunakan Uvicorn dengan fitur *hot-reload*.
  ```sh
  make run
  ```
  API akan tersedia di `http://127.0.0.1:8000`.

- **Mode Produksi (Docker):**
  Membangun dan menjalankan aplikasi di dalam kontainer Docker.
  ```sh
  make docker-up
  ```
  API akan tersedia di `http://localhost:8000`.

Setelah server berjalan, Anda dapat mengakses dokumentasi API interaktif di `http://localhost:8000/docs`.

### Contoh Permintaan API

Anda dapat menggunakan `curl` untuk menguji endpoint `/score`:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/score' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "subject": "biologi",
  "grade_level": "kelas xii",
  "question_id": 123,
  "answer_text": "Mitokondria adalah organel sel yang berfungsi sebagai tempat respirasi seluler."
}'
```

**Contoh Respons:**
```json
{
  "score": 8.75
}
```

---

## Struktur Proyek

```
essay-scoring/
├── app/
│   ├── models/         # Direktori untuk menyimpan file model (.joblib)
│   └── main.py         # Logika utama aplikasi FastAPI
├── queries/
│   ├── get_reference_answers.sql # Kueri untuk mengambil jawaban referensi
│   └── get_training_data.sql     # Kueri untuk mengambil data pelatihan
├── training/
│   └── train.py        # Skrip untuk melatih model
├── .env.example        # Template untuk file variabel lingkungan
├── Dockerfile          # Instruksi untuk membangun image Docker
├── docker-compose.yml  # Konfigurasi untuk menjalankan layanan dengan Docker
├── logs/               # Direktori untuk file log (dibuat otomatis)
├── Makefile            # Perintah otomatisasi untuk pengembangan
└── requirements.txt    # Daftar dependensi Python
```

---

## Daftar Perintah Makefile

| Perintah          | Deskripsi                                                              |
|-------------------|------------------------------------------------------------------------|
| `make help`       | Menampilkan daftar semua perintah yang tersedia.                       |
| `make setup`      | Membuat lingkungan virtual dan menginstal semua dependensi.            |
| `make install`    | Menginstal atau memperbarui dependensi dari `requirements.txt`.        |
| `make train`      | Menjalankan skrip pelatihan (melewatkan model yang sudah ada).         |
| `make train-next` | Melatih satu model berikutnya yang belum ada, lalu berhenti.           |
| `make update-models`| Memaksa pelatihan ulang semua model dari awal.                         |
| `make run`        | Menjalankan server FastAPI secara lokal untuk pengembangan.            |
| `make clean`      | Menghapus lingkungan virtual dan file cache.                           |
| `make docker-build` | Membangun image Docker untuk aplikasi.                                 |
| `make docker-up`  | Menjalankan aplikasi menggunakan Docker Compose.                       |
| `make docker-down`| Menghentikan dan menghapus kontainer aplikasi.                         |
| `make docker-logs`| Menampilkan log dari kontainer yang sedang berjalan.                   |