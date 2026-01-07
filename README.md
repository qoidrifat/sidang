---
title: Sistem Analisis Ekspresi Wajah (VGG16 + SE-Block)
emoji: 🚀
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🚀 Sistem Analisis Ekspresi Wajah  
### Transfer Learning VGG16 dengan Squeeze-and-Excitation (SE-Block)

## 📌 Gambaran Umum Proyek

Repositori ini berisi implementasi **tugas akhir (skripsi)** yang berfokus pada **klasifikasi ekspresi wajah** menggunakan dataset **FER-2013**.  
Penelitian ini membandingkan dua skenario arsitektur berbasis **VGG16** untuk membuktikan efektivitas metode yang diusulkan, yaitu **integrasi SE-Block yang dikombinasikan dengan fine-tuning**.

Sistem dikembangkan dalam bentuk **aplikasi web interaktif** dengan tampilan **modern dark theme**, responsif (mobile-friendly), serta mendukung prediksi ekspresi wajah secara real-time.

## 🎓 Identitas Peneliti

| Peran | Keterangan |
|------|-----------|
| **Mahasiswa** | **Qoid Rif'at** |
| **NIM** | 210411100160 |
| **Fakultas** | Teknik |
| **Prodi** | Teknik Informatika |
| **Perguruan Tinggi** | Universitas Trunojoyo Madura |
| **Dosen Pembimbing I** | Prof. Dr. Arif Muntasa, S.Si., M.T. |
| **Dosen Pembimbing II** | Fifin Ayu Mufarroha, M.Kom. |

## ✨ Fitur Utama Aplikasi

Aplikasi web ini terdiri dari lima modul utama yang dapat diakses melalui sidebar navigasi:

1. **📂 Deskripsi Dataset**  
   - Menampilkan contoh citra dari dataset FER-2013.  
   - Informasi statistik dataset (7 kelas emosi, citra grayscale 48×48).  

2. **⚙️ Preprocessing Data Interaktif**  
   - Mendukung unggah citra oleh pengguna untuk simulasi preprocessing.  
   - Visualisasi tahapan preprocessing:  
     *Citra Grayscale → Resize ke RGB (224×224) → Augmentasi (Rotasi / Flip).*  

3. **📊 Hasil Klasifikasi**  
   - Tabel perbandingan parameter teknis antara **Skenario 1 (Baseline)** dan **Skenario 2 (Optimasi / Usulan)**.  
   - Ringkasan performa model berdasarkan metrik evaluasi.

4. **📈 Implementasi dan Visualisasi Pelatihan**  
   - Grafik learning curve (akurasi training dan validation).  
   - Analisis overfitting dan kemampuan generalisasi model.

5. **🤖 Demo Prediksi Real-Time**  
   - Pengujian inferensi ekspresi wajah melalui webcam atau unggah gambar.  
   - Perbandingan confidence score dari kedua model secara berdampingan.

## 🛠️ Panduan Instalasi (Menjalankan Secara Lokal)

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer lokal (Laptop/PC).

### 1. Prasyarat Sistem

- Python versi **3.10** atau lebih baru  
- Git  

### 2. Clone Repository

```bash
git clone https://huggingface.co/spaces/qoidrifat/sidang
cd sidang
```

### 3. Instalasi Dependensi

Disarankan menggunakan virtual environment.

```bash
# Membuat virtual environment
python -m venv venv

# Aktivasi (Windows)
venv\Scripts\activate

# Aktivasi (macOS / Linux)
source venv/bin/activate

# Instal library yang dibutuhkan
pip install -r requirements.txt
```

### 4. Menjalankan Aplikasi

```bash
python app.py
```

Setelah server berjalan, buka browser dan akses:
```
http://127.0.0.1:7860
```

## 🔬 Metodologi Penelitian

Penelitian ini membandingkan dua skenario eksperimen sebagai berikut:

| Aspek | Skenario 1 (Baseline) | Skenario 2 (Usulan / Optimasi) |
|------|----------------------|-------------------------------|
| **Arsitektur Dasar** | VGG16 (Bobot ImageNet) | VGG16 (Bobot ImageNet) |
| **Integrasi SE-Block** | Tidak | **Ya** |
| **Strategi Training** | Backbone dibekukan | Fine-tuning (Unfreeze layer 11–19) |
| **Optimasi Data** | Tidak ada | Augmentasi + Class Weights |
| **Loss Function** | Categorical Crossentropy | Crossentropy + **Label Smoothing (0.1)** |

**Kontribusi Penelitian:**  
Integrasi **Squeeze-and-Excitation Block** pada arsitektur VGG16 yang di-fine-tune memungkinkan model melakukan kalibrasi ulang bobot fitur secara channel-wise. Pendekatan ini meningkatkan sensitivitas model terhadap fitur mikro wajah (seperti mata dan mulut), khususnya pada kondisi dataset yang terbatas dan tidak seimbang seperti FER-2013.

## 📄 Lisensi

Proyek ini dikembangkan sebagai bagian dari **Tugas Akhir (Skripsi)** pada **Program Studi Teknik Informatika, Universitas Trunojoyo Madura**.

Kode sumber dan model yang disertakan dirilis di bawah **Lisensi MIT**.  
Penggunaan ulang untuk keperluan akademik dan penelitian diperbolehkan dengan mencantumkan sitasi yang sesuai.

© 2025 Qoid Rif'at. Seluruh hak cipta dilindungi.
