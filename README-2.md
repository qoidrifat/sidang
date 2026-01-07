---
title: Sidang Skripsi
emoji: 🦀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/675088c04af97b1321dc8030/0-rbP0MXmtQjWgX9Tpuuy.png
---

# **Facial Expression Analysis System**

### **Transfer Learning VGG16 Terintegrasi Squeeze-and-Excitation (SE-Block)**

## **📌 Tentang Proyek**

Repositori ini merupakan implementasi sistem **Skripsi** untuk klasifikasi ekspresi wajah menggunakan dataset **FER-2013**. Sistem ini membandingkan kinerja dua skenario arsitektur berbasis VGG16 untuk membuktikan efektivitas metode yang diusulkan (SE-Block \+ Fine-Tuning).

Aplikasi ini dibangun dengan antarmuka **Modern Dark Theme** yang profesional, responsif (mobile-friendly), dan interaktif.

## **👥 Identitas Peneliti**

| Peran | Nama / Gelar |
| :---- | :---- |
| **Mahasiswa** | **Qoid Rif'at** (NIM: 210411100160\) |
| **Instansi** | Universitas Trunojoyo Madura |
| **Dospem 1** | Prof. Dr. Arif Muntasa, S.Si., M.T. |
| **Dospem 2** | Fifin Ayu Mufarroha, M.Kom. |

## **🌟 Fitur Utama Aplikasi**

Aplikasi web ini memiliki 5 modul utama yang dapat diakses melalui Sidebar Navigasi:

1. **📂 Deskripsi Dataset**  
   * Menampilkan sampel citra asli dari dataset FER-2013.  
   * Informasi statistik dataset (7 kelas, resolusi 48x48).  
2. **⚙️ Preprocessing Data (Interaktif)**  
   * **Fitur Baru:** Pengguna dapat mengunggah citra sendiri untuk melihat simulasi *pipeline* preprocessing.  
   * Visualisasi tahapan: *Raw Grayscale* ➔ *Resize RGB (224x224)* ➔ *Augmentasi (Rotasi/Flip)*.  
3. **📊 Hasil Klasifikasi**  
   * Tabel komparasi parameter teknis antara **Skenario 1 (Baseline)** dan **Skenario 2 (Optimized)**.  
   * Kesimpulan model terbaik berdasarkan metrik evaluasi.  
4. **📈 Implementasi & Grafik**  
   * Visualisasi kurva pembelajaran (*Learning Curve*) untuk akurasi Training vs Validation.  
   * Analisis performa model terhadap *overfitting*.  
5. **🤖 Demo Prediksi (Real-time)**  
   * Pengujian inferensi langsung menggunakan citra wajah pengguna (Webcam/Upload).  
   * Menampilkan *Confidence Score* dari kedua model secara berdampingan.

## **🛠️ Panduan Instalasi (Menjalankan di Lokal)**

Jika Anda ingin menjalankan sistem ini di komputer lokal (Laptop/PC), ikuti langkah-langkah berikut:

### **1\. Prasyarat Sistem**

* Python 3.10 atau lebih baru.  
* Git.

### **2\. Clone Repository**

Buka terminal/command prompt dan jalankan:

git clone \[https://huggingface.co/spaces/qoidrifat/sidang\](https://huggingface.co/spaces/qoidrifat/sidang)  
cd sidang

### **3\. Instal Dependensi**

Sangat disarankan menggunakan *Virtual Environment* (venv).

\# Membuat virtual environment  
python \-m venv venv

\# Mengaktifkan venv (Windows)  
venv\\Scripts\\activate

\# Mengaktifkan venv (Mac/Linux)  
source venv/bin/activate

\# Instal library yang dibutuhkan  
pip install \-r requirements.txt

### **4\. Menjalankan Aplikasi**

python app.py

Tunggu beberapa saat hingga muncul link lokal, biasanya: http://127.0.0.1:7860. Buka link tersebut di browser Anda.

## **🔬 Metodologi Penelitian**

Penelitian ini membandingkan dua pendekatan:

| Fitur | Skenario 1 (Baseline) | Skenario 2 (Optimized / Usulan) |
| :---- | :---- | :---- |
| **Arsitektur Dasar** | VGG16 (ImageNet Weights) | VGG16 (ImageNet Weights) |
| **Integrasi** | \- | **Squeeze-and-Excitation (SE) Block** |
| **Strategi Training** | **Frozen** (Backbone Beku) | **Aggressive Fine-Tuning** (Unfreeze Layer 11-19) |
| **Optimasi Data** | Tanpa Augmentasi | Augmentasi Geometris \+ Class Weights |
| **Loss Function** | Categorical Crossentropy | Crossentropy \+ **Label Smoothing (0.1)** |

**Novelty:** Integrasi SE-Block pada VGG16 yang di-*fine-tune* memungkinkan model mengalibrasi ulang bobot fitur channel-wise, meningkatkan sensitivitas terhadap fitur mikro wajah (mata, mulut) meskipun dengan data terbatas dan tidak seimbang.

## **📄 Lisensi**

Sistem ini dikembangkan sebagai bagian dari tugas akhir skripsi di **Program Studi Teknik Informatika, Universitas Trunojoyo Madura**.

Kode sumber dan model yang disertakan dilisensikan di bawah **MIT License**. Penggunaan kembali untuk keperluan akademis diperbolehkan dengan mencantumkan sitasi yang sesuai.

*© 2025 Qoid Rif'at. All Rights Reserved.*