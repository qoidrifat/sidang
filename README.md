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

# **🎓 Sistem Klasifikasi Ekspresi Wajah Terintegrasi (VGG16 \+ SE-Block)**

## **📌 Deskripsi Proyek**

Repositori ini berisi **Sistem Demonstrasi Sidang Skripsi** yang mengimplementasikan pendekatan *Deep Learning* untuk pengenalan ekspresi wajah (Facial Expression Recognition) pada dataset **FER-2013**.

Sistem ini dirancang dengan antarmuka web interaktif yang komprehensif, memuat tidak hanya fitur prediksi, tetapi juga visualisasi dataset, alur preprocessing, dan analisis performa model secara mendalam. Inti penelitian ini membandingkan dua skenario arsitektur berbasis **VGG16** untuk membuktikan efektivitas integrasi **Squeeze-and-Excitation (SE) Block**.

## **👥 Identitas Peneliti**

| Peran | Nama / Gelar |
| :---- | :---- |
| **Mahasiswa** | Qoid Rif'at |
| **NIM** | 210411100160 |
| **Instansi** | Universitas Trunojoyo Madura |
| **Dospem 1** | Prof. Dr. Arif Muntasa, S.Si., M.T. |
| **Dospem 2** | Fifin Ayu Mufarroha, M.Kom. |

## **🚀 Fitur Utama Sistem**

Aplikasi ini memiliki **5 Modul Utama** yang dapat diakses melalui Sidebar Navigasi:

### **1\. 📂 Deskripsi Dataset**

* Menampilkan sampel citra asli dari dataset **FER-2013**.  
* Memvisualisasikan distribusi 7 kelas emosi: *Marah, Jijik, Takut, Senang, Sedih, Terkejut, Netral*.  
* Menyajikan statistik dataset (resolusi asli 48x48 piksel, format grayscale).

### **2\. ⚙️ Preprocessing Data Visualizer**

* Mendemonstrasikan tahapan transformasi citra sebelum masuk ke model CNN.  
* **Pipeline:** Citra Asli (48x48) ➔ Resize (224x224) ➔ Normalisasi ➔ Augmentasi (Rotasi, Flip).  
* Penjelasan teknik *Label Smoothing* untuk mencegah overfitting.

### **3\. 📊 Hasil & Parameter Klasifikasi**

* Tabel komparasi parameter teknis antara **Skenario 1 (Baseline)** dan **Skenario 2 (Optimized)**.  
* Ringkasan performa model terbaik berdasarkan metrik evaluasi (Akurasi, Presisi, Recall, F1-Score).

### **4\. 📈 Implementasi & Grafik**

* Visualisasi grafik kurva pembelajaran (*Learning Curves*) secara interaktif menggunakan matplotlib.  
* Membandingkan akurasi *Training* vs *Validation* antar skenario untuk analisis *underfitting/overfitting*.

### **5\. 🤖 Demo Prediksi (Real-time Inference)**

* Modul pengujian langsung menggunakan citra wajah yang diunggah pengguna.  
* Menampilkan hasil prediksi **Side-by-Side** antara model Baseline dan Optimized untuk menunjukkan peningkatan performa secara nyata.

## **🔬 Metodologi Penelitian**

Penelitian ini membandingkan dua skenario arsitektur utama:

| Fitur | Skenario 1 (Baseline) | Skenario 2 (Optimized / Usulan) |
| :---- | :---- | :---- |
| **Arsitektur Dasar** | VGG16 (ImageNet Weights) | VGG16 (ImageNet Weights) |
| **Integrasi Khusus** | \- | **Squeeze-and-Excitation (SE) Block** |
| **Strategi Training** | **Frozen** (Backbone Beku) | **Aggressive Fine-Tuning** (Unfreeze Layer 11-19) |
| **Optimasi Data** | Tanpa Augmentasi | Augmentasi Geometris \+ Class Weights |
| **Loss Function** | Categorical Crossentropy | Crossentropy \+ **Label Smoothing (0.1)** |

**🌟 Novelty (Kebaruan):** Integrasi mekanisme atensi **SE-Block** yang disisipkan pada arsitektur VGG16 yang telah di-*fine-tune*, memungkinkan model untuk secara adaptif mengalibrasi ulang bobot fitur channel-wise, sehingga lebih sensitif terhadap fitur wajah mikro (mata, mulut) meskipun dengan data terbatas.

## **🛠️ Instalasi & Penggunaan Lokal**

Jika Anda ingin menjalankan sistem ini di komputer lokal:

**1\. Clone Repository**

git clone \[https://huggingface.co/spaces/username/nama-repo\](https://huggingface.co/spaces/username/nama-repo)  
cd nama-repo

2\. Instal Dependensi  
Pastikan Python 3.10 terinstal, lalu jalankan:  
pip install \-r requirements.txt

3\. Siapkan Struktur Folder (Opsional)  
Untuk fitur "Deskripsi Dataset" agar berfungsi optimal, buat struktur folder berikut:  
/  
├── app.py  
├── requirements.txt  
├── model\_scenario1.keras  
├── best\_model\_scenario2.keras  
└── dataset/  
    ├── marah/      \# (isi dengan 5-10 sampel gambar .jpg)  
    ├── senang/  
    └── ... (kelas lainnya)

**4\. Jalankan Aplikasi**

python app.py

Akses aplikasi melalui browser di http://localhost:7860.

## **📄 Lisensi & Kredit**

Sistem ini dikembangkan sebagai bagian dari tugas akhir skripsi di **Program Studi Teknik Informatika, Universitas Trunojoyo Madura**.

Kode sumber dan model yang disertakan dilisensikan di bawah **MIT License**. Penggunaan kembali untuk keperluan akademis diperbolehkan dengan mencantumkan sitasi yang sesuai.

*© 2025 Qoid Rif'at. All Rights Reserved.*