---
title: Sidang Skripsi
emoji: 🦀
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.2.0
app_file: app.py
pinned: false
---

# **Klasifikasi Ekspresi Wajah Menggunakan Transfer Learning Berbasis DeepFace Terintegrasi CNN**

## **📌 Deskripsi Proyek**

Repositori ini merupakan **demo interaktif** untuk kebutuhan Sidang Skripsi. Sistem ini mengimplementasikan model *Deep Learning* untuk mengenali 7 kelas ekspresi wajah (Marah, Jijik, Takut, Senang, Sedih, Terkejut, Netral) menggunakan dataset **FER-2013**.

Penelitian ini membandingkan kinerja dua skenario arsitektur berbasis **VGG16** (backbone dari framework DeepFace) untuk membuktikan efektivitas metode yang diusulkan.

### **👥 Identitas Peneliti**

* **Nama:** Qoid Rif'at  
* **NIM:** 210411100160  
* **Instansi:** Universitas Trunojoyo Madura  
* **Dospem 1:** Prof. Dr. Arif Muntasa, S.Si., M.T.  
* **Dospem 2:** Fifin Ayu Mufarroha, M.Kom.

## **🔬 Skenario Pengujian**

Aplikasi ini memuat dua model secara bersamaan untuk perbandingan *side-by-side*:

| Fitur | Skenario 1 (Baseline) | Skenario 2 (Optimized / Usulan) |
| :---- | :---- | :---- |
| **Arsitektur Dasar** | VGG16 (ImageNet Weights) | VGG16 (ImageNet Weights) |
| **Strategi Training** | **Frozen** (Semua layer beku) | **Aggressive Fine-Tuning** (Unfreeze Layer 11-19) |
| **Integrasi Khusus** | Tidak Ada | **Squeeze-and-Excitation (SE) Block** |
| **Optimasi Data** | Tanpa Augmentasi | Augmentasi Geometris \+ Class Weights |
| **Regularisasi** | Dropout Standar | Label Smoothing (0.1) |

**Novelty:** Skenario 2 mengintegrasikan mekanisme atensi *SE-Block* untuk meningkatkan sensitivitas model terhadap fitur wajah penting (mata/mulut) dan melakukan *fine-tuning* pada layer konvolusi tingkat tinggi.

## **🚀 Cara Penggunaan**

1. **Upload Gambar:** Unggah foto wajah yang ingin diuji pada panel sebelah kiri. Pastikan wajah terlihat jelas.  
2. **Klik Prediksi:** Tekan tombol "Prediksi Ekspresi".  
3. **Analisis Hasil:**  
   * Lihat output **Skenario 1**: Biasanya kurang akurat atau probabilitasnya rendah karena hanya menggunakan fitur generik.  
   * Lihat output **Skenario 2**: Seharusnya memberikan prediksi yang lebih akurat dengan tingkat keyakinan (confidence) yang lebih tinggi.

## **🛠️ Dependensi (Requirements)**

Sistem ini dibangun menggunakan pustaka utama berikut:

* tensorflow: Untuk memuat model .keras dan komputasi tensor.  
* gradio: Untuk antarmuka web interaktif.  
* numpy: Untuk manipulasi array citra.  
* pillow: Untuk pemrosesan citra dasar.

## **📄 Lisensi**

Penelitian ini disusun untuk keperluan akademis di Universitas Trunojoyo Madura. Segala bentuk penggunaan ulang kode atau model harus mencantumkan sitasi yang sesuai.