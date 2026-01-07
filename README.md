---
title: Facial Expression Analysis System (VGG16 + SE-Block)
emoji: 🚀
colorFrom: indigo
colorTo: slate
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# 🚀 Facial Expression Analysis System  
### VGG16 Transfer Learning with Squeeze-and-Excitation (SE-Block)

## 📌 Project Overview

This repository contains the implementation of an **undergraduate thesis project** focused on **facial expression classification** using the **FER-2013 dataset**.  
The study evaluates and compares two deep learning scenarios based on the **VGG16** architecture to demonstrate the effectiveness of the proposed approach: **SE-Block integration combined with fine-tuning**.

The system is deployed as an **interactive web application** with a modern dark-themed UI, responsive design (mobile-friendly), and real-time prediction capabilities.

## 🎓 Researcher Information

| Role | Description |
|------|------------|
| **Student** | **Qoid Rif'at** |
| **Student ID (NIM)** | 210411100160 |
| **Institution** | Universitas Trunojoyo Madura |
| **Supervisor I** | Prof. Dr. Arif Muntasa, S.Si., M.T. |
| **Supervisor II** | Fifin Ayu Mufarroha, M.Kom. |

## ✨ Application Features

The web application is structured into five main modules accessible via a sidebar navigation menu:

1. **📂 Dataset Overview**  
   - Visualization of sample images from the FER-2013 dataset.  
   - Dataset statistics (7 emotion classes, 48×48 grayscale images).  

2. **⚙️ Interactive Data Preprocessing**  
   - **User-uploaded image support** for preprocessing simulation.  
   - Step-by-step visualization pipeline:  
     *Grayscale Input → RGB Resize (224×224) → Augmentation (Rotation / Flip).*  

3. **📊 Classification Results**  
   - Comparative table of technical parameters between **Scenario 1 (Baseline)** and **Scenario 2 (Optimized)**.  
   - Model performance summary based on evaluation metrics.

4. **📈 Training Analysis & Visualization**  
   - Learning curves for training and validation accuracy.  
   - Overfitting analysis and generalization discussion.

5. **🤖 Real-Time Prediction Demo**  
   - Facial expression inference via webcam or image upload.  
   - Side-by-side confidence score comparison between both models.

## 🛠️ Local Installation Guide

Follow the steps below to run the application locally on your machine.

### 1. System Requirements

- Python **3.10** or newer  
- Git  

### 2. Clone the Repository

```bash
git clone https://huggingface.co/spaces/qoidrifat/sidang
cd sidang
```

### 3. Install Dependencies

It is strongly recommended to use a virtual environment.

```bash
# Create virtual environment
python -m venv venv

# Activate on Windows
venv\Scripts\activate

# Activate on macOS / Linux
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Once the server is running, open the following URL in your browser:
```
http://127.0.0.1:7860
```

## 🔬 Research Methodology

This study compares two experimental scenarios as follows:

| Aspect | Scenario 1 (Baseline) | Scenario 2 (Proposed / Optimized) |
|------|----------------------|----------------------------------|
| **Base Architecture** | VGG16 (ImageNet weights) | VGG16 (ImageNet weights) |
| **SE-Block Integration** | No | **Yes** |
| **Training Strategy** | Frozen backbone | Fine-tuning (Unfreeze layers 11–19) |
| **Data Optimization** | None | Augmentation + Class Weights |
| **Loss Function** | Categorical Crossentropy | Crossentropy + **Label Smoothing (0.1)** |

**Research Contribution:**  
The integration of **Squeeze-and-Excitation Blocks** within a fine-tuned VGG16 architecture enables channel-wise feature recalibration. This approach improves sensitivity to subtle facial features (e.g., eyes and mouth), particularly under limited and imbalanced data conditions such as FER-2013.

## 📄 License

This project was developed as part of a **Bachelor’s Thesis** in the **Informatics Engineering Program, Universitas Trunojoyo Madura**.

The source code and accompanying models are released under the **MIT License**.  
Reuse for academic and research purposes is permitted with proper citation.

© 2025 Qoid Rif'at. All rights reserved.
