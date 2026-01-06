import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 1. DEFINISI CUSTOM LAYER (SE-BLOCK)
# ---------------------------------------------------------
# Sangat Penting: Keras perlu definisi ini untuk me-load model Skenario 2
# Jika di training Anda menggunakan function, kita define ulang disini.

def squeeze_excite_block(input_tensor, ratio=16):
    """
    Implementasi Squeeze-and-Excitation Block
    Harus sama persis dengan yang ada di file training Anda.
    """
    # Mengambil channel dari shape input
    filters = input_tensor.shape[-1]
    
    # Squeeze: Global Average Pooling
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    
    # Excitation: 2 Fully Connected Layers
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    
    # Scale: Kalibrasi ulang fitur input
    x = tf.keras.layers.Multiply()([input_tensor, se])
    return x

# Mendaftarkan custom object agar load_model mengenali 'squeeze_excite_block'
# Jika saat training namanya beda, sesuaikan string kuncinya.
custom_objects_dict = {'squeeze_excite_block': squeeze_excite_block}

# ---------------------------------------------------------
# 2. LOAD MODEL (CACHED)
# ---------------------------------------------------------
# Kita load model di awal agar tidak berat saat prediksi
print("Sedang memuat model...")

try:
    # Load Model Skenario 1 (Baseline - Tanpa SE Block biasanya)
    # Jika Skenario 1 murni VGG16 frozen, biasanya tidak butuh custom object, 
    # tapi kita pasang saja untuk jaga-jaga.
    model_s1 = tf.keras.models.load_model(
        "model_scenario1.keras", 
        compile=False
    )
    print("Model Skenario 1 berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat Model 1: {e}")
    model_s1 = None

try:
    # Load Model Skenario 2 (Optimized - Dengan SE Block)
    # Wajib pakai custom_objects
    model_s2 = tf.keras.models.load_model(
        "best_model_scenario2.keras", 
        custom_objects=custom_objects_dict,
        compile=False
    )
    print("Model Skenario 2 berhasil dimuat.")
except Exception as e:
    print(f"Gagal memuat Model 2: {e}")
    model_s2 = None

# Definisi Label Kelas (Sesuai urutan folder dataset FER-2013)
# Pastikan urutannya SAMA dengan saat training (alfabetis atau sesuai generator)
LABELS = ['Marah (Angry)', 'Jijik (Disgust)', 'Takut (Fear)', 
          'Senang (Happy)', 'Sedih (Sad)', 'Terkejut (Surprise)', 'Netral (Neutral)']

# ---------------------------------------------------------
# 3. FUNGSI PREDIKSI
# ---------------------------------------------------------
def predict_expression(image):
    if image is None:
        return None, None
    
    # -- Preprocessing (Harus SAMA PERSIS dengan Bab 3) --
    # 1. Resize ke 224x224
    image = image.resize((224, 224))
    
    # 2. Konversi ke Array & Normalisasi (Rescale 1./255)
    img_array = np.array(image)
    
    # Cek jika gambar grayscale (2D), konversi ke RGB (3D)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = img_array.astype('float32') / 255.0
    
    # 3. Expand Dims untuk Batch Size (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    
    # -- Prediksi Skenario 1 --
    if model_s1:
        pred_s1 = model_s1.predict(img_array)[0]
        # Format dictionary untuk Gradio {Label: Probabilitas}
        result_s1 = {LABELS[i]: float(pred_s1[i]) for i in range(len(LABELS))}
    else:
        result_s1 = {"Error": 0.0}

    # -- Prediksi Skenario 2 --
    if model_s2:
        pred_s2 = model_s2.predict(img_array)[0]
        result_s2 = {LABELS[i]: float(pred_s2[i]) for i in range(len(LABELS))}
    else:
        result_s2 = {"Error": 0.0}
        
    return result_s1, result_s2

# ---------------------------------------------------------
# 4. ANTARMUKA GRADIO
# ---------------------------------------------------------
title = "Demo Sidang Skripsi: Klasifikasi Ekspresi Wajah"
description = """
Aplikasi ini membandingkan performa dua skenario model Transfer Learning VGG16 pada dataset FER-2013.
* **Skenario 1 (Baseline):** VGG16 Frozen.
* **Skenario 2 (Optimized):** VGG16 Fine-Tuned + SE-Block + Augmentasi.
"""

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(f"# {title}")
    gr.Markdown(description)
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Upload Foto Wajah")
            submit_btn = gr.Button("Prediksi Ekspresi", variant="primary")
            
        with gr.Column():
            gr.Markdown("### Hasil Prediksi")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Skenario 1 (Baseline)**")
                    output_s1 = gr.Label(num_top_classes=3, label="Probabilitas S1")
                with gr.Column():
                    gr.Markdown("**Skenario 2 (Optimized)**\n*(Metode Usulan)*")
                    output_s2 = gr.Label(num_top_classes=3, label="Probabilitas S2")
    
    # Event Listener
    submit_btn.click(fn=predict_expression, inputs=input_image, outputs=[output_s1, output_s2])
    
    # Contoh gambar untuk dicoba penguji (Optional, simpan gambar di folder space)
    # gr.Examples(['sample_happy.jpg', 'sample_sad.jpg'], inputs=input_image)

# Launch App
if __name__ == "__main__":
    demo.launch()