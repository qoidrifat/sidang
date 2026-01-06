import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. KONFIGURASI SISTEM & CUSTOM LAYER
# ==========================================

# Definisi SE-Block (Wajib ada untuk load model Skenario 2)
# Pastikan registrasi serializable agar aman
@tf.keras.utils.register_keras_serializable()
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = tf.keras.layers.Multiply()([input_tensor, se])
    return x

# Dictionary untuk custom objects saat load model
custom_objects_dict = {'squeeze_excite_block': squeeze_excite_block}

# Label Kelas (Urutan sesuai dataset FER-2013 biasanya)
LABELS = ['Marah 😡', 'Jijik 🤢', 'Takut 😱', 
          'Senang 😊', 'Sedih 😢', 'Terkejut 😲', 'Netral 😐']

# ==========================================
# 2. LOAD MODEL (DENGAN ERROR HANDLING)
# ==========================================
print("🔄 Sedang memuat model ke memori...")

def load_model_safely(path, custom_objs=None):
    try:
        # Menggunakan tf.device CPU untuk menghindari error CUDA di Space basic
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(path, custom_objects=custom_objs, compile=False)
        print(f"✅ Berhasil load: {path}")
        return model
    except Exception as e:
        print(f"❌ Gagal load {path}: {e}")
        return None

# Load kedua model
model_s1 = load_model_safely("model_scenario1.keras")
model_s2 = load_model_safely("best_model_scenario2.keras", custom_objects_dict)

# ==========================================
# 3. FUNGSI PREDIKSI PINTAR (SMART PREDICT)
# ==========================================
def preprocess_image(image, target_shape):
    """
    Menyesuaikan gambar input dengan shape yang diminta model.
    target_shape format: (None, Height, Width, Channels)
    """
    if image is None: 
        return None

    # Ambil target ukuran (H, W) dari model jika tersedia, default 224
    if target_shape and len(target_shape) >= 3:
        target_h, target_w = target_shape[1], target_shape[2]
    else:
        target_h, target_w = 224, 224
    
    # Resize Gambar
    image_resized = image.resize((target_w, target_h))
    img_array = np.array(image_resized)

    # Handling Channel (Grayscale vs RGB)
    # Jika gambar cuma 2 dimensi (H, W), jadikan (H, W, 3)
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Normalisasi (1./255)
    img_array = img_array.astype('float32') / 255.0
    
    # Expand Dimension (Batch Size) -> (1, H, W, C)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict_expression(image):
    if image is None:
        return None, None
    
    # --- PREDIKSI SKENARIO 1 ---
    if model_s1:
        try:
            input_s1 = preprocess_image(image, model_s1.input_shape)
            pred_s1 = model_s1.predict(input_s1, verbose=0)[0]
            result_s1 = {LABELS[i]: float(pred_s1[i]) for i in range(len(LABELS))}
        except Exception as e:
            result_s1 = {f"Error: {str(e)}": 0.0}
    else:
        result_s1 = {"Model 1 Tidak Ditemukan": 0.0}

    # --- PREDIKSI SKENARIO 2 ---
    if model_s2:
        try:
            input_s2 = preprocess_image(image, model_s2.input_shape)
            pred_s2 = model_s2.predict(input_s2, verbose=0)[0]
            result_s2 = {LABELS[i]: float(pred_s2[i]) for i in range(len(LABELS))}
        except Exception as e:
            result_s2 = {f"Error: {str(e)}": 0.0}
    else:
        result_s2 = {"Model 2 Tidak Ditemukan": 0.0}
        
    return result_s1, result_s2

# ==========================================
# 4. ANTARMUKA MODERN
# ==========================================

custom_css = """
.gradio-container { font-family: 'Inter', sans-serif !important; }
h1 { text-align: center; background: -webkit-linear-gradient(45deg, #6366f1, #a855f7); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 2.5rem; }
.subtitle { text-align: center; color: #6b7280; margin-bottom: 2rem; }
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #9ca3af;
}
"""

# Menggunakan theme standar Gradio 4.x yang stabil
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="zinc"), css=custom_css, title="Demo Sidang Skripsi") as demo:
    
    gr.Markdown("# ✨ Facial Expression Analysis ✨")
    gr.Markdown("<div class='subtitle'>VGG16 Transfer Learning + Squeeze-Excitation (SE-Block)</div>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="Upload Wajah", sources=["upload", "clipboard", "webcam"], height=350)
            submit_btn = gr.Button("🚀 Analyze Expression", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Skenario 1 (Baseline) ❄️")
                    output_s1 = gr.Label(num_top_classes=4, label="Confidence")
                with gr.Column():
                    gr.Markdown("### Skenario 2 (Optimized) 🔥")
                    output_s2 = gr.Label(num_top_classes=4, label="Confidence")
    
    submit_btn.click(fn=predict_expression, inputs=input_image, outputs=[output_s1, output_s2])

# --- Footer ---
    gr.Markdown("<div class='footer'>Developed by Qoid Rif'at | Universitas Trunojoyo Madura © 2025</div>")

# Launch dengan ssr_mode=False untuk stabilitas di Spaces
if __name__ == "__main__":
    demo.launch(ssr_mode=False)