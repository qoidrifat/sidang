import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. KONFIGURASI SISTEM & CUSTOM LAYER
# ==========================================

# Definisi SE-Block (Wajib ada untuk load model Skenario 2)
# Decorator ini penting agar TensorFlow mengenali fungsi ini saat serialisasi/deserialisasi
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

# Label Kelas (Urutan sesuai training generator)
LABELS = ['Marah 😡', 'Jijik 🤢', 'Takut 😱', 
          'Senang 😊', 'Sedih 😢', 'Terkejut 😲', 'Netral 😐']

# ==========================================
# 2. LOAD MODEL (CACHED)
# ==========================================
print("🔄 Sedang memuat model ke memori...")

# Load Model 1: Baseline
try:
    # Gunakan CPU untuk menghindari error CUDA di Space basic jika GPU tidak tersedia
    with tf.device('/CPU:0'):
        model_s1 = tf.keras.models.load_model("model_scenario1.keras", compile=False)
    print("✅ Model Skenario 1 (Baseline) Siap.")
except Exception as e:
    print(f"⚠️ Gagal load Model 1: {e}")
    model_s1 = None

# Load Model 2: Optimized (Proposed Method)
try:
    with tf.device('/CPU:0'):
        model_s2 = tf.keras.models.load_model(
            "best_model_scenario2.keras", 
            custom_objects=custom_objects_dict,
            compile=False
        )
    print("✅ Model Skenario 2 (Optimized) Siap.")
except Exception as e:
    print(f"⚠️ Gagal load Model 2: {e}")
    model_s2 = None

# ==========================================
# 3. LOGIKA PREDIKSI
# ==========================================
def predict_expression(image):
    if image is None:
        return None, None
    
    # Preprocessing standar VGG16
    image = image.resize((224, 224))
    img_array = np.array(image)
    
    # Konversi Grayscale ke RGB jika perlu
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediksi Skenario 1
    if model_s1:
        pred_s1 = model_s1.predict(img_array)[0]
        result_s1 = {LABELS[i]: float(pred_s1[i]) for i in range(len(LABELS))}
    else:
        result_s1 = {"Model Missing": 0.0}

    # Prediksi Skenario 2
    if model_s2:
        pred_s2 = model_s2.predict(img_array)[0]
        result_s2 = {LABELS[i]: float(pred_s2[i]) for i in range(len(LABELS))}
    else:
        result_s2 = {"Model Missing": 0.0}
        
    return result_s1, result_s2

# ==========================================
# 4. ANTARMUKA (MODERN STYLE)
# ==========================================

# Custom CSS untuk tampilan minimalis & modern
custom_css = """
.gradio-container {
    font-family: 'Inter', -apple-system, system-ui, sans-serif !important;
}
h1 {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #6366f1, #a855f7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800 !important;
    font-size: 2.5rem !important;
    margin-bottom: 0.5rem !important;
}
.subtitle {
    text-align: center;
    color: #6b7280;
    font-size: 1.1rem;
    margin-bottom: 2rem;
}
.result-header {
    font-weight: 600;
    color: #374151;
    margin-bottom: 0.5rem;
    text-align: center;
}
.footer {
    text-align: center;
    margin-top: 2rem;
    font-size: 0.8rem;
    color: #9ca3af;
}

/* Styling untuk Info Proyek */
.info-box {
    background-color: #f8fafc;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    height: 100%;
}
.info-label {
    font-weight: 600;
    color: #64748b;
    width: 100px;
    display: inline-block;
}
.info-value {
    color: #334155;
    font-weight: 500;
}

/* Styling Modern untuk Tabel Skenario */
.scenario-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 0.95rem;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
}
.scenario-table th {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
    color: white;
    padding: 14px 16px;
    text-align: left;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-size: 0.8rem;
}
.scenario-table td {
    padding: 14px 16px;
    border-bottom: 1px solid #f1f5f9;
    color: #475569;
    vertical-align: middle;
}

/* Kolom Fitur (Kiri) - Modern & Elegan */
.scenario-table td:first-child {
    background-color: #f8fafc;
    color: #312e81; /* Indigo gelap yang elegan */
    font-weight: 600;
    border-right: 2px solid #eef2ff;
    width: 25%;
}

.scenario-table tr:last-child td {
    border-bottom: none;
}
.scenario-table tr:hover td {
    background-color: #f8fafc;
}
.scenario-table tr:hover td:first-child {
    background-color: #e0e7ff; /* Highlight halus saat hover */
    color: #4f46e5;
}

/* Highlight Text yang Lebih Rapi */
.highlight-text {
    color: #4338ca;
    font-weight: 700;
    background-color: #e0e7ff;
    padding: 4px 10px;
    border-radius: 6px;
    font-size: 0.9rem;
    display: inline-block;
}
"""

# Membangun UI dengan Theme Soft (Indigo)
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="zinc"), css=custom_css, title="Demo Sidang Skripsi") as demo:
    
    # --- Header Section ---
    gr.Markdown("# ✨ Facial Expression Analysis ✨")
    gr.Markdown("<div class='subtitle'>VGG16 Transfer Learning + Squeeze-Excitation Attention Mechanism</div>")
    
    # --- INFO PROYEK (HTML Styled) ---
    with gr.Accordion("ℹ️ Informasi Peneliti & Skenario Pengujian", open=False):
        with gr.Row():
            # Kolom Kiri: Kartu Identitas
            with gr.Column(scale=2):
                gr.HTML("""
                <div class="info-box">
                    <h3 style="margin-top:0; color: #4f46e5; margin-bottom: 15px;">👥 Identitas Peneliti</h3>
                    <div style="margin-bottom: 8px;"><span class="info-label">Nama:</span> <span class="info-value">Qoid Rif'at</span></div>
                    <div style="margin-bottom: 8px;"><span class="info-label">NIM:</span> <span class="info-value">210411100160</span></div>
                    <div style="margin-bottom: 15px;"><span class="info-label">Instansi:</span> <span class="info-value">Universitas Trunojoyo Madura</span></div>
                    <hr style="border: 0; border-top: 1px dashed #cbd5e1; margin: 10px 0;">
                    <div style="margin-bottom: 8px;"><span class="info-label">Dospem 1:</span> <span class="info-value">Prof. Dr. Arif Muntasa, S.Si., M.T.</span></div>
                    <div><span class="info-label">Dospem 2:</span> <span class="info-value">Fifin Ayu Mufarroha, M.Kom.</span></div>
                </div>
                """)
            
            # Kolom Kanan: Tabel Perbandingan
            with gr.Column(scale=3):
                gr.HTML("""
                <div class="info-box" style="padding: 0; border: none; background: transparent;">
                    <table class="scenario-table">
                        <thead>
                            <tr>
                                <th>Fitur</th>
                                <th>Skenario 1 (Baseline)</th>
                                <th>Skenario 2 (Optimized)</th>
                            </tr>
                        </thead>
                        <tbody>
                            <tr>
                                <td>Arsitektur</td>
                                <td>VGG16 (Frozen Weights)</td>
                                <td>VGG16 (Fine-Tuned)</td>
                            </tr>
                            <tr>
                                <td>Integrasi</td>
                                <td>-</td>
                                <td><span class="highlight-text">SE-Block Attention</span></td>
                            </tr>
                            <tr>
                                <td>Training</td>
                                <td>Frozen Layers</td>
                                <td>Unfreeze Layer 11-19</td>
                            </tr>
                            <tr>
                                <td>Optimasi</td>
                                <td>Tanpa Augmentasi</td>
                                <td>Augmentasi + Label Smoothing</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
                """)

    # --- Main Content ---
    with gr.Row():
        
        # Kolom Kiri: Input
        with gr.Column(scale=1):
            gr.Markdown("### 📸 Input Image")
            input_image = gr.Image(
                type="pil", 
                label="Upload Wajah", 
                sources=["upload", "clipboard", "webcam"],
                height=350
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear", variant="secondary")
                submit_btn = gr.Button("🚀 Analyze Expression", variant="primary")
            
            gr.Markdown("""
            **Petunjuk:**
            1. Upload foto wajah yang jelas (frontal face).
            2. Klik **Analyze Expression**.
            3. Bandingkan hasil Baseline vs Optimized.
            """)

        # Kolom Kanan: Output Side-by-Side
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Comparative Results")
            
            with gr.Row():
                # Card Skenario 1
                with gr.Column():
                    gr.Markdown("<div class='result-header'>Skenario 1 (Baseline) ❄️</div>")
                    gr.Markdown("*VGG16 Frozen, No Attention*")
                    output_s1 = gr.Label(num_top_classes=4, label="Prediction Confidence")
                
                # Card Skenario 2 (Highlight)
                with gr.Column():
                    gr.Markdown("<div class='result-header' style='color: #4f46e5;'>Skenario 2 (Optimized) 🔥</div>")
                    gr.Markdown("*Fine-Tuned + SE-Block + Augmentasi*")
                    output_s2 = gr.Label(num_top_classes=4, label="Prediction Confidence")
    
    # --- Footer ---
    gr.Markdown("<div class='footer'>Developed by Qoid Rif'at | Universitas Trunojoyo Madura © 2025</div>")

    # --- Event Handlers ---
    submit_btn.click(
        fn=predict_expression, 
        inputs=input_image, 
        outputs=[output_s1, output_s2]
    )
    clear_btn.click(
        lambda: (None, None, None), 
        outputs=[input_image, output_s1, output_s2]
    )

# Jalankan Aplikasi
if __name__ == "__main__":
    # ssr_mode=False menonaktifkan Server-Side Rendering yang sering konflik dengan TensorFlow di Spaces
    demo.launch(ssr_mode=False)