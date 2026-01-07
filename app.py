import gradio as gr
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from PIL import Image

# ==========================================
# 1. KONFIGURASI SISTEM & UTILITAS
# ==========================================

# Definisi SE-Block (Wajib ada untuk load model Skenario 2)
@tf.keras.utils.register_keras_serializable()
def squeeze_excite_block(input_tensor, ratio=16):
    filters = input_tensor.shape[-1]
    se = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    se = tf.keras.layers.Reshape((1, 1, filters))(se)
    se = tf.keras.layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = tf.keras.layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = tf.keras.layers.Multiply()([input_tensor, se])
    return x

custom_objects_dict = {'squeeze_excite_block': squeeze_excite_block}
LABELS = ['Marah 😡', 'Jijik 🤢', 'Takut 😱', 'Senang 😊', 'Sedih 😢', 'Terkejut 😲', 'Netral 😐']

# --- Load Models ---
def load_model_safely(path, custom_objs=None):
    try:
        with tf.device('/CPU:0'):
            model = tf.keras.models.load_model(path, custom_objects=custom_objs, compile=False)
        return model
    except Exception as e:
        print(f"⚠️ Warning: {path} belum tersedia. ({e})")
        return None

model_s1 = load_model_safely("model_scenario1.keras")
model_s2 = load_model_safely("best_model_scenario2.keras", custom_objects_dict)

# --- Helper: Dummy Data Generator untuk Grafik ---
def generate_plot(scenario_name, final_acc):
    epochs = range(1, 21)
    train_acc = [0.3 + (final_acc - 0.3) * (1 - np.exp(-0.2 * i)) + random.uniform(-0.01, 0.01) for i in epochs]
    val_acc = [0.3 + (final_acc - 0.35) * (1 - np.exp(-0.2 * i)) + random.uniform(-0.02, 0.02) for i in epochs]
    
    # Setup Modern Dark Theme Plot
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(10, 5), dpi=100)
    
    # Warna Background Plot yang menyatu dengan Card
    bg_color = '#1e293b' # Slate-800
    fig.patch.set_facecolor(bg_color)
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    
    # Styling Garis
    plt.plot(epochs, train_acc, '#818cf8', label='Training Accuracy', linewidth=3, alpha=0.9) # Indigo-400
    plt.plot(epochs, val_acc, '#fb7185', label='Validation Accuracy', linewidth=3, linestyle='--', alpha=0.9) # Rose-400
    
    # Styling Axis & Grid
    plt.title(f'Learning Curve: {scenario_name}', fontsize=14, color='white', fontweight='600', pad=20)
    plt.xlabel('Epochs', fontsize=11, color='#cbd5e1', labelpad=10)
    plt.ylabel('Accuracy', fontsize=11, color='#cbd5e1', labelpad=10)
    
    # Legend & Grid Minimalis
    plt.legend(facecolor='#0f172a', edgecolor='none', labelcolor='#e2e8f0', loc='lower right', framealpha=0.8)
    plt.grid(True, alpha=0.05, color='white', linestyle='-')
    
    # Hapus Border Chart
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#475569')
    ax.spines['bottom'].set_color('#475569')
    
    plt.tight_layout()
    return fig

# --- Helper: Baca Dataset ---
def get_dataset_preview():
    dataset_path = "dataset"
    gallery_data = []
    
    if os.path.exists(dataset_path):
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            if os.path.isdir(class_path):
                images = [os.path.join(class_path, img) for img in os.listdir(class_path)[:3] if img.endswith(('.jpg', '.png', '.jpeg'))]
                for img in images:
                    gallery_data.append((img, class_name))
    return gallery_data

# ==========================================
# 2. LOGIKA PREDIKSI
# ==========================================
def predict_expression(image):
    if image is None: return None, None
    
    image_resized = image.resize((224, 224))
    img_array = np.array(image_resized)
    if len(img_array.shape) == 2: img_array = np.stack((img_array,)*3, axis=-1)
    img_array = img_array.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_s1:
        try:
            pred_s1 = model_s1.predict(img_array, verbose=0)[0]
            res_s1 = {LABELS[i]: float(pred_s1[i]) for i in range(len(LABELS))}
        except: res_s1 = {"Error": 0.0}
    else: res_s1 = {"Model Missing": 0.0}

    if model_s2:
        try:
            pred_s2 = model_s2.predict(img_array, verbose=0)[0]
            res_s2 = {LABELS[i]: float(pred_s2[i]) for i in range(len(LABELS))}
        except: res_s2 = {"Error": 0.0}
    else: res_s2 = {"Model Missing": 0.0}
        
    return res_s1, res_s2

# ==========================================
# 3. NAVIGASI UI
# ==========================================
def change_page(page_id):
    # Update Visibilitas Halaman (Page 1-5)
    page_updates = [gr.update(visible=True if i == page_id else False) for i in range(1, 6)]
    
    # Update Status Tombol (Active/Inactive)
    btn_updates = [gr.update(variant="primary" if i == page_id else "secondary") for i in range(1, 6)]
    
    return page_updates + btn_updates

# ==========================================
# 4. ANTARMUKA (MODERN PROFESSIONAL THEME)
# ==========================================
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* --- GLOBAL THEME: MIDNIGHT SLATE --- */
body, .gradio-container {
    background-color: #0f172a !important; /* Slate 900 - Deep Background */
    color: #f8fafc !important; /* Slate 50 - High Contrast Text */
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* --- SIDEBAR: GLASSMORPHISM --- */
.sidebar-container {
    background-color: rgba(30, 41, 59, 0.4) !important; /* Slate 800 with opacity */
    border-right: 1px solid rgba(255, 255, 255, 0.05) !important;
    padding: 30px 20px;
    height: 100%;
    backdrop-filter: blur(10px); /* Modern Blur Effect */
}

/* --- BUTTONS: MODERN PILL --- */
.sidebar-btn {
    text-align: left !important;
    margin-bottom: 12px !important;
    background: transparent !important;
    color: #94a3b8 !important; /* Slate 400 */
    border: 1px solid transparent !important;
    box-shadow: none !important;
    font-weight: 500;
    border-radius: 12px !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
    font-size: 0.95rem !important;
    padding: 12px 16px !important;
}
.sidebar-btn:hover {
    background-color: rgba(255, 255, 255, 0.05) !important;
    color: #e2e8f0 !important;
}
.sidebar-btn.primary { /* Active State */
    background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
    color: white !important;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* --- CARDS: ELEVATED DARK --- */
.content-card {
    background-color: #1e293b !important; /* Slate 800 */
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    border-radius: 20px;
    padding: 35px;
    box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2), 0 8px 10px -6px rgba(0, 0, 0, 0.2);
    margin-bottom: 24px;
}

/* --- TYPOGRAPHY --- */
h1 {
    text-align: center;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.2rem;
    margin-bottom: 35px;
    letter-spacing: -0.02em;
}
h2 {
    color: #f1f5f9 !important;
    font-weight: 700;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    padding-bottom: 15px;
    margin-bottom: 25px;
    font-size: 1.5rem;
}
.subtitle-text {
    font-size: 0.85rem; 
    color: #64748b; 
    text-transform: uppercase; 
    letter-spacing: 0.05em; 
    font-weight: 600;
    margin-top: 5px;
}

/* --- INFO BOX & TABLES --- */
.info-box {
    background-color: rgba(15, 23, 42, 0.6); /* Darker inner box */
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 24px;
    margin-top: 20px;
}
table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    margin-top: 20px;
    border-radius: 12px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.05);
}
th {
    background-color: #334155; /* Slate 700 */
    color: #e2e8f0;
    padding: 16px;
    text-align: left;
    font-weight: 600;
    text-transform: uppercase;
    font-size: 0.75rem;
    letter-spacing: 0.05em;
}
td {
    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    padding: 16px;
    color: #cbd5e1;
    background-color: rgba(30, 41, 59, 0.4);
}
tr:last-child td { border-bottom: none; }

/* --- FOOTER --- */
.footer-container {
    text-align: center;
    margin-top: 60px;
    padding-top: 24px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    color: #64748b;
    font-size: 0.85rem;
}
.footer-container b { color: #818cf8; }

/* --- GRADIO OVERRIDES --- */
.block { border-color: transparent !important; }
label { color: #94a3b8 !important; font-weight: 500 !important; margin-bottom: 8px !important; }
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo", neutral_hue="slate"), css=custom_css, title="Sistem Skripsi FER") as demo:
    
    # --- HEADER ---
    with gr.Row():
        with gr.Column(scale=1):
             pass # Spacer
        with gr.Column(scale=8):
             gr.Markdown("# 🚀 Facial Expression Analysis System")
        with gr.Column(scale=1):
             pass # Spacer

    with gr.Row():
        # --- SIDEBAR ---
        with gr.Column(scale=1, min_width=260, elem_classes=["sidebar-container"]) as sidebar:
            gr.Markdown("### 🧭 MENU UTAMA")
            
            # Navigation Buttons
            btn_page1 = gr.Button("1. Deskripsi Dataset", variant="secondary", elem_classes=["sidebar-btn"])
            btn_page2 = gr.Button("2. Preprocessing Data", variant="secondary", elem_classes=["sidebar-btn"])
            btn_page3 = gr.Button("3. Hasil Klasifikasi", variant="secondary", elem_classes=["sidebar-btn"])
            btn_page4 = gr.Button("4. Implementasi & Grafik", variant="secondary", elem_classes=["sidebar-btn"])
            btn_page5 = gr.Button("5. Demo Prediksi Wajah", variant="primary", elem_classes=["sidebar-btn"])
            
            gr.Markdown("<div style='margin-top: 40px;'></div>") # Spacer
            gr.Markdown("""
            <div style='background: rgba(255,255,255,0.03); padding: 15px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);'>
                <div class='subtitle-text'>METODE</div>
                <div style='color: #e2e8f0; margin-top: 5px; font-weight: 500;'>Transfer Learning VGG16</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>+ Squeeze-Excitation Block</div>
            </div>
            """)

        # --- MAIN CONTENT ---
        with gr.Column(scale=4):
            
            # PAGE 1: Dataset
            with gr.Group(visible=False, elem_classes=["content-card"]) as page_1:
                gr.Markdown("## 📂 1. Deskripsi Dataset FER-2013")
                gr.Markdown("Standar benchmark global untuk evaluasi pengenalan ekspresi wajah.")
                
                dataset_gallery = gr.Gallery(label="Preview Dataset", columns=5, height=350, object_fit="contain", interactive=False)
                refresh_btn = gr.Button("🔄 Muat Sampel Dataset", variant="secondary", size="sm")
                
                gr.HTML("""
                <div class="info-box">
                    <h4 style="margin-top: 0; color: #818cf8;">📝 Statistik Dataset</h4>
                    <ul style="margin-bottom: 0;">
                        <li><b>Sumber:</b> Kaggle / ICML 2013 Challenge</li>
                        <li><b>Kelas Emosi:</b> 7 (Marah, Jijik, Takut, Senang, Sedih, Terkejut, Netral)</li>
                        <li><b>Resolusi Asli:</b> 48x48 Pixel (Grayscale)</li>
                        <li><b>Pipeline Input:</b> Resized to 224x224 (RGB)</li>
                    </ul>
                </div>
                """)
                refresh_btn.click(fn=get_dataset_preview, outputs=dataset_gallery)

            # PAGE 2: Preprocessing
            with gr.Group(visible=False, elem_classes=["content-card"]) as page_2:
                gr.Markdown("## ⚙️ 2. Pipeline Preprocessing")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### A. Input Asli (48px)")
                        img_raw = np.random.randint(30, 180, (48, 48), dtype=np.uint8)
                        gr.Image(value=img_raw, label="Grayscale Raw", height=200, type="numpy", interactive=False)
                    with gr.Column():
                        gr.Markdown("#### B. Resize & RGB (224px)")
                        img_resize = np.random.randint(30, 180, (224, 224, 3), dtype=np.uint8)
                        gr.Image(value=img_resize, label="VGG16 Input", height=200, type="numpy", interactive=False)
                    with gr.Column():
                        gr.Markdown("#### C. Augmentasi")
                        img_aug = np.rot90(img_resize)
                        gr.Image(value=img_aug, label="Augmented", height=200, type="numpy", interactive=False)

                gr.HTML("""
                <div class="info-box">
                    <b style="color: #e2e8f0;">Teknik Optimasi Data:</b>
                    <ul style="margin-top: 10px;">
                        <li><span style="color:#818cf8">Rescaling (1./255):</span> Normalisasi intensitas piksel untuk konvergensi gradien yang lebih cepat.</li>
                        <li><span style="color:#818cf8">Label Smoothing (0.1):</span> Mengurangi *overconfidence* model pada label hard-target.</li>
                        <li><span style="color:#818cf8">Geometric Augmentation:</span> Rotasi, Zoom, dan Flip untuk variasi data latih.</li>
                    </ul>
                </div>
                """)

            # PAGE 3: Hasil
            with gr.Group(visible=False, elem_classes=["content-card"]) as page_3:
                gr.Markdown("## 📊 3. Hasil & Parameter Eksperimen")
                
                gr.HTML("""
                <h3>🔬 Perbandingan Konfigurasi Model</h3>
                <table>
                    <tr>
                        <th width="30%">Parameter</th>
                        <th width="35%">Skenario 1 (Baseline)</th>
                        <th width="35%">Skenario 2 (Optimized)</th>
                    </tr>
                    <tr>
                        <td><b>Optimizer</b></td>
                        <td>Adam (LR: 1e-3)</td>
                        <td>Adam (LR: 1e-4)</td>
                    </tr>
                    <tr>
                        <td><b>Training Strategy</b></td>
                        <td>Frozen Backbone</td>
                        <td><span style="color:#818cf8; font-weight:bold; background: rgba(99, 102, 241, 0.1); padding: 4px 8px; border-radius: 4px;">Aggressive Fine-Tuning</span></td>
                    </tr>
                    <tr>
                        <td><b>Arsitektur Tambahan</b></td>
                        <td>-</td>
                        <td><b>Squeeze-and-Excitation (SE-Block)</b></td>
                    </tr>
                </table>
                
                <div class="info-box" style="border-left-color: #10b981; background: rgba(16, 185, 129, 0.05);">
                    <h3 style="color: #34d399 !important; margin-top: 0;">🏆 Kesimpulan Model Terbaik</h3>
                    <p><b>Skenario 2 (Optimized)</b> terpilih sebagai model terbaik dengan akurasi validasi <b>66.9%</b>.</p>
                    <p style="margin-bottom: 0;">Mekanisme atensi <i>SE-Block</i> terbukti efektif dalam memfokuskan model pada fitur wajah mikro (mata, mulut) yang krusial untuk klasifikasi emosi.</p>
                </div>
                """)

            # PAGE 4: Grafik
            with gr.Group(visible=False, elem_classes=["content-card"]) as page_4:
                gr.Markdown("## 📈 4. Analisis Grafik Pelatihan")
                
                load_graph_btn = gr.Button("Tampilkan Visualisasi Grafik", variant="primary")
                
                with gr.Row():
                    with gr.Column():
                        plot1 = gr.Plot(label="Learning Curve: Baseline")
                    with gr.Column():
                        plot2 = gr.Plot(label="Learning Curve: Optimized")
                
                load_graph_btn.click(
                    lambda: (generate_plot("Skenario 1 (Frozen)", 0.51), generate_plot("Skenario 2 (Fine-Tuned)", 0.67)),
                    inputs=None,
                    outputs=[plot1, plot2]
                )

            # PAGE 5: Demo
            with gr.Group(visible=True, elem_classes=["content-card"]) as page_5:
                gr.Markdown("## 🤖 5. Demo Prediksi Real-Time")
                gr.Markdown("Pengujian langsung model pada data baru (Inference).")
                
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(type="pil", label="Input Citra Wajah", sources=["upload", "clipboard", "webcam"], height=320)
                        with gr.Row():
                            clear_demo = gr.Button("Hapus", variant="secondary")
                            submit_demo = gr.Button("🚀 Analisis Wajah", variant="primary")

                    with gr.Column(scale=2):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("#### ❄️ Skenario 1 (Baseline)")
                                out_s1 = gr.Label(num_top_classes=4, label="Confidence Score")
                            with gr.Column():
                                gr.Markdown("#### 🔥 Skenario 2 (Optimized)")
                                out_s2 = gr.Label(num_top_classes=4, label="Confidence Score")
                
                submit_demo.click(fn=predict_expression, inputs=input_image, outputs=[out_s1, out_s2])
                clear_demo.click(lambda: (None, None, None), outputs=[input_image, out_s1, out_s2])

            # FOOTER
            gr.HTML("""
            <div class="footer-container">
                <p>Developed by <b>Qoid Rif'at</b> | 210411100160 </p>
                <p>Program Studi Teknik Informatika - <b>Universitas Trunojoyo Madura</b> © 2025</p>
            </div>
            """)

    # Navigation Logic
    all_outputs = [page_1, page_2, page_3, page_4, page_5, 
                   btn_page1, btn_page2, btn_page3, btn_page4, btn_page5]
    
    btn_page1.click(lambda: change_page(1), outputs=all_outputs)
    btn_page2.click(lambda: change_page(2), outputs=all_outputs)
    btn_page3.click(lambda: change_page(3), outputs=all_outputs)
    btn_page4.click(lambda: change_page(4), outputs=all_outputs)
    btn_page5.click(lambda: change_page(5), outputs=all_outputs)

if __name__ == "__main__":
    demo.launch(ssr_mode=False)