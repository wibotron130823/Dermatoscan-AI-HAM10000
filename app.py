import streamlit as st
import tensorflow as tf
import numpy as np
import google.generativeai as genai
from PIL import Image
from streamlit_cropper import st_cropper
import pandas as pd
import os
from dotenv import load_dotenv
# import google.generativeai as genai
from google import genai 
from google.genai import types
import tensorflow as tf
from tensorflow.keras import backend as K

# ============================
#   KONFIGURASI SISTEM & API
# ============================
st.set_page_config(
    page_title="DermatoScan AI - Telkom University",
    layout="wide",
    page_icon="🩺"
)

# GEMINI INTEGRATION
load_dotenv()

# Inisialisasi variabel kosong
GEN_AI_KEY = None

# Coba ambil dari st.secrets dulu (untuk Cloud)
try:
    if "GEMINI_API_KEY" in st.secrets:
        GEN_AI_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    # Jika gagal (berarti di lokal tanpa secrets.toml), ambil dari .env
    GEN_AI_KEY = os.getenv("GEMINI_API_KEY")

if not GEN_AI_KEY:
    st.error("API Key tidak ditemukan! Pastikan .env terisi di lokal atau Secrets terisi di Cloud.")
else:
    # INISIALISASI Client
    client = genai.Client(api_key=GEN_AI_KEY)

# ================================
#   MODEL & CUSTOM LOSS HANDLING
# ================================

def macro_f1(y_true, y_pred):
    """Metrik Macro F1-score untuk evaluasi model."""
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2 * p * r / (p + r + K.epsilon())
    f1 = tf.where(tf.math.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

def weighted_focal_loss(class_weights, gamma=2.0):
    weights = tf.constant(
        [class_weights[i] for i in sorted(class_weights.keys())],
        dtype=tf.float32
    )
    def loss(y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1. - eps)
        ce = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        pt = tf.reduce_sum(y_true * y_pred, axis=-1)
        focal = tf.pow(1. - pt, gamma)
        weight = tf.reduce_sum(weights * y_true, axis=-1)
        return tf.reduce_mean(weight * focal * ce)
    return loss

@st.cache_resource
def load_my_model():
    dummy_weights = {i: 1.0 for i in range(7)} 
    model_path = 'model/HAM10000_model_01.keras'
    
    return tf.keras.models.load_model(
        model_path, 
        custom_objects={'loss': weighted_focal_loss(dummy_weights, gamma=2.5), 'macro_f1': macro_f1}
    )

# Inisialisasi Model
try:
    model = load_my_model()
except Exception as e:
    st.error(f"Gagal memuat model. Pastikan path benar. Error: {e}")

# Mapping Label & Informasi Edukasi
class_names = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
lesion_type_dict = {
    'akiec': 'Actinic Keratoses (Pra-Kanker)',
    'bcc': 'Basal Cell Carcinoma (Kanker)',
    'bkl': 'Benign Keratosis (Jinak)',
    'df': 'Dermatofibroma (Jinak)',
    'mel': 'Melanoma (Sangat Ganas)',
    'nv': 'Melanocytic Nevi (Tahi Lalat Biasa)',
    'vasc': 'Vascular Lesions (Pembuluh Darah)'
}

# =======================
#   CORE LOGIC FUNCTIONS
# =======================

def get_gemini_consultation(prediction, confidence, diagnosis_banding):
    """Menghasilkan edukasi medis berbasis AI menggunakan Gemini."""
    prompt = f"""
    Anda adalah seorang konsultan medis dermatologi AI profesional. 
    Sistem CNN memprediksi lesi kulit ini sebagai: {prediction} 
    dengan tingkat kepercayaan: {confidence:.2f}%.
    Diagnosa banding lainnya adalah: {diagnosis_banding}.
    
    Berikan respon terstruktur:
    1. Edukasi singkat mengenai apa itu {prediction}.
    2. Potensi bahaya (Ganas/Jinak) dan urgensi medisnya.
    3. Langkah praktis yang disarankan (misal: ABCDE rule, proteksi UV, atau segera ke dokter).
    4. DISCLAIMER WAJIB: Hasil ini hanya bantuan AI dan bukan diagnosa final dokter.
    
    Gunakan bahasa Indonesia yang empatik, tenang, namun sangat profesional.
    """
    # PEMANGGILAN BARU
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )
    return response.text

# ===============================
#    USER INTERFACE (STREAMLIT)
# ===============================
st.sidebar.title("🩺 DermatoScan AI")
menu = st.sidebar.radio("Navigasi", ["Beranda", "Analisis Lesi", "Edukasi Kanker", "Tentang"])
if menu == "Beranda":
    st.title("🛡️ Deteksi Dini Lesi Kulit Berbasis AI")
    st.write("""
    Selamat datang di **DermatoScan AI**. Aplikasi ini menggunakan arsitektur *Deep Convolutional Neural Network* yang dilatih pada dataset HAM10000 untuk membantu mengidentifikasi jenis lesi kulit secara cepat.
    """)
    st.image("https://img.freepik.com/free-vector/dermatology-concept-illustration_114360-8025.jpg", width=500)
    st.info("Gunakan menu 'Analisis Lesi' untuk memulai pemeriksaan.")

elif menu == "Analisis Lesi":
    st.title("🔍 Analisis Gambar Lesi")
    
    col_input, col_output = st.columns([1, 1])
    
    with col_input:
        source = st.radio("Pilih Sumber Gambar:", ("Upload File", "Kamera"))
        img_file = st.file_uploader("Pilih file gambar", type=["jpg", "png", "jpeg"]) if source == "Upload File" else st.camera_input("Ambil foto")

        if img_file:
            img = Image.open(img_file).convert("RGB")
            st.write("---")
            st.write("**Langkah 1: Crop Gambar** (Fokuskan pada area lesi)")
            cropped_img = st_cropper(img, aspect_ratio=(1, 1), box_color="blue")
            st.session_state['ready_to_analyze'] = True

    if img_file and st.session_state.get('ready_to_analyze'):
        with col_output:
            if st.button("🚀 Mulai Analisis Diagnostik"):
                # Preprocessing
                img_resized = cropped_img.resize((128, 128))
                img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
                img_array = np.expand_dims(img_array / 255.0, axis=0)
                
                # Inference
                preds = model.predict(img_array)[0]
                
                # Top-3 Calculation
                top_3_indices = preds.argsort()[-3:][::-1]
                top_label = lesion_type_dict[class_names[top_3_indices[0]]]
                top_conf = preds[top_3_indices[0]] * 100
                
                # Display Metrics
                st.subheader("Hasil Analisis")
                st.metric(label="Prediksi Utama", value=top_label)
                st.progress(int(top_conf))
                st.write(f"**Tingkat Kepercayaan:** {top_conf:.2f}%")
                
                # Visualisasi Probabilitas
                st.write("---")
                st.write("📊 **Probabilitas Top 3:**")
                chart_data = pd.DataFrame({
                    'Lesi': [lesion_type_dict[class_names[i]] for i in top_3_indices],
                    'Probabilitas (%)': [preds[i]*100 for i in top_3_indices]
                })
                st.bar_chart(chart_data.set_index('Lesi'))

                # Gemini Consultation Context
                st.divider()
                st.subheader("🤖 Konsultasi Edukatif AI (Gemini)")
                diagnosis_banding = f"{lesion_type_dict[class_names[top_3_indices[1]]]} dan {lesion_type_dict[class_names[top_3_indices[2]]]}"
                
                with st.spinner("Menganalisis hasil dan menyusun edukasi..."):
                    try:
                        consultation = get_gemini_consultation(top_label, top_conf, diagnosis_banding)
                        st.markdown(consultation)
                    except Exception as e:
                        st.error("Gagal terhubung dengan Gemini API. Pastikan internet Anda aktif.")

elif menu == "Edukasi Kanker":
    st.title("📚 Library Informasi Lesi Kulit")
    st.write("Pelajari lebih lanjut tentang jenis-jenis lesi kulit yang dapat dideteksi oleh sistem.")
    # dummy
    disease_details = {
        'akiec': {
            'deskripsi': "Bercak kasar dan bersisik pada kulit yang berkembang setelah bertahun-tahun terpapar sinar matahari. Ini adalah kondisi pra-kanker.",
            'gejala': "Permukaan kasar, kering, atau bersisik, biasanya berdiameter kurang dari 2,5 cm.",
            'saran': "Gunakan tabir surya secara rutin dan konsultasikan ke dokter untuk tindakan laser atau cryotherapy."
        },
        'bcc': {
            'deskripsi': "Tipe kanker kulit yang paling umum. Jarang menyebar ke bagian tubuh lain, tetapi bisa merusak jaringan di sekitarnya jika didiamkan.",
            'gejala': "Benjolan putih mutiara atau merah muda yang mengkilap, seringkali dengan pembuluh darah yang terlihat.",
            'saran': "Memerlukan tindakan bedah kecil oleh dokter spesialis kulit."
        },
        'bkl': {
            'deskripsi': "Pertumbuhan kulit non-kanker yang umum terjadi pada orang tua. Sering dianggap sebagai 'kutil' penuaan.",
            'gejala': "Biasanya berwarna cokelat, hitam, atau cokelat muda. Terlihat seperti 'menempel' di kulit.",
            'saran': "Tidak berbahaya secara medis, namun bisa diangkat jika terjadi iritasi."
        },
        'df': {
            'deskripsi': "Pertumbuhan jaringan ikat yang umum dan jinak. Sering muncul setelah cedera kecil seperti gigitan serangga.",
            'gejala': "Benjolan keras yang bisa berwarna cokelat hingga merah gelap. Jika dicubit, biasanya akan membentuk cekungan.",
            'saran': "Biasanya tidak memerlukan pengobatan kecuali jika mengganggu secara estetika."
        },
        'mel': {
            'deskripsi': "Jenis kanker kulit paling berbahaya. Memiliki potensi tinggi untuk menyebar (metastasis) ke organ lain.",
            'gejala': "Gunakan aturan ABCDE (Asymmetry, Border, Color, Diameter, Evolving) untuk deteksi dini.",
            'saran': "URGEN. Segera temui dokter spesialis kulit jika Anda mencurigai adanya Melanoma."
        },
        'nv': {
            'deskripsi': "Tahi lalat biasa. Merupakan pertumbuhan jinak dari sel melanosit.",
            'gejala': "Bentuk simetris, pinggiran rata, dan warna yang seragam.",
            'saran': "Lakukan pemeriksaan mandiri secara berkala untuk memastikan tidak ada perubahan bentuk atau warna."
        },
        'vasc': {
            'deskripsi': "Lesi yang berasal dari pembuluh darah, seperti angioma atau pyogenic granuloma.",
            'gejala': "Berwarna merah cerah, ungu, atau biru tua. Terkadang bisa berdarah jika terkena trauma.",
            'saran': "Konsultasikan ke dokter jika lesi sering berdarah atau tumbuh dengan sangat cepat."
        }
    }
    # display info
    for key, name in lesion_type_dict.items():
        with st.expander(f"📌 {name}"):
            info = disease_details.get(key, {})
            # Layout Kolom untuk Informasi
            col_text, col_img = st.columns([2, 1])
            with col_text:
                st.markdown(f"**Deskripsi:**\n{info.get('deskripsi')}")
                st.markdown(f"**Gejala Visual:**\n{info.get('gejala')}")
                st.info(f"💡 **Saran Medis:**\n{info.get('saran')}")
    st.markdown("---")
    st.warning("⚠️ **Penting:** Informasi di atas hanya untuk tujuan edukasi. Jangan gunakan sebagai satu-satunya dasar diagnosa medis tanpa pengawasan profesional.")

elif menu == "Tentang":
    st.title("📌 Detail Proyek")
    st.info("Aplikasi ini dikembangkan untuk Tugas Besar SG AI Laboratory - Telkom University.")
    st.markdown("""
    **Spesifikasi Teknis:**
    - **Arsitektur:** Custom CNN (Sequential)
    - **Optimization:** Weighted Focal Loss (Gamma 2.5)
    - **Input Resolution:** 128x128 pixels
    - **Dataset:** HAM10000 (Human Against Machine)
    
    **Developers:**
    1. Achmad Baihaqie Wibowo
    2. Bayu Alif Aryo Wiputra
    """)

# =================
#   FINALIZATION
# =================
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 DermatoScan AI Project. For Educational Purposes Only.")