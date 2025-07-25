import streamlit as st
import os
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import glob
import numpy as np
import cv2
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from tensorflow.keras.models import load_model
import pytesseract
import tempfile

# Untuk fitur cropping (opsional)
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False

# ===============================
# KONFIGURASI APLIKASI
# ===============================
st.set_page_config(
    page_title="Sistem Aksara Bima",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# KONFIGURASI PATH & VARIABEL
# ===============================
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_aksara_bima')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'aksara_bima_m.h5')
LOGO_PATH = os.path.join(BASE_DIR, 'logo', 'logo.png')
TESSDATA_PATH = os.path.join(BASE_DIR, 'tessdata')
TESSERACT_CMD = 'tesseract'

try:
    os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
except Exception:
    pass

# Konstanta karakter (sudah dirapikan)
BIMA_CHARACTERS = sorted(list(set([
    'A', 'BA', 'BE', 'BI', 'BO', 'BU', 'CA', 'CE', 'CI', 'CO', 'CU', 'DA', 'DE',
    'DI', 'DO', 'DU', 'E', 'FA', 'FE', 'FI', 'FO', 'FU', 'GA', 'GE', 'GI', 'GO',
    'GU', 'HA', 'HE', 'HI', 'HO', 'HU', 'I', 'JA', 'JE', 'JI', 'JO', 'JU', 'KA',
    'KE', 'KI', 'KO', 'KU', 'LA', 'LE', 'LI', 'LO', 'LU', 'MA', 'ME', 'MI', 'MO',
    'MPA', 'MPE', 'MPI', 'MPO', 'MPU', 'MU', 'NA', 'NCA', 'NCE', 'NCI', 'NCO', 'NCU',
    'NE', 'NGA', 'NGGE', 'NGGI', 'NGGO', 'NGGU', 'NGE', 'NGI', 'NGO', 'NGU', 'NI',
    'NO', 'NTA', 'NTE', 'NTI', 'NTO', 'NTU', 'NU', 'O', 'PA', 'PE', 'PI', 'PO', 'PU',
    'RA', 'RE', 'RI', 'RO', 'RU', 'SA', 'SE', 'SI', 'SO', 'SU', 'TA', 'TE', 'TI',
    'TO', 'TU', 'U', 'WA', 'WE', 'WI', 'WO', 'WU', 'YA', 'YE', 'YI', 'YO', 'YU',
    'MBA', 'MBE', 'MBI', 'MBO', 'MBU', 'NDA', 'NDE', 'NDI', 'NDO', 'NDU',
    'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Y', 'NG'
])), key=len, reverse=True)

CLASSIFICATION_CHARACTERS = [
    'A', 'BA', 'CA', 'DA', 'FA', 'GA', 'HA', 'JA',
    'KA', 'LA', 'MA', 'MPA', 'NA', 'NCA', 'NGA',
    'NTA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA'
]

if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# ===============================
# FUNGSI-FUNGSI UTAMA (LOGIKA APLIKASI)
# ===============================

@st.cache_data
def load_character_images():
    """Memuat gambar karakter dari dataset."""
    char_images = {}
    if not os.path.exists(DATASET_PATH): return {}
    for folder_name in sorted(os.listdir(DATASET_PATH)):
        folder_path = os.path.join(DATASET_PATH, folder_name)
        if os.path.isdir(folder_path):
            try:
                img_path = glob.glob(os.path.join(folder_path, '*'))[0]
                char_images[folder_name] = Image.open(img_path)
            except (IndexError, OSError):
                pass
    return char_images

@st.cache_resource
def load_classification_model():
    """Memuat model klasifikasi tanpa menampilkan pesan di UI."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return load_model(MODEL_PATH)
    except Exception:
        return None

def transliterate_to_bima(text):
    """Mengubah teks Latin menjadi daftar karakter Aksara Bima."""
    text = text.upper().strip()
    result, i = [], 0
    while i < len(text):
        if text[i].isspace():
            if result and result[-1] != ' ': result.append(' ')
            i += 1
            continue
        matched = False
        for char in BIMA_CHARACTERS:
            if text[i:].startswith(char):
                result.append(char)
                i += len(char)
                matched = True
                break
        if not matched:
            result.append(f"[{text[i]}]")
            i += 1
    return result

def create_full_text_image(result, char_images, char_spacing=5, word_spacing=20):
    """Membuat satu gambar dari hasil transliterasi (versi detail)."""
    all_images = []
    for char in result:
        if char == ' ':
            if all_images:
                space_img = Image.new('RGB', (word_spacing, 80), 'white')
                all_images.append(space_img)
        elif not char.startswith('['):
            img = char_images.get(char)
            if img:
                all_images.append(img.resize((80, 80), Image.Resampling.LANCZOS))
    if not all_images: return None
    
    total_width = sum(img.width for img in all_images)
    if len(all_images) > 1:
        char_spacings = 0
        for i, img in enumerate(all_images):
            if i > 0 and all_images[i-1].width != word_spacing:
                char_spacings += char_spacing
        total_width += char_spacings

    combined = Image.new('RGB', (total_width, 80), 'white')
    x_offset = 0
    for i, img in enumerate(all_images):
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
        if i < len(all_images) - 1 and all_images[i].width != word_spacing:
            x_offset += char_spacing
            
    return combined

def preprocess_image(image_input):
    """Mempersiapkan gambar untuk input model klasifikasi."""
    try:
        if isinstance(image_input, Image.Image):
            img_array = np.array(image_input.convert('RGB'))
        else: # Dari Canvas (numpy array)
            if image_input.shape[-1] == 4: # RGBA -> RGB
                alpha = image_input[:, :, 3:4] / 255.0
                bg = np.ones_like(image_input[:, :, :3]) * 255
                img_array = (alpha * image_input[:, :, :3] + (1 - alpha) * bg).astype(np.uint8)
            else:
                img_array = image_input
        resized = cv2.resize(img_array, (224, 224))
        return np.expand_dims(resized.astype('float32') / 255.0, axis=0)
    except Exception:
        return None

def ocr_image(image):
    """Menjalankan OCR pada gambar."""
    try:
        config = f'--tessdata-dir "{TESSDATA_PATH}" -l aksaralengkap --psm 8'
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception:
        return ""

def image_to_base64(img):
    """Konversi gambar PIL ke base64 untuk diunduh."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# ===============================
# HALAMAN-HALAMAN APLIKASI (UI)
# ===============================

def home_page():
    """Halaman Beranda."""
    col1, col2 = st.columns([1, 6], gap="medium")
    if os.path.exists(LOGO_PATH):
        with col1:
            st.image(LOGO_PATH, width=120)
    with col2:
        st.title("Aplikasi Pengenalan Aksara Bima")
        st.markdown("Sebuah alat bantu digital untuk melestarikan, mempelajari, dan menggunakan Aksara Bima.")
    st.markdown("---")

    with st.expander("📖 **Panduan Penggunaan Aplikasi (Klik untuk Buka)**", expanded=True):
        st.markdown("""
        Selamat datang! Aplikasi ini memiliki tiga fitur utama. Ikuti langkah-langkah di bawah ini untuk setiap fitur.
        
        ---
        
        ### 1. 🔤 Transliterasi (Mengubah Teks Latin ke Tulisan Aksara Bima)
        Gunakan fitur ini untuk melihat bagaimana sebuah kata atau kalimat Latin ditulis dalam Aksara Bima.
        1.  **Pilih Fitur**: Klik **"Transliterasi"** di sidebar.
        2.  **Masukkan Teks**: Ketik kata atau kalimat yang ingin Anda ubah di kotak teks.
        3.  **Proses**: Klik tombol **"Transliterasi"**.
        4.  **Lihat Hasil**: Aplikasi akan menampilkan tulisan Aksara Bima dalam bentuk gambar yang bisa diunduh.
        
        ---
        
        ### 2. 🎯 Klasifikasi (Mengenali Satu Karakter Aksara Bima)
        Gunakan fitur ini jika Anda memiliki gambar satu karakter Aksara Bima dan ingin tahu namanya.
        1.  **Pilih Fitur**: Klik **"Klasifikasi"** di sidebar.
        2.  **Pilih Metode Input**:
            * **Gambar di Kanvas**: Pilih opsi ini untuk menggambar karakter secara langsung di layar.
            * **Unggah Gambar**: Pilih opsi ini untuk menggunakan file gambar dari komputer Anda.
        3.  **Proses**: Klik tombol **"Klasifikasi Sekarang"**.
        4.  **Lihat Hasil**: Aplikasi akan menampilkan nama karakter yang paling cocok beserta tingkat akurasinya.
            
        ---

        ### 3. 📖 OCR (Mengubah Gambar Tulisan Aksara Bima ke Teks Latin)
        Gunakan fitur ini untuk "membaca" tulisan Aksara Bima dari sebuah gambar.
        1.  **Pilih Fitur**: Klik **"OCR"** di sidebar.
        2.  **Unggah Gambar**: Pilih file gambar yang berisi tulisan Aksara Bima. **Penting**: Untuk hasil terbaik, gunakan gambar yang jelas, berisi **satu baris tulisan** dengan **maksimal 8 karakter**.
        3.  **Proses**: Klik tombol **"Proses OCR Sekarang"**.
        4.  **Lihat Hasil**: Teks Latin hasil pembacaan akan muncul di kotak teks.
        """)
        
    st.markdown("---")
    st.markdown("### 🚀 Pilih Fitur")
    cols = st.columns(3)
    if cols[0].button("Mulai Transliterasi", use_container_width=True): st.session_state.current_page = "transliterasi"
    if cols[1].button("Mulai Klasifikasi", use_container_width=True): st.session_state.current_page = "klasifikasi"
    if cols[2].button("Mulai OCR", use_container_width=True): st.session_state.current_page = "ocr"

def transliteration_page():
    """Halaman Transliterasi."""
    st.header("🔤 Transliterasi: Latin → Aksara Bima")
    st.markdown("---")
    char_images = load_character_images()
    
    input_text = st.text_area("Masukkan teks Latin di sini:", placeholder="Contoh: NDAI DOU MBOJO", height=100)
    
    # --- BLOK KODE YANG DIKEMBALIKAN SESUAI PERMINTAAN ---
    if st.button("🔄 Transliterasi", type="primary"):
        if input_text.strip():
            result = transliterate_to_bima(input_text)
            
            st.markdown("---")
            st.markdown("### 📝 Hasil Transliterasi")
            result_text = " + ".join([char if char != ' ' else '|' for char in result]).replace(' | ', '   ').replace('|+|', ' | ')
            st.markdown("**Pemecahan Karakter:**")
            st.code(result_text)

            full_image = create_full_text_image(result, char_images)
            
            if full_image:
                st.markdown("### 🖼️ Hasil Lengkap (Gambar)")
                st.image(full_image, caption=f"Transliterasi untuk: {input_text}", use_column_width=True)
                
                img_base64 = image_to_base64(full_image)
                st.download_button(
                    label="📥 Download Gambar",
                    data=base64.b64decode(img_base64),
                    file_name=f"transliterasi_bima_{input_text[:20].replace(' ', '_')}.png",
                    mime="image/png"
                )
    # --- AKHIR BLOK KODE ---

def classification_page():
    """Halaman Klasifikasi Karakter."""
    st.header("🎯 Klasifikasi: Kenali Karakter Aksara Bima")
    st.markdown("---")
    model = load_classification_model()

    if not model:
        st.error("Model klasifikasi tidak dapat dimuat. Fitur ini tidak tersedia.")
        return
        
    st.info(f"Model ini dapat mengenali **{len(CLASSIFICATION_CHARACTERS)} karakter** dasar.")
    with st.expander("📋 Lihat Daftar Karakter yang Didukung"):
        cols = st.columns(4)
        for i, char in enumerate(CLASSIFICATION_CHARACTERS):
            with cols[i % 4]:
                st.write(f"• **{char}**")

    input_method = st.radio("Pilih Metode Input:", ["🎨 Gambar di Kanvas", "📁 Unggah Gambar"], horizontal=True)
    
    image_to_process = None
    if input_method == "🎨 Gambar di Kanvas":
        st.markdown("Gambar **satu** karakter di dalam kotak putih di bawah ini:")
        canvas_result = st_canvas(stroke_width=15, stroke_color="black", background_color="white", height=300, width=300, key="canvas")
        if canvas_result.image_data is not None and canvas_result.image_data.any():
            image_to_process = canvas_result.image_data
    else:
        uploaded_file = st.file_uploader("Pilih file gambar (berisi satu karakter):", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            image_to_process = Image.open(uploaded_file)
            st.image(image_to_process, "Gambar yang diunggah:", width=250)
            
    if image_to_process is not None:
        if st.button("🔍 Klasifikasi Sekarang", type="primary", use_container_width=True):
            processed_image = preprocess_image(image_to_process)
            if processed_image is not None:
                with st.spinner("Menganalisis..."):
                    predictions = model.predict(processed_image, verbose=0)[0]
                    idx = np.argmax(predictions)
                    char = CLASSIFICATION_CHARACTERS[idx]
                    conf = predictions[idx]
                
                st.success(f"### Prediksi: **{char}**")
                st.metric("Tingkat Keyakinan", f"{conf:.2%}")
                st.progress(float(conf))
                # --- FITUR TOP 3 PREDIKSI DIHAPUS SESUAI PERMINTAAN ---

def ocr_page():
    """Halaman OCR."""
    st.header("📖 OCR: Gambar Aksara Bima → Teks Latin")
    st.markdown("---")
    
    st.warning(
        "⚠️ **Penting:** Untuk hasil terbaik, unggah gambar yang jelas berisi **satu baris** dengan **maksimal 8 karakter**."
    )
    
    uploaded_file = st.file_uploader("Unggah gambar Aksara Bima:", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Gambar yang diunggah", width=500)
        
        if st.button("🔍 Proses OCR Sekarang", type="primary", use_container_width=True):
            with st.spinner("Membaca teks dari gambar..."):
                tesseract_config = setup_tesseract()
                result_text = ocr_image(original_image) if tesseract_config else "Error: Tesseract tidak terkonfigurasi"

            if result_text:
                st.success("✅ OCR berhasil!")
                st.text_area("Hasil Teks Latin:", result_text, height=100)
            else:
                st.error("❌ OCR gagal mengenali teks.")

def info_page():
    """Halaman Informasi Sistem."""
    st.header("ℹ️ Informasi Sistem")
    st.markdown("---")
    st.markdown("### Status Komponen")
    
    model_ok = load_classification_model() is not None
    dataset_ok = os.path.exists(DATASET_PATH) and len(os.listdir(DATASET_PATH)) > 0
    tesseract_ok = setup_tesseract() is not None

    st.metric("Status Model Klasifikasi", "✅ Siap" if model_ok else "❌ Bermasalah")
    st.metric("Status Dataset", "✅ Siap" if dataset_ok else "❌ Bermasalah")
    st.metric("Status Mesin OCR (Tesseract)", "✅ Siap" if tesseract_ok else "❌ Bermasalah")
    st.markdown("---")
    st.markdown("### Tentang Aplikasi")
    st.info("Aplikasi ini dibuat menggunakan Python dan Streamlit untuk antarmuka pengguna, TensorFlow/Keras untuk model klasifikasi, dan Tesseract untuk OCR.")
    st.code(f"Lokasi Model: {MODEL_PATH}\nLokasi Dataset: {DATASET_PATH}", language=None)

# ===============================
# ROUTER & EKSEKUSI UTAMA
# ===============================
def main():
    """Fungsi utama untuk menjalankan aplikasi dan navigasi halaman."""
    st.sidebar.title("MENU NAVIGASI")
    if st.sidebar.button("🏠 Beranda", use_container_width=True): st.session_state.current_page = "home"
    if st.sidebar.button("🔤 Transliterasi", use_container_width=True): st.session_state.current_page = "transliterasi"
    if st.sidebar.button("🎯 Klasifikasi", use_container_width=True): st.session_state.current_page = "klasifikasi"
    if st.sidebar.button("📖 OCR", use_container_width=True): st.session_state.current_page = "ocr"
    if st.sidebar.button("ℹ️ Informasi Sistem", use_container_width=True): st.session_state.current_page = "info"
    
    pages = {
        "home": home_page,
        "transliterasi": transliteration_page,
        "klasifikasi": classification_page,
        "ocr": ocr_page,
        "info": info_page
    }
    pages.get(st.session_state.current_page, home_page)()

if __name__ == "__main__":
    main()
