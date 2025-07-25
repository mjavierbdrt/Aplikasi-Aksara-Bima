import streamlit as st
import os
import sys
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

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Aksara Bima",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS untuk styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    body {
        background-color: #ffffff;
    }

    .feature-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e9ecef;
        margin: 0.5rem 0;
    }

    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }

    .status-success {
        color: #28a745;
    }

    .status-error {
        color: #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# KONFIGURASI PATH
# ===============================
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_aksara_bima')
MODEL_PATH = os.path.join(BASE_DIR, 'models/aksara_bima_m.h5')
TESSDATA_PATH = os.path.join(BASE_DIR, 'tessdata')
TESSERACT_CMD = 'tesseract' # Pastikan tesseract ada di PATH sistem atau berikan path lengkap

os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Konstanta karakter Bima (untuk transliterasi lengkap)
BIMA_CHARACTERS = [
    'A', 'BA', 'BE', 'BI', 'BO', 'BU', 'CA', 'CE', 'CI', 'CO', 'CU', 'DA', 'DE',
    'DI', 'DO', 'DU', 'E', 'FA', 'FE', 'FI', 'FO', 'FU', 'GA', 'GE', 'GI', 'GO', 
    'GU', 'HA', 'HE', 'HI', 'HO', 'HU', 'I', 'JA', 'JE', 'JI', 'JO', 'JU', 'KA',
    'KE', 'KI', 'KO', 'KU', 'LA', 'LE', 'LI', 'LO', 'LU', 'MA', 'ME', 'MI', 'MO', 
    'MPA', 'MPE', 'MPI', 'MPO', 'MPU', 'MU', 'NA', 'NCA', 'NCE', 'NCI', 'NCO', 'NCU',
    'NE', 'NGA', 'NGE', 'NGI', 'NGO', 'NGU', 'NI', 'NO', 'NTA', 'NTE', 'NTI', 
    'NTO', 'NTU', 'NU', 'O', 'PA', 'PE', 'PI', 'PO', 'PU', 'RA', 'RE', 'RI', 'RO', 
    'RU', 'SA', 'SE', 'SI', 'SO', 'SU', 'TA', 'TE', 'TI', 'TO', 'TU', 'U', 'WA', 
    'WE', 'WI', 'WO', 'WU', 'YA', 'YE', 'YI', 'YO', 'YU',"NGGA"," NGGE", "NGGI", "NGGO", 
    "NGGU", "MBA", "MBE", "MBI", "MBO", "MBU", "NDA", "NDI", "NDO", "NDU","NDE",
    'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Y', 'NG'
]

# Konstanta karakter untuk klasifikasi model (22 karakter)
CLASSIFICATION_CHARACTERS = [
    'A', 'BA', 'CA', 'DA', 'FA', 'GA', 'HA', 'JA', 
    'KA', 'LA', 'MA', 'MPA', 'NA', 'NCA', 'NGA', 
    'NTA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA'
]

# Inisialisasi session state untuk navigasi
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# ===============================
# SETUP KONFIGURASI
# ===============================

def setup_tesseract():
    """Setup Tesseract dengan path yang benar."""
    try:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        custom_config = f'--tessdata-dir "{TESSDATA_PATH}" -l aksaralengkap --psm 8'
        pytesseract.get_tesseract_version()
        return custom_config
    except Exception:
        st.error(f"Error setting up Tesseract. Pastikan Tesseract terinstal dan path '{TESSERACT_CMD}' sudah benar.")
        return None

# ===============================
# FUNCTIONS UNTUK CROPPING
# ===============================

def crop_image_interactive(image, key_suffix=""):
    """Interactive cropping menggunakan streamlit-cropper."""
    if not CROPPER_AVAILABLE:
        st.error("⚠️ Fitur crop interaktif tidak tersedia. `streamlit-cropper` belum terinstal.")
        st.code("pip install streamlit-cropper")
        return None
    
    st.markdown("### ✂️ Crop Gambar")
    st.info("Geser dan ubah ukuran kotak untuk memilih area yang ingin diproses.")
    
    cropped_img = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0004',
        aspect_ratio=None,
        return_type='image',
        key=f"cropper_{key_suffix}"
    )
    
    if cropped_img:
        st.image(cropped_img, caption="✂️ Hasil Crop", use_container_width=True)
        crop_width, crop_height = cropped_img.size
        st.info(f"📐 Ukuran area crop: {crop_width} x {crop_height} piksel")
        return cropped_img
    
    return None

# ===============================
# FUNCTIONS UNTUK LOADING DATA
# ===============================

@st.cache_data
def load_character_images():
    """Load gambar karakter dari dataset."""
    char_images = {}
    if not os.path.exists(DATASET_PATH):
        st.error(f"Direktori dataset tidak ditemukan: {DATASET_PATH}")
        return char_images
    
    try:
        for folder_name in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, folder_name)
            if os.path.isdir(folder_path):
                image_files = glob.glob(os.path.join(folder_path, '*'))
                image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    try:
                        img = Image.open(image_files[0])
                        char_images[folder_name] = img
                    except Exception as e:
                        st.warning(f"Error memuat gambar untuk {folder_name}: {e}")
        return char_images
    except Exception as e:
        st.error(f"Error memuat dataset: {e}")
        return {}

@st.cache_resource
def load_classification_model():
    """Load model klasifikasi tanpa menampilkan pesan sukses di main page."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di: {MODEL_PATH}")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error memuat model: {e}")
        return None

# ===============================
# FUNCTIONS TRANSLITERASI
# ===============================

def transliterate_to_bima(text):
    """Transliterasi teks Latin ke aksara Bima (longest match first)."""
    sorted_chars = sorted(BIMA_CHARACTERS, key=len, reverse=True)
    text = text.upper().strip()
    result = []
    i = 0
    while i < len(text):
        if text[i] == ' ':
            result.append(' ')
            i += 1
            continue
            
        matched = False
        for char in sorted_chars:
            if text[i:i+len(char)] == char:
                result.append(char)
                i += len(char)
                matched = True
                break
        
        if not matched:
            result.append(f"[{text[i]}]") # Karakter tidak dikenal
            i += 1
    return result

def create_character_image(char, char_images, width=80, height=80):
    """Membuat gambar karakter dari dataset atau placeholder."""
    if char in char_images:
        img = char_images[char].copy()
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
    
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    draw.rectangle([2, 2, width-3, height-3], outline='lightgray', width=1)
    try:
        font = ImageFont.load_default(size=20)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (width - text_width) // 2, (height - text_height) // 2
        draw.text((x, y), char, fill='red', font=font)
    except Exception:
        draw.text((10, height//2-10), char, fill='red')
    return img

def combine_images(images, spacing=5):
    """Menggabungkan list gambar menjadi satu gambar horizontal."""
    if not images:
        return None
    
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    combined = Image.new('RGB', (total_width, max_height), color='white')
    
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    return combined

def create_full_text_image(result, char_images, char_spacing=5, word_spacing=20):
    """Membuat gambar untuk seluruh teks yang telah ditransliterasi."""
    all_images = []
    for char in result:
        if char == ' ':
            if all_images:
                space_img = Image.new('RGB', (word_spacing, 80), color='white')
                all_images.append(space_img)
        elif not char.startswith('['):
            img = create_character_image(char, char_images)
            all_images.append(img)
    
    if not all_images:
        return None

    # Menggunakan fungsi combine_images yang sudah disederhanakan
    # Logika spasi antar kata sudah dihandle dengan menambahkan gambar kosong
    return combine_images(all_images, spacing=char_spacing)


# ===============================
# FUNCTIONS KLASIFIKASI & OCR
# ===============================

def preprocess_image(image, is_from_canvas=False):
    """Preprocess gambar (dari upload/canvas) untuk model klasifikasi."""
    try:
        if is_from_canvas:
            if image.shape[-1] == 4: # RGBA dari canvas
                rgb_canvas = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
                alpha = image[:, :, 3:4] / 255.0
                rgb_canvas = (1 - alpha) * rgb_canvas + alpha * image[:, :, :3]
                img_array = rgb_canvas.astype(np.uint8)
            else:
                img_array = image
        else: # Dari PIL Image
            if image.mode != 'RGB':
                image = image.convert('RGB')
            img_array = np.array(image)

        resized = cv2.resize(img_array, (224, 224))
        normalized = resized.astype('float32') / 255.0
        input_image = np.expand_dims(normalized, axis=0)
        return input_image
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None

def classify_character(image_input, model):
    """Klasifikasi karakter menggunakan model."""
    if model is None: return None, 0.0
    try:
        predictions = model.predict(image_input, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class_idx])
        if predicted_class_idx < len(CLASSIFICATION_CHARACTERS):
            character = CLASSIFICATION_CHARACTERS[predicted_class_idx]
            return character, confidence, predictions[0]
        else:
            return "Unknown", confidence, predictions[0]
    except Exception as e:
        st.error(f"Error pada saat klasifikasi: {e}")
        return None, 0.0, None

def ocr_bima_to_latin(image, tesseract_config):
    """OCR Aksara Bima ke Latin."""
    try:
        # Konversi PIL image ke format yang bisa dibaca OpenCV
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Menggunakan konfigurasi yang sudah disiapkan
        text = pytesseract.image_to_string(thresh, config=tesseract_config)
        return text.strip()
    except Exception as e:
        st.error(f"Error saat proses OCR: {e}")
        return ""

# ===============================
# UTILITY FUNCTIONS
# ===============================

def image_to_base64(img):
    """Konversi gambar PIL ke base64 untuk diunduh."""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def check_system_requirements():
    """Check ketersediaan komponen sistem."""
    tesseract_ok = False
    try:
        pytesseract.get_tesseract_version()
        tesseract_ok = True
    except Exception:
        tesseract_ok = False
        
    return {
        'dataset': os.path.exists(DATASET_PATH) and len(os.listdir(DATASET_PATH)) > 0,
        'model': os.path.exists(MODEL_PATH),
        'tesseract': tesseract_ok
    }

# ===============================
# SIDEBAR NAVIGATION
# ===============================

def sidebar_navigation():
    """Sidebar dengan tombol navigasi dan status sistem."""
    st.sidebar.title("🔤 Sistem Aksara Bima")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Fitur Utama")
    if st.sidebar.button("🏠 Beranda", key="home_btn"):
        st.session_state.current_page = "home"
    if st.sidebar.button("🔤 Transliterasi", key="trans_btn"):
        st.session_state.current_page = "transliterasi"
    if st.sidebar.button("🎯 Klasifikasi", key="class_btn"):
        st.session_state.current_page = "klasifikasi"
    if st.sidebar.button("📖 OCR", key="ocr_btn"):
        st.session_state.current_page = "ocr"
    
    st.sidebar.markdown("---")
    if st.sidebar.button("ℹ️ Informasi Sistem", key="info_btn"):
        st.session_state.current_page = "info"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status Sistem")
    
    status = check_system_requirements()
    
    st.sidebar.success("✅ Dataset Siap") if status['dataset'] else st.sidebar.error("❌ Dataset Bermasalah")
    st.sidebar.success("✅ Model Siap") if status['model'] else st.sidebar.error("❌ Model Tidak Ditemukan")
    st.sidebar.success("✅ OCR Siap") if status['tesseract'] else st.sidebar.error("❌ OCR (Tesseract) Bermasalah")

# ===============================
# HALAMAN APLIKASI (PAGES)
# ===============================

def home_page():
    """Halaman beranda."""
    logo_path = os.path.join(BASE_DIR, 'logo/logo.png')
    col1, col2 = st.columns([1, 5])
    if os.path.exists(logo_path):
        with col1:
            st.image(logo_path, width=120)
    with col2:
        st.title("Aplikasi Pengenalan Aksara Bima")
    
    st.markdown("---")
    
    st.markdown("## Selamat Datang!")
    st.markdown("""
    Aplikasi terintegrasi ini menyediakan tiga fitur utama untuk membantu Anda mengenal, menulis, dan membaca Aksara Bima. Silakan pilih salah satu fitur di bawah atau melalui menu di sidebar.
    """)
    
    # --- PANDUAN PENGGUNAAN BARU ---
    with st.expander("📖 Panduan Penggunaan Aplikasi", expanded=False):
        st.markdown("""
        Berikut adalah panduan singkat untuk menggunakan setiap fitur dalam aplikasi ini.

        ### 1. 🔤 Transliterasi (Latin → Aksara Bima)
        Fitur ini mengubah teks Latin yang Anda ketik menjadi gambar Aksara Bima.
        - **Cara Penggunaan:**
            1.  Buka halaman **Transliterasi** dari sidebar.
            2.  Ketik kata atau kalimat dalam bahasa Latin di kotak teks yang tersedia. Contoh: `NDAI DOU MBOJO`.
            3.  Klik tombol **"Transliterasi"**.
            4.  Aplikasi akan menampilkan pemecahan karakter, visualisasi per kata, dan gambar lengkap dari hasil transliterasi.
            5.  Anda dapat mengunduh gambar hasil transliterasi dengan mengklik tombol **"Download Gambar"**.

        ### 2. 🎯 Klasifikasi Karakter (Gambar → Karakter)
        Fitur ini mengenali satu karakter Aksara Bima dari gambar yang Anda berikan.
        - **Cara Penggunaan:**
            - **Opsi A: Menggambar di Kanvas**
                1. Buka halaman **Klasifikasi**.
                2. Pilih metode input **"🎨 Canvas Drawing"**.
                3. Gambarlah **satu** karakter Aksara Bima di kanvas putih yang tersedia. Usahakan gambar jelas dan tebal.
                4. Klik tombol **"Klasifikasi"**.
            - **Opsi B: Mengunggah Gambar**
                1. Buka halaman **Klasifikasi**.
                2. Pilih metode input **"📁 Upload Gambar"**.
                3. Unggah file gambar yang berisi **satu** karakter Aksara Bima.
                4. (Opsional) Jika gambar terlalu besar atau berisi elemen lain, centang kotak **"✂️ Crop gambar"** untuk memilih area spesifik karakter.
                5. Klik tombol **"Klasifikasi"**.
            - **Hasil:** Aplikasi akan menampilkan karakter yang paling sesuai beserta tingkat keyakinan (confidence score).

        ### 3. 📖 OCR (Gambar Teks → Teks Latin)
        Fitur ini membaca gambar yang berisi tulisan Aksara Bima dan mengubahnya menjadi teks Latin.
        - **Cara Penggunaan:**
            1. Buka halaman **OCR** dari sidebar.
            2. Unggah gambar yang berisi satu baris tulisan Aksara Bima.
            3. (Opsional tapi disarankan) Centang kotak **"✂️ Crop area teks"** untuk memilih hanya bagian teks yang ingin dibaca. Ini akan meningkatkan akurasi.
            4. Klik tombol **"Proses OCR"**.
            5. Hasilnya akan ditampilkan dalam bentuk teks Latin yang dapat Anda salin atau unduh.
        """)

    st.markdown("---")
    
    st.markdown("## 🚀 Fitur Utama")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔤 Transliterasi")
        st.markdown("Konversi teks Latin ke aksara Bima dengan algoritma *longest match first* untuk hasil yang akurat.")
        if st.button("🔄 Mulai Transliterasi", key="start_trans"):
            st.session_state.current_page = "transliterasi"
            st.rerun()
    with col2:
        st.markdown("### 🎯 Klasifikasi")
        st.markdown("Identifikasi karakter aksara Bima dari gambar atau tulisan tangan Anda secara langsung.")
        if st.button("🔍 Mulai Klasifikasi", key="start_class"):
            st.session_state.current_page = "klasifikasi"
            st.rerun()
    with col3:
        st.markdown("### 📖 OCR")
        st.markdown("Ekstrak teks dari gambar aksara Bima dan ubah menjadi teks Latin menggunakan Tesseract.")
        if st.button("📷 Mulai OCR", key="start_ocr"):
            st.session_state.current_page = "ocr"
            st.rerun()
            
def transliteration_page():
    """Halaman transliterasi Latin ke Aksara Bima."""
    st.header("🔤 Transliterasi Latin → Aksara Bima")
    st.markdown("---")
    
    char_images = load_character_images()
    
    st.markdown("Aplikasi ini mengubah teks Latin menjadi aksara Bima. Algoritma menggunakan prinsip **longest match first** untuk hasil yang optimal.")
    
    input_text = st.text_area("Masukkan teks Latin:", placeholder="Contoh: NGAHA, MPALA, BIMA SAKTI", height=100)
    
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
            else:
                st.error("Gagal membuat gambar. Periksa input dan pastikan dataset termuat.")

def classification_page():
    """Halaman klasifikasi karakter."""
    st.header("🎯 Klasifikasi Karakter Aksara Bima")
    st.markdown("---")
    
    model = load_classification_model()
    char_images = load_character_images()
    
    if model is None:
        st.error("Model klasifikasi tidak tersedia. Fitur ini tidak dapat digunakan.")
        return
    
    st.info(f"Model ini dapat mengklasifikasikan **{len(CLASSIFICATION_CHARACTERS)} karakter** dasar aksara Bima.")
    with st.expander("📋 Lihat Daftar Karakter yang Didukung"):
        cols = st.columns(3)
        for i, char in enumerate(CLASSIFICATION_CHARACTERS):
            with cols[i % 3]:
                st.write(f"• {char}")
    
    input_method = st.radio("Pilih metode input:", ["🎨 Canvas Drawing", "📁 Upload Gambar"], horizontal=True)
    
    processed_image = None
    original_display = None
    
    if input_method == "🎨 Canvas Drawing":
        st.markdown("### 🎨 Gambar Karakter di Kanvas")
        st.info("Pastikan gambar karakter jelas dan berada di tengah kanvas untuk hasil terbaik.")
        
        try:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=15, stroke_color="black", background_color="white",
                width=400, height=400, drawing_mode="freedraw", key="canvas"
            )
            
            if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
                processed_image = preprocess_image(canvas_result.image_data, is_from_canvas=True)
                original_display = canvas_result.image_data
                st.image(canvas_result.image_data, caption="Gambar yang akan diproses", width=250)
                
        except Exception as e:
            st.error(f"Error dengan kanvas: {e}")

    elif input_method == "📁 Upload Gambar":
        st.markdown("### 📁 Upload Gambar Karakter")
        uploaded_file = st.file_uploader("Pilih file gambar:", type=['png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            original_image = Image.open(uploaded_file)
            final_image_to_process = original_image

            st.image(original_image, caption="📷 Gambar Asli", use_container_width=True)

            if CROPPER_AVAILABLE:
                use_crop = st.checkbox("✂️ Crop gambar sebelum klasifikasi", help="Pilih area spesifik dari gambar.")
                if use_crop:
                    cropped_image = crop_image_interactive(original_image, "classification")
                    if cropped_image:
                        final_image_to_process = cropped_image
                        st.success("✅ Menggunakan gambar yang di-crop untuk klasifikasi.")

            processed_image = preprocess_image(final_image_to_process, is_from_canvas=False)
            original_display = np.array(final_image_to_process)

    st.markdown("---")
    if processed_image is not None and st.button("🔍 Klasifikasi Sekarang", type="primary", use_container_width=True):
        with st.spinner("Menganalisis gambar..."):
            character, confidence, all_preds = classify_character(processed_image, model)
            
            if character:
                st.success(f"### Hasil Prediksi: **{character}**")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Gambar Input Anda:**")
                    if original_display is not None:
                        st.image(original_display, width=200)
                with col2:
                    st.markdown("**Referensi dari Dataset:**")
                    if character in char_images:
                        st.image(char_images[character], width=200)
                    else:
                        st.warning("Gambar referensi tidak tersedia.")
                
            else:
                st.error("Klasifikasi gagal. Coba lagi dengan gambar yang lebih jelas.")


def ocr_page():
    """
    Halaman OCR Aksara Bima ke Latin
    """
    st.header("📖 OCR Aksara Bima → Latin")
    st.markdown("---")

    tesseract_config = setup_tesseract()

    if tesseract_config is None:
        st.error("Tesseract tidak tersedia. Pastikan path sudah benar.")
        st.code(f"Expected path: {TESSERACT_PATH}")
        return

    st.markdown("""
    Unggah gambar yang berisi teks aksara Bima untuk diubah menjadi teks Latin.
    """)

    # --- PERUBAHAN DI SINI: Menambahkan Peringatan ---
    st.warning(
        "⚠️ **Perhatian:** Untuk hasil OCR yang lebih akurat, "
        "mohon unggah gambar yang hanya berisi **satu baris** tulisan "
        "dengan jumlah **maksimal 8 karakter**."
    )

    uploaded_file = st.file_uploader(
        "Upload gambar aksara Bima (satu baris, maks. 8 karakter):",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar untuk OCR"
    )
    if uploaded_file is not None:
        original_image = Image.open(uploaded_file)
        final_image_to_process = original_image
        
        st.image(original_image, caption="📷 Gambar Asli", use_container_width=True)
        
        if CROPPER_AVAILABLE:
            use_crop = st.checkbox("✂️ Crop area teks untuk akurasi lebih baik", help="Fokuskan OCR pada area teks yang relevan.")
            if use_crop:
                cropped_image = crop_image_interactive(original_image, "ocr")
                if cropped_image:
                    final_image_to_process = cropped_image
                    st.success("✅ Menggunakan area yang di-crop untuk OCR.")

        st.markdown("---")
        if st.button("🔍 Proses OCR Sekarang", type="primary", use_container_width=True):
            with st.spinner("🔄 Membaca teks dari gambar..."):
                # Pastikan gambar dalam mode RGB untuk konsistensi
                if final_image_to_process.mode != 'RGB':
                    final_image_to_process = final_image_to_process.convert('RGB')

                result_text = ocr_bima_to_latin(final_image_to_process, tesseract_config)
                
                if result_text:
                    st.success("✅ OCR berhasil!")
                    st.markdown("### 📝 Hasil OCR (Teks Latin):")
                    st.text_area("Teks yang berhasil dibaca:", result_text, height=150)
                    st.download_button("📥 Download Hasil Teks", result_text, "hasil_ocr.txt")
                else:
                    st.error("❌ OCR gagal mengenali teks.")
                    with st.expander("💡 Tips untuk Meningkatkan Hasil OCR"):
                        st.markdown("""
                        - **Gunakan Fitur Crop**: Centang opsi "Crop area teks" dan pilih hanya bagian tulisan.
                        - **Kualitas Gambar**: Pastikan gambar memiliki kontras yang baik (tulisan gelap, latar belakang terang) dan tidak buram.
                        - **Orientasi**: Pastikan teks dalam posisi horizontal.
                        - **Pencahayaan**: Hindari bayangan pada teks.
                        """)
                    st.image(final_image_to_process, "Gambar yang diproses OCR (gagal).")

def info_page():
    """Halaman informasi sistem."""
    st.header("ℹ️ Informasi Sistem & Teknologi")
    st.markdown("---")
    
    requirements = check_system_requirements()
    
    st.markdown("### 🔧 Status Kesiapan Sistem")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Dataset Gambar", "✅ Siap" if requirements['dataset'] else "❌ Bermasalah")
    with col2:
        st.metric("Model Klasifikasi", "✅ Siap" if requirements['model'] else "❌ Bermasalah")
    with col3:
        st.metric("Mesin OCR", "✅ Siap" if requirements['tesseract'] else "❌ Bermasalah")

    st.markdown("---")
    
    st.markdown("""
    ### 🎯 Tentang Aplikasi
    Aplikasi ini dibangun untuk melestarikan dan memfasilitasi pembelajaran Aksara Bima melalui teknologi. Berikut adalah rincian teknis dari setiap fitur:

    - **Transliterasi**: Menggunakan algoritma *longest match first* untuk memetakan fonem Latin ke karakter Aksara Bima yang sesuai. Proses ini memastikan bahwa unit karakter yang lebih panjang (misalnya 'MPA') diprioritaskan daripada yang lebih pendek ('MA' atau 'PA').
    - **Klasifikasi**: Menggunakan model *Deep Learning* (arsitektur Convolutional Neural Network) yang telah dilatih pada dataset gambar 22 karakter dasar Aksara Bima. Gambar input diproses menjadi ukuran 224x224 piksel sebelum dimasukkan ke model.
    - **OCR**: Memanfaatkan *Tesseract OCR Engine* yang dikonfigurasi khusus dengan file `traineddata` untuk Aksara Bima (`aksaralengkap.traineddata`). Gambar dipra-pemrosesan (Grayscale, Binarization) untuk memaksimalkan akurasi pengenalan.

    ### 📦 Dependensi Utama
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("- `streamlit`\n- `tensorflow`\n- `opencv-python-headless`\n- `pytesseract`")
    with col2:
        st.markdown("- `pillow`\n- `numpy`\n- `streamlit-drawable-canvas`\n- `streamlit-cropper` (Opsional)")

# ===============================
# MAIN APP ROUTER
# ===============================

def main():
    """Fungsi utama untuk menjalankan aplikasi dan navigasi halaman."""
    sidebar_navigation()
    
    page_map = {
        "home": home_page,
        "transliterasi": transliteration_page,
        "klasifikasi": classification_page,
        "ocr": ocr_page,
        "info": info_page
    }
    
    # Jalankan fungsi halaman yang sesuai dengan state
    page_function = page_map.get(st.session_state.current_page, home_page)
    page_function()

if __name__ == "__main__":
    main()
