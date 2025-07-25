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
    .stButton > button {
        width: 100%;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===============================
# KONFIGURASI PATH (LEBIH PORTABEL)
# ===============================
BASE_DIR = os.path.dirname(__file__)
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_aksara_bima')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'aksara_bima_m.h5')
TESSDATA_PATH = os.path.join(BASE_DIR, 'tessdata')
TESSERACT_CMD = 'tesseract' 

# Atur environment variable untuk Tesseract
os.environ['TESSDATA_PREFIX'] = TESSDATA_PATH
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# Konstanta karakter Bima (sudah dirapikan)
BIMA_CHARACTERS = [
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
]

# Konstanta karakter untuk klasifikasi model
CLASSIFICATION_CHARACTERS = [
    'A', 'BA', 'CA', 'DA', 'FA', 'GA', 'HA', 'JA',
    'KA', 'LA', 'MA', 'MPA', 'NA', 'NCA', 'NGA',
    'NTA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA'
]

# Inisialisasi session state untuk navigasi
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# ===============================
# SETUP & LOADING FUNCTIONS
# ===============================

@st.cache_data
def load_character_images():
    """Load gambar karakter dari dataset."""
    char_images = {}
    if not os.path.exists(DATASET_PATH):
        st.error(f"Direktori dataset tidak ditemukan: {DATASET_PATH}")
        return char_images
    try:
        for folder_name in sorted(os.listdir(DATASET_PATH)):
            folder_path = os.path.join(DATASET_PATH, folder_name)
            if os.path.isdir(folder_path):
                image_files = glob.glob(os.path.join(folder_path, '*'))
                image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    try:
                        img = Image.open(image_files[0])
                        char_images[folder_name] = img
                    except Exception:
                        pass
        return char_images
    except Exception as e:
        st.error(f"Error memuat dataset: {e}")
        return {}

@st.cache_resource
def load_classification_model():
    """Load model klasifikasi tanpa menampilkan pesan di UI."""
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception:
        return None

# ===============================
# CORE LOGIC FUNCTIONS
# ===============================

def transliterate_to_bima(text):
    """Transliterasi teks Latin ke aksara Bima (longest match first)."""
    sorted_chars = sorted(list(set(BIMA_CHARACTERS)), key=len, reverse=True)
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
            if text[i:].startswith(char):
                result.append(char)
                i += len(char)
                matched = True
                break
        if not matched:
            result.append(f"[{text[i]}]")
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
    
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=20)
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x, y = (width - text_width) // 2, (height - text_height) // 2
        draw.text((x, y), f"{char}\n(x)", fill='red', font=font, align="center")
    except Exception:
        draw.text((10, 10), f"{char}\n(x)", fill='red')
    return img

def create_full_text_image(result, char_images, char_spacing=5, word_spacing=20):
    """Membuat gambar untuk seluruh teks yang telah ditransliterasi."""
    images = []
    for char in result:
        if char == ' ':
            if images:
                images.append(Image.new('RGB', (word_spacing, 80), color=(255, 255, 255)))
        elif not char.startswith('['):
            images.append(create_character_image(char, char_images))
    if not images: return None
    
    total_width = sum(img.width for img in images) + char_spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    combined = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width + char_spacing
    return combined

def preprocess_image(image_input):
    """Preprocess gambar (dari PIL Image/Numpy array) untuk model."""
    try:
        if isinstance(image_input, Image.Image):
            if image_input.mode != 'RGB':
                image_input = image_input.convert('RGB')
            img_array = np.array(image_input)
        else: # Asumsikan numpy array dari canvas
            if image_input.shape[-1] == 4: # RGBA
                alpha = image_input[:, :, 3:4] / 255.0
                bg = np.ones_like(image_input[:, :, :3]) * 255
                img_array = (alpha * image_input[:, :, :3] + (1 - alpha) * bg).astype(np.uint8)
            else:
                img_array = image_input

        resized = cv2.resize(img_array, (224, 224))
        normalized = resized.astype('float32') / 255.0
        return np.expand_dims(normalized, axis=0)
    except Exception as e:
        st.error(f"Error saat memproses gambar: {e}")
        return None

def classify_character(image_input, model):
    """Klasifikasi karakter, mengembalikan prediksi teratas dan semua prediksi."""
    if model is None: return None, 0.0, None
    try:
        predictions = model.predict(image_input, verbose=0)[0]
        idx = np.argmax(predictions)
        char = CLASSIFICATION_CHARACTERS[idx] if idx < len(CLASSIFICATION_CHARACTERS) else "Unknown"
        conf = float(predictions[idx])
        return char, conf, predictions
    except Exception as e:
        st.error(f"Error pada saat klasifikasi: {e}")
        return None, 0.0, None

def ocr_bima_to_latin(image):
    """OCR Aksara Bima ke Latin dari objek gambar PIL."""
    try:
        config = f'--tessdata-dir "{TESSDATA_PATH}" -l aksaralengkap --psm 8'
        return pytesseract.image_to_string(image, config=config).strip()
    except Exception as e:
        st.error(f"Error saat proses OCR: {e}")
        return ""

# ===============================
# UI COMPONENTS & PAGES
# ===============================

def sidebar_navigation():
    """Sidebar dengan navigasi dan status sistem."""
    st.sidebar.title("🔤 Sistem Aksara Bima")
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Fitur Utama")
    if st.sidebar.button("🏠 Beranda", use_container_width=True):
        st.session_state.current_page = "home"
    if st.sidebar.button("🔤 Transliterasi", use_container_width=True):
        st.session_state.current_page = "transliterasi"
    if st.sidebar.button("🎯 Klasifikasi", use_container_width=True):
        st.session_state.current_page = "klasifikasi"
    if st.sidebar.button("📖 OCR", use_container_width=True):
        st.session_state.current_page = "ocr"
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status Sistem")
    
    # Cek status sekali saja
    model_ok = load_classification_model() is not None
    dataset_ok = os.path.exists(DATASET_PATH)
    tesseract_ok = True
    try:
        pytesseract.get_tesseract_version()
    except Exception:
        tesseract_ok = False

    st.sidebar.success("✅ Dataset Siap") if dataset_ok else st.sidebar.error("❌ Dataset Bermasalah")
    st.sidebar.success("✅ Model Siap") if model_ok else st.sidebar.error("❌ Model Tidak Ditemukan")
    st.sidebar.success("✅ OCR (Tesseract) Siap") if tesseract_ok else st.sidebar.error("❌ OCR Bermasalah")


def home_page():
    """Halaman Beranda."""
    logo_path = os.path.join(BASE_DIR, 'logo', 'logo.png')
    col1, col2 = st.columns([1, 6])
    if os.path.exists(logo_path):
        with col1:
            st.image(logo_path, width=120)
    with col2:
        st.title("Aplikasi Pengenalan Aksara Bima")
        st.markdown("Sebuah alat bantu untuk transliterasi, klasifikasi, dan OCR Aksara Bima.")
    st.markdown("---")

    with st.expander("📖 Panduan Penggunaan Aplikasi", expanded=True):
        st.markdown("""
        Selamat datang! Berikut adalah panduan singkat untuk menggunakan setiap fitur.
        - **Transliterasi**: Ketik teks Latin, lalu klik tombol "Transliterasi" untuk mengubahnya menjadi gambar Aksara Bima.
        - **Klasifikasi**: Gambar atau unggah satu karakter Aksara Bima untuk mengetahui namanya.
        - **OCR**: Unggah gambar berisi tulisan Aksara Bima (satu baris, sedikit karakter) untuk mengubahnya menjadi teks Latin.
        """)

    st.markdown("### 🚀 Fitur Utama")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 🔤 Transliterasi")
        st.markdown("Ubah teks Latin menjadi Aksara Bima secara visual.")
        if st.button("Mulai Transliterasi"):
            st.session_state.current_page = "transliterasi"
            st.rerun()
    with col2:
        st.markdown("#### 🎯 Klasifikasi")
        st.markdown("Kenali karakter Aksara Bima dari gambar atau tulisan tangan.")
        if st.button("Mulai Klasifikasi"):
            st.session_state.current_page = "klasifikasi"
            st.rerun()
    with col3:
        st.markdown("#### 📖 OCR")
        st.markdown("Ekstrak teks dari gambar Aksara Bima ke tulisan Latin.")
        if st.button("Mulai OCR"):
            st.session_state.current_page = "ocr"
            st.rerun()
    st.markdown("---")

def transliteration_page():
    """Halaman Transliterasi."""
    st.header("🔤 Transliterasi Latin → Aksara Bima")
    st.markdown("---")
    char_images = load_character_images()
    
    input_text = st.text_area("Masukkan teks Latin:", placeholder="Contoh: NDAI DOU MBOJO", height=100)
    
    if st.button("🔄 Transliterasi", type="primary"):
        if input_text.strip():
            result = transliterate_to_bima(input_text)
            st.markdown("---")
            st.markdown("#### Hasil Transliterasi")
            
            # Tampilkan pemecahan karakter
            result_text = " + ".join([f"'{c}'" if c != ' ' else 'SPASI' for c in result])
            st.markdown(f"**Pemecahan Karakter:** `{result_text}`")
            
            full_image = create_full_text_image(result, char_images)
            if full_image:
                st.image(full_image, caption=f"Gambar untuk: {input_text}")
                buffered = io.BytesIO()
                full_image.save(buffered, format="PNG")
                st.download_button("📥 Download Gambar", data=buffered.getvalue(), file_name="transliterasi.png", mime="image/png")
            else:
                st.error("Gagal membuat gambar. Pastikan input tidak kosong.")
        else:
            st.warning("Mohon masukkan teks terlebih dahulu.")

def classification_page():
    """Halaman Klasifikasi."""
    st.header("🎯 Klasifikasi Karakter Aksara Bima")
    st.markdown("---")
    model = load_classification_model()
    char_images = load_character_images()

    if model is None:
        st.error("Model klasifikasi tidak tersedia. Fitur ini tidak dapat digunakan.")
        return

    st.info(f"Model ini dapat mengklasifikasikan **{len(CLASSIFICATION_CHARACTERS)} karakter** dasar.")

    # --- PERUBAHAN DI SINI: Menambahkan kembali daftar karakter ---
    with st.expander("📋 Lihat Daftar Karakter yang Didukung"):
        cols = st.columns(3)
        # Loop untuk membagi karakter ke dalam 3 kolom
        for i, char in enumerate(CLASSIFICATION_CHARACTERS):
            with cols[i % 3]:
                st.write(f"• {char}")
    # -----------------------------------------------------------

    input_method = st.radio("Pilih metode input:", ["🎨 Gambar di Kanvas", "📁 Unggah Gambar"], horizontal=True)

    final_image = None
    if input_method == "🎨 Gambar di Kanvas":
        st.markdown("#### Gambar satu karakter di kanvas bawah ini:")
        canvas_result = st_canvas(
            stroke_width=15, stroke_color="black", background_color="white",
            height=300, width=300, drawing_mode="freedraw", key="canvas"
        )
        if canvas_result.image_data is not None and np.any(canvas_result.image_data[:, :, 3] > 0):
            final_image = canvas_result.image_data
            st.image(final_image, caption="Gambar Anda", width=200)

    else: # Unggah Gambar
        uploaded_file = st.file_uploader("Pilih file gambar (berisi satu karakter):", type=['png', 'jpg', 'jpeg'])
        if uploaded_file:
            original_image = Image.open(uploaded_file)
            image_to_process = original_image
            st.image(original_image, "Gambar Asli", use_container_width=False, width=250)
            
            if CROPPER_AVAILABLE:
                if st.checkbox("✂️ Crop gambar untuk hasil lebih baik"):
                    cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0004')
                    if cropped_image:
                        st.image(cropped_image, "Hasil Crop", use_container_width=False, width=250)
                        image_to_process = cropped_image
            final_image = image_to_process

    if final_image is not None:
        if st.button("🔍 Klasifikasi Sekarang", type="primary"):
            processed_image_for_model = preprocess_image(final_image)
            if processed_image_for_model is not None:
                with st.spinner("Menganalisis..."):
                    char, conf, all_preds = classify_character(processed_image_for_model, model)
                
                st.success(f"Hasil Prediksi: **{char}**")
                st.metric("Tingkat Keyakinan", f"{conf:.2%}")
                st.progress(conf)
                
                if char in char_images:
                    st.markdown("##### Contoh Karakter dari Dataset:")
                    st.image(char_images[char], width=150)

def ocr_page():
    """Halaman OCR."""
    st.header("📖 OCR Aksara Bima → Latin")
    st.markdown("---")

    st.warning(
        "⚠️ **Perhatian:** Untuk hasil OCR yang lebih akurat, "
        "mohon unggah gambar yang hanya berisi **satu baris** tulisan "
        "dengan jumlah **maksimal 8 karakter**."
    )

    uploaded_file = st.file_uploader(
        "Upload gambar aksara Bima (satu baris, maks. 8 karakter):",
        type=['png', 'jpg', 'jpeg']
    )

    if uploaded_file:
        original_image = Image.open(uploaded_file)
        image_to_process = original_image
        st.image(original_image, caption="Gambar Asli", use_container_width=False, width=400)

        if CROPPER_AVAILABLE and st.checkbox("✂️ Crop area teks"):
             cropped_image = st_cropper(original_image, realtime_update=True, box_color='#FF0004')
             if cropped_image:
                st.image(cropped_image, "Hasil Crop", width=300)
                image_to_process = cropped_image

        if st.button("🔍 Proses OCR Sekarang", type="primary"):
            with st.spinner("Membaca teks dari gambar..."):
                if image_to_process.mode != 'RGB':
                    image_to_process = image_to_process.convert('RGB')
                
                result_text = ocr_bima_to_latin(image_to_process)
            
            if result_text:
                st.success("✅ OCR berhasil!")
                st.text_area("Hasil Teks Latin:", result_text, height=100)
            else:
                st.error("❌ OCR gagal mengenali teks. Coba gunakan gambar yang lebih jelas atau crop area teks.")

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
    }
    
    # Jalankan fungsi halaman yang sesuai dengan state
    page_function = page_map.get(st.session_state.current_page, home_page)
    page_function()

if __name__ == "__main__":
    main()
