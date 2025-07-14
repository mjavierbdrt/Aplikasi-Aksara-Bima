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

# Untuk fitur cropping
try:
    from streamlit_cropper import st_cropper
    CROPPER_AVAILABLE = True
except ImportError:
    CROPPER_AVAILABLE = False
    st.warning("streamlit-cropper tidak tersedia. Install dengan: pip install streamlit-cropper")

# Konfigurasi halaman
st.set_page_config(
    page_title="Sistem Aksara Bima",
    page_icon="🔤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===============================
# KONFIGURASI PATH
# ===============================

# Path direktori yang disesuaikan dengan sistem user
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset_aksara_bima')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models/aksara_bima_model_A.h5')
TESSDATA_PATH = os.path.join(os.path.dirname(__file__), 'tessdata')
TESSERACT_CMD = 'tesseract'

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
    'WE', 'WI', 'WO', 'WU', 'YA', 'YE', 'YI', 'YO', 'YU',
    'B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Y', 'NG'
]

# Konstanta karakter untuk klasifikasi model (22 karakter)
CLASSIFICATION_CHARACTERS = [
    'A', 'BA', 'CA', 'DA', 'FA', 'GA', 'HA', 'JA', 
    'KA', 'LA', 'MA', 'MPA', 'NA', 'NCA', 'NGA', 
    'NTA', 'PA', 'RA', 'SA', 'TA', 'WA', 'YA'
]

# Initialize session state untuk navigasi
if 'current_page' not in st.session_state:
    st.session_state.current_page = "home"

# ===============================
# SETUP KONFIGURASI
# ===============================

def setup_tesseract():
    """
    Setup Tesseract dengan path yang benar
    """
    try:
        # Set path untuk tesseract
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD
        
        # Konfigurasi untuk OCR aksara Bima
        # Menggunakan aksarareal.traineddata yang sudah ada
        custom_config = f'--tessdata-dir "{TESSDATA_PATH}" -l aksaralengkap --psm 8'
        
        # Test jika tesseract berjalan
        pytesseract.get_tesseract_version()
        
        return custom_config
        
    except Exception as e:
        st.error(f"Error setting up Tesseract: {e}")
        st.error("Pastikan Tesseract sudah terinstall dan path sudah benar")
        return None

# ===============================
# FUNCTIONS UNTUK CROPPING
# ===============================

def crop_image_manual(image):
    """
    Manual cropping menggunakan koordinat yang diinput user
    """
    st.markdown("### ✂️ Crop Gambar Manual")
    
    # Display image info
    width, height = image.size
    st.info(f"Ukuran gambar asli: {width} x {height} pixels")
    
    # Input coordinates
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Koordinat Kiri Atas:**")
        left = st.number_input("X (kiri)", min_value=0, max_value=width-1, value=0, key="crop_left")
        top = st.number_input("Y (atas)", min_value=0, max_value=height-1, value=0, key="crop_top")
    
    with col2:
        st.markdown("**Koordinat Kanan Bawah:**")
        right = st.number_input("X (kanan)", min_value=left+1, max_value=width, value=width, key="crop_right")
        bottom = st.number_input("Y (bawah)", min_value=top+1, max_value=height, value=height, key="crop_bottom")
    
    # Preview crop area
    if st.button("👁️ Preview Crop Area", key="preview_crop"):
        # Create a copy with crop area highlighted
        preview_img = image.copy()
        draw = ImageDraw.Draw(preview_img)
        draw.rectangle([left, top, right, bottom], outline="red", width=3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(preview_img, caption="Preview dengan area crop (kotak merah)", use_container_width=True)
        with col2:
            # Show crop preview
            cropped_preview = image.crop((left, top, right, bottom))
            st.image(cropped_preview, caption=f"Hasil crop ({right-left}x{bottom-top})", use_container_width=True)
    
    # Perform crop
    if st.button("✂️ Crop Gambar", type="primary", key="perform_crop"):
        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image
    
    return None

def crop_image_interactive(image, key_suffix=""):
    """
    Interactive cropping menggunakan streamlit-cropper
    """
    if not CROPPER_AVAILABLE:
        st.error("⚠️ streamlit-cropper tidak tersedia.")
        st.code("pip install streamlit-cropper")
        st.info("Aplikasi memerlukan streamlit-cropper untuk fitur crop.")
        return None
    
    st.markdown("### ✂️ Crop Gambar")
    st.info("🎯 Drag untuk memilih area yang ingin diproses, kemudian klik tombol di bawah")
    
    # Cropping interface
    cropped_img = st_cropper(
        image, 
        realtime_update=True, 
        box_color='#FF0004',
        aspect_ratio=None,
        return_type='image',
        key=f"cropper_{key_suffix}"
    )
    
    if cropped_img:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="📷 Gambar Asli", use_container_width=True)
        with col2:
            st.image(cropped_img, caption="✂️ Hasil Crop", use_container_width=True)
        
        # Show crop dimensions
        crop_width, crop_height = cropped_img.size
        st.info(f"📐 Ukuran area crop: {crop_width} x {crop_height} pixels")
        
        return cropped_img
    
    return None

# ===============================
# FUNCTIONS UNTUK LOADING DATA
# ===============================

@st.cache_data
def load_character_images():
    """
    Load gambar karakter dari dataset dengan path yang benar
    """
    char_images = {}
    
    if not os.path.exists(DATASET_PATH):
        st.error(f"Dataset path tidak ditemukan: {DATASET_PATH}")
        return char_images
    
    try:
        for folder_name in os.listdir(DATASET_PATH):
            folder_path = os.path.join(DATASET_PATH, folder_name)
            
            if os.path.isdir(folder_path):
                # Cari file gambar dalam folder
                image_files = glob.glob(os.path.join(folder_path, '*'))
                image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if image_files:
                    try:
                        img = Image.open(image_files[0])
                        char_images[folder_name] = img
                    except Exception as e:
                        st.warning(f"Error loading image for {folder_name}: {e}")
        
        return char_images
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return {}

@st.cache_resource
def load_classification_model():
    """
    Load model klasifikasi dengan path yang benar
    """
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model tidak ditemukan di: {MODEL_PATH}")
        return None
    
    try:
        model = load_model(MODEL_PATH)
        st.success(f"Model berhasil dimuat dari: {MODEL_PATH}")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# ===============================
# FUNCTIONS TRANSLITERASI
# ===============================

def transliterate_to_bima(text):
    """
    Transliterasi teks Latin ke aksara Bima dengan prinsip longest match first
    """
    # Urutkan karakter berdasarkan panjang (terpanjang dulu)
    sorted_chars = sorted(BIMA_CHARACTERS, key=len, reverse=True)
    
    text = text.upper().strip()
    result = []
    i = 0
    
    while i < len(text):
        # Skip spasi
        if text[i] == ' ':
            result.append(' ')
            i += 1
            continue
            
        # Cari karakter terpanjang yang cocok
        matched = False
        for char in sorted_chars:
            if text[i:i+len(char)] == char:
                result.append(char)
                i += len(char)
                matched = True
                break
        
        if not matched:
            # Jika tidak ada yang cocok, tambahkan sebagai karakter tidak dikenal
            result.append(f"[{text[i]}]")
            i += 1
    
    return result

def create_character_image(char, char_images, width=80, height=80):
    """
    Membuat gambar karakter aksara Bima dari dataset atau placeholder
    """
    # Coba gunakan gambar dari dataset
    if char in char_images:
        img = char_images[char].copy()
        
        # Resize gambar ke ukuran yang diinginkan
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        
        # Konversi ke RGB jika diperlukan
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        return img
    
    # Jika tidak ada gambar, buat placeholder
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Gambar border
    draw.rectangle([2, 2, width-3, height-3], outline='black', width=2)
    
    # Tambahkan teks karakter (placeholder)
    try:
        font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        draw.text((x, y), char, fill='red', font=font)
    except:
        draw.text((10, height//2-10), char, fill='red')
    
    return img

def create_full_text_image(result, char_images, char_spacing=5, word_spacing=20):
    """
    Membuat gambar untuk seluruh teks yang telah ditransliterasi dalam satu gambar panjang
    """
    all_images = []
    
    for i, char in enumerate(result):
        if char == ' ':
            # Untuk spasi antar kata, buat gambar kosong
            if all_images:  # Jika bukan spasi di awal
                space_img = Image.new('RGB', (word_spacing, 80), color='white')
                all_images.append(space_img)
        else:
            # Tambahkan gambar karakter (skip karakter yang tidak dikenal)
            if char not in ['[', ']'] and not char.startswith('['):
                img = create_character_image(char, char_images)
                all_images.append(img)
    
    if not all_images:
        return None
    
    # Gabungkan semua gambar dengan spacing karakter
    total_width = sum(img.width for img in all_images)
    if len(all_images) > 1:
        # Tambahkan spacing antar karakter (tidak antar kata karena sudah dihandle di atas)
        char_spacings = 0
        for i, img in enumerate(all_images):
            if i > 0:
                # Cek apakah gambar sebelumnya adalah spasi kata
                prev_img = all_images[i-1]
                if prev_img.width != word_spacing:  # Bukan spasi kata
                    char_spacings += char_spacing
        total_width += char_spacings
    
    max_height = max(img.height for img in all_images)
    
    # Buat gambar gabungan
    combined = Image.new('RGB', (total_width, max_height), color='white')
    
    # Paste setiap gambar
    x_offset = 0
    for i, img in enumerate(all_images):
        combined.paste(img, (x_offset, 0))
        x_offset += img.width
        
        # Tambahkan spacing antar karakter (bukan antar kata)
        if i < len(all_images) - 1:
            next_img = all_images[i+1]
            if img.width != word_spacing and next_img.width != word_spacing:
                x_offset += char_spacing
    
    return combined

def combine_images(images, spacing=5):
    """
    Menggabungkan list gambar menjadi satu gambar horizontal
    """
    if not images:
        return None
    
    # Hitung total lebar dan tinggi maksimum
    total_width = sum(img.width for img in images) + spacing * (len(images) - 1)
    max_height = max(img.height for img in images)
    
    # Buat gambar gabungan
    combined = Image.new('RGB', (total_width, max_height), color='white')
    
    # Paste setiap gambar
    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width + spacing
    
    return combined

# ===============================
# FUNCTIONS KLASIFIKASI
# ===============================

def preprocess_canvas_for_classification(canvas_data):
    """
    Preprocess data canvas untuk klasifikasi
    Canvas dari st_canvas memiliki format RGBA, perlu dikonversi
    """
    try:
        # Convert RGBA to RGB (remove alpha channel)
        if canvas_data.shape[-1] == 4:  # RGBA
            # Buat background putih
            rgb_canvas = np.ones((canvas_data.shape[0], canvas_data.shape[1], 3), dtype=np.uint8) * 255
            
            # Composite RGBA over white background
            alpha = canvas_data[:, :, 3:4] / 255.0
            rgb_canvas = (1 - alpha) * rgb_canvas + alpha * canvas_data[:, :, :3]
            canvas_data = rgb_canvas.astype(np.uint8)
        
        # Convert to grayscale
        if len(canvas_data.shape) == 3:
            gray = cv2.cvtColor(canvas_data, cv2.COLOR_RGB2GRAY)
        else:
            gray = canvas_data
        
        # Resize ke 224x224 (sesuai model)
        resized = cv2.resize(gray, (224, 224))
        
        # Normalisasi
        normalized = resized.astype('float32') / 255.0
        
        # Reshape untuk model (batch_size, height, width, channels)
        # Model kemungkinan mengharapkan RGB 3 channel
        if len(normalized.shape) == 2:
            # Convert grayscale to RGB
            rgb_image = cv2.cvtColor(normalized, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = normalized
        
        # Add batch dimension
        input_image = np.expand_dims(rgb_image, axis=0)
        
        return input_image
        
    except Exception as e:
        st.error(f"Error preprocessing canvas: {e}")
        return None

def preprocess_uploaded_image_for_classification(image):
    """
    Preprocess uploaded image untuk klasifikasi
    """
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            img_array = np.array(image)
        
        # Resize ke 224x224
        resized = cv2.resize(img_array, (224, 224))
        
        # Normalisasi
        normalized = resized.astype('float32') / 255.0
        
        # Add batch dimension
        input_image = np.expand_dims(normalized, axis=0)
        
        return input_image
        
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None

def classify_character(image_input, model):
    """
    Klasifikasi karakter aksara Bima (hanya 22 karakter)
    """
    if model is None:
        return None, 0.0
    
    try:
        # Prediksi
        predictions = model.predict(image_input, verbose=0)
        
        # Get hasil prediksi
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Map ke nama karakter menggunakan CLASSIFICATION_CHARACTERS
        if predicted_class < len(CLASSIFICATION_CHARACTERS):
            character = CLASSIFICATION_CHARACTERS[predicted_class]
            return character, confidence
        else:
            return "Unknown", confidence
            
    except Exception as e:
        st.error(f"Error in classification: {e}")
        return None, 0.0

# ===============================
# FUNCTIONS OCR
# ===============================

def ocr_bima_to_latin(image_path, tesseract_config):
    """
    OCR Aksara Bima ke Latin menggunakan konfigurasi yang sudah bekerja
    """
    try:
        # Baca gambar
        image = cv2.imread(image_path)
        
        # Preprocess untuk OCR
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold untuk meningkatkan kontras
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # OCR dengan Tesseract menggunakan konfigurasi yang sudah bekerja
        text = pytesseract.image_to_string(thresh, lang='aksaralengkap', config='--psm 8')
        
        return text.strip()
        
    except Exception as e:
        st.error(f"Error in OCR: {e}")
        return ""

# ===============================
# UTILITY FUNCTIONS
# ===============================

def image_to_base64(img):
    """
    Konversi gambar PIL ke base64 untuk download
    """
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def check_system_requirements():
    """
    Check apakah semua requirement tersedia
    """
    requirements = {
        'dataset': os.path.exists(DATASET_PATH),
        'model': os.path.exists(MODEL_PATH),
        'tesseract': os.path.exists(TESSERACT_CMD)
    }
    
    return requirements

# ===============================
# SIDEBAR NAVIGATION
# ===============================

def sidebar_navigation():
    """
    Sidebar dengan tombol navigasi yang lebih menarik
    """
    st.sidebar.title("🔤 Sistem Aksara Bima")
    st.sidebar.markdown("---")
    
    # Header navigasi
    st.sidebar.markdown("### 🎯 Fitur Utama")
    
    # Tombol navigasi dengan style yang menarik
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("🏠 Beranda", key="home_btn", use_container_width=True):
            st.session_state.current_page = "home"
        
        if st.button("🔤 Transliterasi", key="trans_btn", use_container_width=True):
            st.session_state.current_page = "transliterasi"
    
    with col2:
        if st.button("🎯 Klasifikasi", key="class_btn", use_container_width=True):
            st.session_state.current_page = "klasifikasi"
        
        if st.button("📖 OCR", key="ocr_btn", use_container_width=True):
            st.session_state.current_page = "ocr"
    
    # Tombol informasi sistem
    st.sidebar.markdown("---")
    if st.sidebar.button("ℹ️ Informasi Sistem", key="info_btn", use_container_width=True):
        st.session_state.current_page = "info"
    
    # Status sistem di sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status Sistem")
    
    # Load resources for status
    char_images = load_character_images()
    classification_model = load_classification_model()
    tesseract_config = setup_tesseract()
    
    # Status indicators
    if len(char_images) > 0:
        st.sidebar.success(f"✅ Dataset: {len(char_images)} karakter")
    else:
        st.sidebar.error("❌ Dataset tidak tersedia")
    
    if classification_model:
        st.sidebar.success("✅ Model: Siap digunakan")
    else:
        st.sidebar.error("❌ Model tidak tersedia")
    
    if tesseract_config:
        st.sidebar.success("✅ OCR: Siap digunakan")
    else:
        st.sidebar.error("❌ OCR tidak tersedia")

# ===============================
# HALAMAN APLIKASI
# ===============================

def home_page():
    """
    Halaman beranda yang menarik
    """
    st.title(" Sistem Aksara Bima Lengkap")
    st.markdown("---")
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## 🎯 Selamat Datang di Sistem Aksara Bima!
        
        Aplikasi terintegrasi yang menyediakan tiga fitur utama untuk membantu Anda
        bekerja dengan aksara Bima:
        
        - **🔤 Transliterasi**: Ubah teks Latin menjadi aksara Bima
        - **🎯 Klasifikasi**: Identifikasi karakter aksara Bima dari gambar
        - **📖 OCR**: Ekstrak teks dari gambar aksara Bima
        """)
    
    with col2:
        st.info("""
        **💡 Tips Penggunaan:**
        - Gunakan tombol di sidebar untuk navigasi
        - Periksa status sistem di sidebar
        - Setiap fitur memiliki panduan lengkap
        """)
    
    st.markdown("---")
    
    # Fitur cards
    st.markdown("## 🚀 Fitur Utama")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🔤 Transliterasi
        Konversi teks Latin ke aksara Bima dengan:
        - Algoritma longest match first
        - Output gambar berkualitas tinggi
        - Pengaturan spacing yang fleksibel
        """)
        if st.button("🔄 Mulai Transliterasi", key="start_trans"):
            st.session_state.current_page = "transliterasi"
    
    with col2:
        st.markdown("""
        ### 🎯 Klasifikasi
        Identifikasi karakter aksara Bima:
        - 22 karakter yang didukung
        - Input canvas drawing atau upload
        - Confidence score dan top prediksi
        """)
        if st.button("🔍 Mulai Klasifikasi", key="start_class"):
            st.session_state.current_page = "klasifikasi"
    
    with col3:
        st.markdown("""
        ### 📖 OCR
        Ekstrak teks dari gambar:
        - Teknologi Tesseract OCR
        - Khusus untuk aksara Bima
        - Output teks Latin
        """)
        if st.button("📷 Mulai OCR", key="start_ocr"):
            st.session_state.current_page = "ocr"
    
    st.markdown("---")
    
    # Statistik dan informasi
    st.markdown("## 📊 Statistik Sistem")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Karakter Transliterasi", len(BIMA_CHARACTERS))
    
    with col2:
        st.metric("Karakter Klasifikasi", len(CLASSIFICATION_CHARACTERS))
    
    with col3:
        char_images = load_character_images()
        st.metric("Dataset Tersedia", len(char_images))
    
    with col4:
        requirements = check_system_requirements()
        ready_count = sum(requirements.values())
        st.metric("Sistem Siap", f"{ready_count}/3")

def transliteration_page():
    """
    Halaman transliterasi Latin ke Aksara Bima
    """
    st.header("🔤 Transliterasi Latin → Aksara Bima")
    st.markdown("---")
    
    # Asumsi fungsi ini memuat gambar dari path yang sudah ditentukan
    # atau dari cache jika sudah pernah dimuat.
    char_images = load_character_images() 
    
    st.markdown("""
    Aplikasi ini mengubah teks Latin menjadi aksara Bima. 
    Algoritma menggunakan prinsip **longest match first** untuk hasil yang optimal.
    """)
    
    # Input teks
    col1, col2 = st.columns([2, 1])
    
    with col1:
        input_text = st.text_area(
            "Masukkan teks Latin:",
            placeholder="Contoh: NGAHA, MPALA, BIMA SAKTI",
            height=100
        )
    
    with col2:
        st.markdown("### 💡 Contoh Input:")
        st.code("""
NGAHA 
COU NGARA NGGOMI
NGARA NAHU JAFIR
        """)
    
    if st.button("🔄 Transliterasi", type="primary"):
        if input_text.strip():
            # Proses transliterasi
            result = transliterate_to_bima(input_text)
            
            # --- TAMPILAN HASIL ---
            st.markdown("---")
            st.markdown("### 📝 Hasil Transliterasi")
            
            # 1. Tampilkan pemecahan karakter keseluruhan
            result_text = " + ".join([char if char != ' ' else '|' for char in result]).replace(' | ', '   ').replace('|+|', ' | ')
            st.markdown("**Pemecahan Karakter:**")
            st.code(result_text)

            # 2. Pisahkan hasil menjadi kata-kata
            words = []
            current_word = []
            for char in result:
                if char == ' ':
                    if current_word:
                        words.append(current_word)
                        current_word = []
                else:
                    current_word.append(char)
            if current_word:
                words.append(current_word)

            # 3. Tampilkan visualisasi per kata (FITUR BARU)
            st.markdown("### 🖼️ Visualisasi Per Kata")
            if not words:
                st.warning("Tidak ada kata yang bisa divisualisasikan.")
            else:
                for i, word in enumerate(words):
                    st.markdown(f"**Kata {i+1}:** `{' + '.join(word)}`")
                    
                    images = []
                    for char in word:
                        # Pastikan karakter tidak dikenal tidak diproses
                        if not char.startswith('['):
                            img = create_character_image(char, char_images)
                            images.append(img)
                    
                    if images:
                        # Gabungkan gambar untuk kata ini
                        # Pastikan Anda memiliki fungsi combine_images dari skrip pertama
                        word_img = combine_images(images, spacing=5) 
                        if word_img:
                            st.image(word_img, caption=f"Visualisasi untuk kata: {''.join(word)}")

            st.markdown("---")

            # 4. Tampilkan gambar gabungan seluruh teks
            st.markdown("### 🖼️ Hasil Lengkap (Satu Gambar)")
            st.markdown("*Seluruh teks digabungkan menjadi satu gambar panjang.*")
            
            # Buat gambar untuk seluruh kalimat (menggunakan spacing default)
            full_image = create_full_text_image(result, char_images)
            
            if full_image:
                # Tampilkan gambar
                st.image(full_image, caption=f"Transliterasi lengkap untuk: {input_text}", use_column_width=True)
                
                st.markdown("---")

                # Opsi download dan metrik (tanpa pengaturan spacing)
                col_dl, col_metrics = st.columns(2)
                
                with col_dl:
                    # Button download
                    img_base64 = image_to_base64(full_image)
                    st.download_button(
                        label="📥 Download Gambar",
                        data=base64.b64decode(img_base64),
                        file_name=f"transliterasi_bima_{input_text[:20].replace(' ', '_')}.png",
                        mime="image/png"
                    )
                
                with col_metrics:
                    # Metrik gambar
                    st.metric("Resolusi Gambar", f"{full_image.width} x {full_image.height} px")

            else:
                st.error("Gagal membuat gambar. Periksa input teks dan pastikan dataset termuat dengan benar.")

def classification_page():
    """
    Halaman klasifikasi karakter aksara Bima
    """
    st.header("🎯 Klasifikasi Karakter Aksara Bima")
    st.markdown("---")
    
    # Load resources
    model = load_classification_model()
    char_images = load_character_images()
    
    if model is None:
        st.error("Model klasifikasi tidak tersedia. Pastikan file model ada di path yang benar.")
        st.code(f"Expected path: {MODEL_PATH}")
        return
    
    # Info tentang karakter yang didukung
    st.info(f"Model ini dapat mengklasifikasikan {len(CLASSIFICATION_CHARACTERS)} karakter aksara Bima.")
    
    with st.expander("📋 Lihat Daftar Karakter yang Didukung"):
        col1, col2, col3 = st.columns(3)
        
        # Bagi karakter menjadi 3 kolom
        chars_per_col = len(CLASSIFICATION_CHARACTERS) // 3
        
        with col1:
            st.write("**Kelompok 1:**")
            for char in CLASSIFICATION_CHARACTERS[:chars_per_col]:
                st.write(f"• {char}")
        
        with col2:
            st.write("**Kelompok 2:**")
            for char in CLASSIFICATION_CHARACTERS[chars_per_col:2*chars_per_col]:
                st.write(f"• {char}")
        
        with col3:
            st.write("**Kelompok 3:**")
            for char in CLASSIFICATION_CHARACTERS[2*chars_per_col:]:
                st.write(f"• {char}")
    
    # Pilihan input method
    input_method = st.radio(
        "Pilih metode input:",
        ["🎨 Canvas Drawing", "📁 Upload Gambar"]
    )
    
    processed_image = None
    original_display = None
    
    if input_method == "🎨 Canvas Drawing":
        st.markdown("### 🎨 Gambar Karakter Aksara Bima")
        st.info("Gambar akan diubah ke ukuran 224x224 pixel untuk model klasifikasi")
        
        try:
            # Canvas untuk menggambar
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",  # Transparent fill
                stroke_width=10,  # Stroke lebih tebal untuk clarity
                stroke_color="black",
                background_color="white",
                width=400,
                height=400,
                drawing_mode="freedraw",
                key="canvas"
            )
            
            if canvas_result.image_data is not None:
                # Check if anything is drawn
                if np.any(canvas_result.image_data[:, :, 3] > 0):  # Check alpha channel
                    # Preprocess untuk model
                    processed_image = preprocess_canvas_for_classification(canvas_result.image_data)
                    original_display = canvas_result.image_data
                    
                    # Show preview
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(canvas_result.image_data, caption="Gambar Asli (400x400)", width=200)
                    with col2:
                        # Show resized version
                        if processed_image is not None:
                            display_resized = (processed_image[0] * 255).astype(np.uint8)
                            st.image(display_resized, caption="Diproses untuk Model (224x224)", width=200)
                else:
                    st.info("Silakan gambar karakter di canvas")
                    
        except Exception as e:
            st.error(f"Error dengan canvas: {e}")
            st.info("Silakan gunakan metode 'Upload Gambar' sebagai alternatif")
    
    elif input_method == "📁 Upload Gambar":
        st.markdown("### 📁 Upload Gambar Karakter")
        
        uploaded_file = st.file_uploader(
            "Upload gambar karakter aksara Bima:",
            type=['png', 'jpg', 'jpeg'],
            help="Upload gambar untuk klasifikasi"
        )
        
        if uploaded_file is not None:
            # Load gambar original
            original_image = Image.open(uploaded_file)
            
            # Display gambar original
            st.image(original_image, caption="📷 Gambar Original", use_container_width=True)
            
            # Opsi crop (tidak wajib)
            col1, col2 = st.columns([3, 1])
            
            with col2:
                st.markdown("#### ⚙️ Opsi")
                use_crop = st.checkbox("✂️ Crop gambar", help="Centang jika ingin crop area tertentu")
            
            # Initialize gambar yang akan diproses
            final_image = original_image
            
            # Jika user memilih crop
            if use_crop:
                with col1:
                    st.markdown("---")
                    cropped_image = crop_image_interactive(original_image, "classification")
                    if cropped_image:
                        final_image = cropped_image
                        st.success("✅ Menggunakan gambar crop untuk klasifikasi")
            
            # Process final image untuk klasifikasi
            if final_image:
                original_display = np.array(final_image)
                processed_image = preprocess_uploaded_image_for_classification(final_image)
                
    
    # Classification
    if processed_image is not None and st.button("🔍 Klasifikasi", type="primary"):
        with st.spinner("Memproses klasifikasi..."):
            character, confidence = classify_character(processed_image, model)
            
            if character:
                st.success(f"Prediksi: **{character}** (Confidence: {confidence:.2%})")
                
                # Tampilkan hasil
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Gambar Input:**")
                    if original_display is not None:
                        st.image(original_display, width=200)
                
                with col2:
                    st.markdown("**Karakter Referensi dari Dataset:**")
                    if character in char_images:
                        st.image(char_images[character], width=200)
                    else:
                        st.warning("Gambar referensi tidak tersedia dalam dataset")
                
                # Confidence bar
                st.markdown("### 📊 Confidence Score")
                st.progress(confidence)
                
                if confidence < 0.5:
                    st.warning("⚠️ Confidence rendah. Hasil mungkin tidak akurat.")
                elif confidence < 0.8:
                    st.info("ℹ️ Confidence sedang. Hasil cukup akurat.")
                else:
                    st.success("✅ Confidence tinggi. Hasil sangat akurat!")
                    
                # Tambahan: Top 3 prediksi
                if hasattr(model, 'predict'):
                    try:
                        all_predictions = model.predict(processed_image, verbose=0)[0]
                        top_3_indices = np.argsort(all_predictions)[-3:][::-1]
                        
                        st.markdown("### 📋 Top 3 Prediksi:")
                        for i, idx in enumerate(top_3_indices):
                            if idx < len(CLASSIFICATION_CHARACTERS):
                                char_name = CLASSIFICATION_CHARACTERS[idx]
                                conf = all_predictions[idx]
                                st.write(f"{i+1}. **{char_name}** - {conf:.2%}")
                    except:
                        pass
                        
            else:
                st.error("Klasifikasi gagal. Silakan coba lagi dengan gambar yang lebih jelas.")

def ocr_page():
    """
    Halaman OCR Aksara Bima ke Latin
    """
    st.header("📖 OCR Aksara Bima → Latin")
    st.markdown("---")
    
    tesseract_config = setup_tesseract()
    
    if tesseract_config is None:
        st.error("Tesseract tidak tersedia. Pastikan path sudah benar.")
        st.code(f"Expected path: {TESSERACT_CMD}")
        return
    
    st.markdown("""
    Upload gambar yang berisi teks aksara Bima, dan sistem akan mengkonversinya ke teks Latin.
    Anda dapat langsung memproses atau crop area tertentu untuk hasil yang lebih akurat.
    """)
    
    uploaded_file = st.file_uploader(
        "Upload gambar aksara Bima:",
        type=['png', 'jpg', 'jpeg'],
        help="Upload gambar untuk OCR"
    )
    
    if uploaded_file is not None:
        # Load original image
        original_image = Image.open(uploaded_file)
        
        # Display gambar original
        st.image(original_image, caption="📷 Gambar Original", use_container_width=True)
        
        # Opsi crop (tidak wajib)
        col1, col2 = st.columns([3, 1])
        
        with col2:
            st.markdown("### ⚙️ Opsi")
            use_crop = st.checkbox("✂️ Crop area teks", help="Centang untuk crop area teks spesifik")
        
        # Initialize gambar yang akan diproses
        final_image = original_image
        
        # Jika user memilih crop
        if use_crop:
            with col1:
                st.markdown("---")
                cropped_image = crop_image_interactive(original_image, "ocr")
                if cropped_image:
                    final_image = cropped_image
                    st.success("✅ Menggunakan area crop untuk OCR")
        
        # OCR processing dengan gambar final
        if final_image:
            st.markdown("---")
            
            # Info tentang gambar yang akan diproses
            if use_crop and final_image != original_image:
                st.info("🎯 OCR akan dijalankan pada area yang di-crop")
            else:
                st.info("📷 OCR akan dijalankan pada gambar lengkap")
            
            if st.button("🔍 Proses OCR", type="primary", use_container_width=True):
                with st.spinner("🔄 Memproses OCR..."):
                    # Save final image temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
                        final_image.save(tmp_file.name)
                        tmp_path = tmp_file.name
                    
                    result_text = ocr_bima_to_latin(tmp_path, tesseract_config)
                    
                    if result_text:
                        st.success("✅ OCR berhasil!")
                        
                        # Display hasil OCR
                        st.markdown("### 📝 Hasil OCR:")
                        result_container = st.container()
                        with result_container:
                            st.text_area(
                                "Teks Latin yang berhasil dibaca:", 
                                result_text, 
                                height=150,
                                help="Hasil OCR dari gambar yang diproses"
                            )
                        
                        st.markdown("---")
                        
                        # Download dan actions
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Download teks
                            st.download_button(
                                label="📥 Download Hasil OCR",
                                data=result_text,
                                file_name=f"ocr_result_{len(result_text)}_chars.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
                        
                        with col2:
                            # Download gambar yang diproses
                            img_buffer = io.BytesIO()
                            final_image.save(img_buffer, format="PNG")
                            img_data = img_buffer.getvalue()
                            
                            download_name = "cropped_image_ocr.png" if (use_crop and final_image != original_image) else "processed_image_ocr.png"
                            
                            st.download_button(
                                label="📷 Download Gambar Proses",
                                data=img_data,
                                file_name=download_name,
                                mime="image/png",
                                use_container_width=True
                            )
                        
                    else:
                        st.error("❌ OCR gagal mengenali teks.")
                        
                        # Enhanced tips untuk perbaikan
                        with st.expander("💡 Tips untuk Meningkatkan Hasil OCR"):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **🎯 Tips Cropping:**
                                - Coba **centang opsi crop** untuk fokus pada area teks
                                - Crop **hanya area teks** yang ingin dibaca
                                - Hindari background yang rumit
                                - Sisakan sedikit margin di sekitar teks
                                """)
                            
                            with col2:
                                st.markdown("""
                                **📸 Tips Gambar:**
                                - Kontras tinggi antara teks dan background
                                - Resolusi minimal 300 DPI
                                - Teks harus horizontal (tidak miring)
                                - Pencahayaan merata tanpa bayangan
                                """)
                        
                        # Show processed image for reference
                        st.markdown("### 🔍 Gambar yang Diproses:")
                        st.image(final_image, caption="Gambar yang diproses OCR", use_container_width=True)
                    
                    # Clean up
                    try:
                        os.unlink(tmp_path)
                    except:
                        pass

def info_page():
    """
    Halaman informasi sistem
    """
    st.header("ℹ️ Informasi Sistem")
    st.markdown("---")
    
    # Check system requirements
    requirements = check_system_requirements()
    
    st.markdown("### 🔧 Status Sistem")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if requirements['dataset']:
            st.success("✅ Dataset tersedia")
            st.code(f"Path: {DATASET_PATH}")
        else:
            st.error("❌ Dataset tidak ditemukan")
            st.code(f"Expected: {DATASET_PATH}")
    
    with col2:
        if requirements['model']:
            st.success("✅ Model tersedia")
            st.code(f"Path: {MODEL_PATH}")
        else:
            st.error("❌ Model tidak ditemukan")
            st.code(f"Expected: {MODEL_PATH}")
    
    with col3:
        if requirements['tesseract']:
            st.success("✅ Tesseract tersedia")
            st.code(f"Path: {TESSERACT_CMD}")
        else:
            st.error("❌ Tesseract tidak ditemukan")
            st.code(f"Expected: {TESSERACT_CMD}")
    
    st.markdown("---")
    
    st.markdown("""
    ## 🎯 Fitur Aplikasi
    
    ### 1. 🔤 Transliterasi Latin → Aksara Bima
    - Konversi teks Latin ke representasi visual aksara Bima
    - Menggunakan algoritma longest match first
    - Output berupa gambar yang dapat diunduh
    
    ### 2. 🎯 Klasifikasi Karakter Aksara Bima
    - Identifikasi karakter aksara Bima dari gambar
    - Input: Canvas drawing atau upload gambar
    - **✂️ Optional Crop**: Upload → [Crop jika perlu] → Klasifikasi
    - Model: Deep Learning (.h5) dengan input 224x224 pixel
    - Menampilkan confidence score dan top 3 prediksi
    - **Karakter yang didukung: 22 karakter**
    
    ### 3. 📖 OCR Aksara Bima → Latin
    - Ekstraksi teks dari gambar aksara Bima
    - **✂️ Optional Crop**: Upload → [Crop jika perlu] → OCR
    - Konversi ke teks Latin
    - Menggunakan Tesseract OCR dengan aksarareal.traineddata
    - Statistik hasil OCR yang detail
    - Download hasil teks dan gambar yang diproses
    
    ### 🚀 Workflow Fleksibel:
    ```
    📁 Upload Gambar
    ↓
    ⚙️ Pilihan: [✓] Crop atau [✗] Langsung Proses
    ↓
    🔍 Proses (Klasifikasi/OCR)
    ↓
    📊 Hasil & Download
    ```
    
    **Default**: Langsung proses setelah upload   
    **Optional**: Crop area tertentu untuk hasil lebih akurat
    
    ## 📦 Dependencies
    """)
    
    # Check cropper availability
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Core Libraries:**")
        st.code("""
streamlit
streamlit-drawable-canvas
tensorflow
opencv-python
pytesseract
pillow
numpy
        """)
    
    with col2:
        st.markdown("**Optional Crop Feature:**")
        if CROPPER_AVAILABLE:
            st.success("✅ streamlit-cropper: Tersedia")
            st.info("🎯 Fitur crop interaktif tersedia")
        else:
            st.warning("⚠️ streamlit-cropper: Tidak tersedia")
            st.info("📷 Hanya mode langsung proses yang tersedia")
        
        st.code("pip install streamlit-cropper")
    
    if not CROPPER_AVAILABLE:
        st.info("""
        💡 **Tanpa streamlit-cropper**: 
        Aplikasi tetap berfungsi normal. Fitur crop tidak akan tersedia, 
        tetapi klasifikasi dan OCR dapat menggunakan gambar lengkap.
        """)
    
    st.markdown("""
    ## 📊 Informasi Dataset
    """)
    
    # Statistik dataset
    char_images = load_character_images()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Karakter Transliterasi", len(BIMA_CHARACTERS))
    
    with col2:
        st.metric("Karakter Klasifikasi Model", len(CLASSIFICATION_CHARACTERS))
    
    with col3:
        st.metric("Gambar Dataset Tersedia", len(char_images))
    
    with col4:
        missing = len(BIMA_CHARACTERS) - len(char_images)
        st.metric("Belum Tersedia", missing)
    
    # Daftar karakter
    with st.expander("📚 Daftar Lengkap Karakter untuk Transliterasi"):
        # Kategorisasi karakter
        vowels = ['A', 'E', 'I', 'O', 'U']
        consonants = ['B', 'C', 'D', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'R', 'S', 'T', 'W', 'Y', 'NG']
        cv_chars = [char for char in BIMA_CHARACTERS if char not in vowels + consonants]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Vokal (5):**")
            st.write(", ".join(vowels))
        
        with col2:
            st.markdown("**Konsonan Mati (18):**")
            st.write(", ".join(consonants))
        
        with col3:
            st.markdown(f"**Konsonan + Vokal ({len(cv_chars)}):**")
            # Tampilkan dalam chunks
            chunks = [cv_chars[i:i+10] for i in range(0, len(cv_chars), 10)]
            for chunk in chunks:
                st.write(", ".join(chunk))
    
    with st.expander("🎯 Daftar Karakter yang Didukung Model Klasifikasi"):
        st.markdown(f"**Model dapat mengklasifikasikan {len(CLASSIFICATION_CHARACTERS)} karakter:**")
        
        # Tampilkan dalam 3 kolom
        col1, col2, col3 = st.columns(3)
        chars_per_col = len(CLASSIFICATION_CHARACTERS) // 3
        
        with col1:
            st.write("**Kelompok 1:**")
            for char in CLASSIFICATION_CHARACTERS[:chars_per_col]:
                st.write(f"• {char}")
        
        with col2:
            st.write("**Kelompok 2:**")
            for char in CLASSIFICATION_CHARACTERS[chars_per_col:2*chars_per_col]:
                st.write(f"• {char}")
        
        with col3:
            st.write("**Kelompok 3:**")
            for char in CLASSIFICATION_CHARACTERS[2*chars_per_col:]:
                st.write(f"• {char}")
            
        st.info("💡 Model klasifikasi hanya dilatih untuk 22 karakter di atas. Untuk karakter lain, gunakan fitur transliterasi.")

# ===============================
# MAIN APP
# ===============================

def main():
    # Custom CSS untuk styling yang lebih menarik
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
    
    # Setup sidebar navigation
    sidebar_navigation()
    
    # Router untuk halaman berdasarkan session state
    if st.session_state.current_page == "home":
        home_page()
    elif st.session_state.current_page == "transliterasi":
        transliteration_page()
    elif st.session_state.current_page == "klasifikasi":
        classification_page()
    elif st.session_state.current_page == "ocr":
        ocr_page()
    elif st.session_state.current_page == "info":
        info_page()
    else:
        # Default ke home jika ada error
        st.session_state.current_page = "home"
        home_page()

if __name__ == "__main__":
    main()
