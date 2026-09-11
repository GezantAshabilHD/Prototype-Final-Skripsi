
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image


# ============================================================
# KONFIGURASI APLIKASI
# ============================================================

IMG_SIZE = 224
DECISION_THRESHOLD = 0.5

CLASS_NAMES = {
    0: "Real Image",
    1: "AI-Generated Image"
}

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "mobilenetv2_real_vs_ai_final.keras"
)


# ============================================================
# LOAD FINAL MODEL
# Fungsi:
# Memuat model akhir hasil penelitian satu kali dan
# menyimpannya pada cache Streamlit.
# ============================================================

@st.cache_resource
def load_final_model():

    return tf.keras.models.load_model(
        MODEL_PATH
    )


model = load_final_model()


# ============================================================
# IMAGE PREPROCESSING
# Fungsi:
# Menerapkan preprocessing yang sama dengan notebook:
#
# 1. Decode citra menjadi RGB 3 channel
# 2. Konversi ke float32
# 3. Resize with Padding 224 x 224
# 4. MobileNetV2 preprocess_input
# 5. Menambahkan batch dimension
# ============================================================

def preprocess_image(image_bytes):

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [None, None, 3]
    )

    image = tf.cast(
        image,
        tf.float32
    )

    image = tf.image.resize_with_pad(
        image,
        IMG_SIZE,
        IMG_SIZE
    )

    image = (
        tf.keras.applications
        .mobilenet_v2
        .preprocess_input(image)
    )

    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# MODEL PREDICTION
# Fungsi:
# Menghasilkan probabilitas AI kemudian menentukan kelas
# menggunakan threshold tetap 0.5.
# ============================================================

def predict_image(image_bytes):

    processed_image = preprocess_image(
        image_bytes
    )

    probability_ai = float(
        model(
            processed_image,
            training=False
        )
        .numpy()
        .reshape(-1)[0]
    )

    predicted_label = int(
        probability_ai
        >= DECISION_THRESHOLD
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_label
        ]
    )

    if predicted_label == 1:

        confidence = probability_ai

    else:

        confidence = (
            1.0
            - probability_ai
        )

    return (
        predicted_class,
        probability_ai,
        confidence
    )


# ============================================================
# WEB INTERFACE
# Fungsi:
# Menampilkan halaman prototype dan menyediakan fasilitas
# upload citra untuk proses klasifikasi.
# ============================================================

st.set_page_config(
    page_title=(
        "Klasifikasi Real Image "
        "dan AI-Generated Image"
    ),
    layout="centered"
)


st.title(
    "Klasifikasi Real Image "
    "dan AI-Generated Image"
)

st.write(
    "Prototype klasifikasi citra "
    "menggunakan MobileNetV2."
)


uploaded_file = st.file_uploader(
    "Unggah citra",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# MENAMPILKAN HASIL PREDIKSI
# Fungsi:
# Menampilkan citra yang diunggah serta hasil klasifikasi,
# probabilitas AI, dan confidence model.
# ============================================================

if uploaded_file is not None:

    image_bytes = (
        uploaded_file.getvalue()
    )

    display_image = Image.open(
        uploaded_file
    ).convert(
        "RGB"
    )

    st.image(
        display_image,
        caption="Citra Input",
        use_container_width=True
    )

    with st.spinner(
        "Melakukan klasifikasi..."
    ):

        (
            predicted_class,
            probability_ai,
            confidence
        ) = predict_image(
            image_bytes
        )

    st.subheader(
        "Hasil Klasifikasi"
    )

    st.write(
        f"**Prediksi:** "
        f"{predicted_class}"
    )

    st.write(
        f"**Probability AI:** "
        f"{probability_ai:.6f}"
    )

    st.write(
        f"**Confidence:** "
        f"{confidence * 100:.2f}%"
    )
