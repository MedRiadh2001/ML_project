import streamlit as st
import requests
from PIL import Image
import io

# Titre
st.title("🧠 Détection de Tumeurs Cérébrales avec Flask + Streamlit")

# API URL
API_URL = "http://127.0.0.1:5000/predict"

# Upload image
uploaded_file = st.file_uploader("📁 Téléversez une IRM", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="🖼️ Image IRM", width=300)

    if st.button("🔍 Prédire"):
        # Préparer fichier à envoyer
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')
        image_bytes.seek(0)

        files = {'file': image_bytes}
        response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            result = response.json()
            st.success(f"✅ Tumeur détectée : **{result['prediction']}**")
            st.info(f"📊 Confiance : {result['confidence']:.2f}%")
        else:
            st.error(f"❌ Erreur : {response.text}")
