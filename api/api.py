from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charger le modèle
model = tf.keras.models.load_model("brain_tumor_model.h5")
CLASS_NAMES = ['Glioma', 'Meningioma', 'Pituitary', 'No Tumor']
IMG_SIZE = 224

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    image_file = request.files['file']
    image_bytes = image_file.read()

    img = preprocess_image(image_bytes)
    preds = model.predict(img)
    predicted_index = np.argmax(preds)
    confidence = float(np.max(preds)) * 100
    predicted_label = CLASS_NAMES[predicted_index]

    return jsonify({
        'prediction': predicted_label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
