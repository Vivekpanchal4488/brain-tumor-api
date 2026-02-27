from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
CORS(app, origins="*")

print("Loading model...")
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_model.keras')
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = float(np.max(predictions) * 100)
        has_tumor = predicted_class != 'notumor'
        return jsonify({
            'hasTumor': has_tumor,
            'tumorType': predicted_class,
            'confidence': round(confidence, 2),
            'allProbabilities': {
                class_names[i]: round(float(predictions[0][i]) * 100, 2)
                for i in range(4)
            }
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'API is running!', 'model': 'loaded'})

if __name__ == '__main__':
    app.run(debug=True)
