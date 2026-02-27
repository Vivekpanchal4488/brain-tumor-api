from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import os
import gc
import tensorflow as tf

app = Flask(__name__)
CORS(app, origins="*")

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'brain_tumor.tflite')
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("TFLite model loaded!")

class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    try:
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB').resize((224, 224))
        img_array = np.expand_dims(np.array(img, dtype=np.float32), axis=0)
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        gc.collect()
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
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def health():
    return jsonify({'status': 'API is running!', 'model': 'tflite'})

if __name__ == '__main__':
    app.run(debug=True)
