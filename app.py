import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_model.keras')
    return model

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

st.set_page_config(page_title="Brain Tumor Classifier", page_icon="🧠")
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan to detect the type of brain tumor")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded MRI', width=300)

    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner('Analyzing...'):
        predictions = model.predict(img_array)
        predicted_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100

    st.subheader("Results:")
    if predicted_class == 'notumor':
        st.success(f"✅ No Tumor Detected ({confidence:.1f}% confidence)")
    else:
        st.error(f"⚠️ {predicted_class.capitalize()} Tumor Detected ({confidence:.1f}% confidence)")

    st.subheader("Probabilities:")
    for i, cls in enumerate(class_names):
        st.progress(float(predictions[0][i]), text=f"{cls}: {predictions[0][i]*100:.1f}%")