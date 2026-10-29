import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Skin Cancer Classifier", layout="centered")

st.title("🩺 Skin Cancer & Disease Classification")
st.write("Upload an image of the skin lesion to predict the skin condition using our trained model.")

# 1. Load the model (cached so it only loads once)
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('skin_cancer_model.keras')

try:
    model = load_my_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.info("Make sure 'skin_cancer_model.keras' is in the same directory as this script.")

# 2. File Uploader
uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

# Define the actual disease classes found inside the 'train' folder
class_names = [
    'Acne and Rosacea Photos',
    'Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions',
    'Atopic Dermatitis Photos',
    'Cellulitis Impetigo and other Bacterial Infections',
    'Eczema Photos',
    'Exanthems and Drug Eruptions',
    'Herpes HPV and other STDs Photos',
    'Light Diseases and Disorders of Pigmentation',
    'Lupus and other Connective Tissue diseases',
    'Melanoma Skin Cancer Nevi and Moles',
    'Poison Ivy Photos and other Contact Dermatitis',
    'Psoriasis pictures Lichen Planus and related diseases',
    'Seborrheic Keratoses and other Benign Tumors',
    'Systemic Disease',
    'Tinea Ringworm Candidiasis and other Fungal Infections',
    'Urticaria Hives',
    'Vascular Tumors',
    'Vasculitis Photos',
    'Warts Molluscum and other Viral Infections'
]

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    st.write("Processing and predicting...")

    # 3. Preprocess Image
    img = image.convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # 4. Make prediction
    predictions = model.predict(img_tensor)
    confidence_scores = predictions[0]
    
    # If the model was compiled for 2 classes, map dynamically, otherwise map across all categories
    if len(confidence_scores) == 2:
        # Map binary output
        predicted_class_idx = np.argmax(confidence_scores)
        # Fallback to binary test/train mapping if model was trained as binary
        binary_mapping = ["Test Class Image", "Train Class Image"]
        predicted_class = binary_mapping[predicted_class_idx]
        st.subheader("Prediction Results:")
        st.metric(label="Predicted Class Group", value=predicted_class)
        st.write(f"**Confidence:** {confidence_scores[predicted_class_idx]*100:.2f}%")
    else:
        # Map multi-class classification labels
        predicted_class_idx = np.argmax(confidence_scores)
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class {predicted_class_idx}"
        
        confidence = confidence_scores[predicted_class_idx] * 100

        # 5. Display Results
        st.subheader("Prediction Results:")
        st.metric(label="Predicted Skin Condition", value=predicted_class)
        st.metric(label="Confidence Score", value=f"{confidence:.2f}%")

        st.write("**Condition Breakdown:**")
        top_indices = np.argsort(confidence_scores)[::-1][:3] # Top 3 predictions
        for idx in top_indices:
            if idx < len(class_names):
                st.write(f"- **{class_names[idx]}**: {confidence_scores[idx] * 100:.2f}%")
