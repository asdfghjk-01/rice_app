 import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model (assuming it's saved as 'model.h5')
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('imageclassifier.h5')
    return model

model = load_model()

# Define class names corresponding to your dataset classes
CLASS_NAMES = ['Bacterial Blight', 'Healthy Rice Leaf', 'Rice Blast', 'Tungro']
CONFIDENCE_THRESHOLD = 0.5  # Set a threshold for confidence

# Updated Image preprocessing function
def preprocess_image(image):
    img = image.resize((256, 256))  # Use the correct size as per your model
    img = np.array(img) / 255.0      # Normalize the image
    img = np.expand_dims(img, axis=0) # Add batch dimension
    return img

# Streamlit app interface
st.title("Rice Leaf Disease Classifier")
st.write("Upload an image of a rice leaf, and the model will classify its disease.")

# Image upload
uploaded_file = st.file_uploader("Choose a rice leaf image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image for the model
    preprocessed_image = preprocess_image(image)
    
    # Make a prediction
    predictions = model.predict(preprocessed_image)
    predicted_class = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)  # Get the confidence score (not converted to percentage yet)

    # Check confidence against the threshold
    if confidence < CONFIDENCE_THRESHOLD:
        st.write("The uploaded image does not match any known rice leaf disease.")
    else:
        # Convert confidence to percentage for display
        confidence_percentage = confidence * 100
        # Display the result
        st.write(f"Predicted class: {CLASS_NAMES[predicted_class]}")
        st.write(f"Confidence: {confidence_percentage:.2f}%")  # Correct format with two decimal places
