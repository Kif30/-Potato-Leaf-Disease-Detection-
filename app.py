import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="Potato Disease Detection",
    page_icon="🥔",
    layout="centered"
)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "potato_model (1).keras",compile=False
    )

model = load_model()

# ---------------------------------
# CLASS NAMES
# MUST MATCH TRAINING ORDER
# ---------------------------------
CLASS_NAMES = [
    "Early Blight",
    "Late Blight",
    "Healthy"
]

# ---------------------------------
# TITLE
# ---------------------------------
st.title("🥔 Potato Leaf Disease Detector")

st.write(
    "Upload a potato leaf image and the CNN model "
    "will classify the disease."
)

# ---------------------------------
# IMAGE UPLOADER
# ---------------------------------
uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------------------------
# PREDICTION FUNCTION
# ---------------------------------
def predict(image):

    # Convert to RGB
    image = image.convert("RGB")

    # Resize to TRAINING SIZE
    image = image.resize((224, 224))

    # Convert to numpy
    img_array = np.array(image)

    # Normalize EXACTLY like training
    img_array = img_array / 255.0

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # Predict
    predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions[0])

    predicted_class = CLASS_NAMES[predicted_index]

    confidence = float(
        predictions[0][predicted_index]
    )

    return predicted_class, confidence, predictions[0]

# ---------------------------------
# RUN PREDICTION
# ---------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    with st.spinner("Analyzing image..."):

        predicted_class, confidence, probs = predict(image)

    st.success(
        f"Prediction: {predicted_class}"
    )

    st.info(
        f"Confidence: {confidence * 100:.2f}%"
    )

    # ---------------------------------
    # SHOW PROBABILITIES
    # ---------------------------------
    st.subheader("Prediction Probabilities")

    for i, class_name in enumerate(CLASS_NAMES):
        st.write(
            f"{class_name}: "
            f"{probs[i] * 100:.2f}%"
        )
        st.progress(float(probs[i]))

    # ---------------------------------
    # DISEASE DETAILS
    # ---------------------------------
    if predicted_class == "Early Blight":

        st.warning(
            """
            Early Blight Detected.

            Symptoms:
            - Brown spots
            - Yellow leaf edges

            Recommendation:
            - Use fungicide
            - Remove infected leaves
            """
        )

    elif predicted_class == "Late Blight":

        st.error(
            """
            Late Blight Detected.

            Symptoms:
            - Dark water-soaked lesions
            - Rapid spread

            Recommendation:
            - Immediate fungicide treatment
            - Isolate infected plants
            """
        )

    else:

        st.success(
            """
            Healthy Leaf Detected.

            No disease symptoms found.
            """
        )
st.write("NEW VERSION RUNNING")
