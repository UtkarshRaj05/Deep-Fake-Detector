import streamlit as st
from PIL import Image
from inference import DeepfakeInference

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 30px;
    }

    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
    }

    .real {
        border: 2px solid #28a745;
    }

    .fake {
        border: 2px solid #dc3545;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# Title
# --------------------------------------------------

st.markdown(
    '<div class="main-title">🔍 Deepfake Image Detector</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Upload an image to determine whether it is Real or AI-Generated</div>',
    unsafe_allow_html=True
)

# --------------------------------------------------
# Load Model
# --------------------------------------------------


@st.cache_resource
def load_model():
    return DeepfakeInference()


with st.spinner("Loading deepfake detection model..."):
    try:
        engine = load_model()
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()


# --------------------------------------------------
# Model Status
# --------------------------------------------------

if engine.checkpoint_loaded:
    st.success("✅ Trained model loaded successfully")
else:
    st.warning(
        "⚠️ Trained checkpoint was not loaded. "
        "Please check your model checkpoint path."
    )

# --------------------------------------------------
# File Upload
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP and BMP"
)

# --------------------------------------------------
# Image Analysis
# --------------------------------------------------

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file).convert("RGB")

    except Exception:
        st.error("❌ Could not read this file as an image.")
        st.stop()

    st.divider()

    # Two columns
    col1, col2 = st.columns(2)

    # --------------------------------------------------
    # Original Image
    # --------------------------------------------------

    with col1:

        st.subheader("📷 Uploaded Image")

        st.image(
            image,
            use_container_width=True
        )

        st.caption(
            f"Filename: {uploaded_file.name}"
        )

    # --------------------------------------------------
    # Analyze Button
    # --------------------------------------------------

    if st.button(
        "🔎 Analyze Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Analyzing image..."):

            try:

                result = engine.predict(
                    image,
                    with_heatmap=True
                )

            except Exception as e:

                st.error(
                    f"❌ Error while analyzing image: {e}"
                )

                st.stop()

        st.divider()

        # --------------------------------------------------
        # Result
        # --------------------------------------------------

        st.subheader("📊 Detection Result")

        # Print result temporarily so we can see
        # exactly what DeepfakeInference returns.
        st.write("Raw prediction:", result)

        # --------------------------------------------------
        # Extract Prediction
        # --------------------------------------------------

        label = result.get(
            "label",
            result.get(
                "prediction",
                result.get("class", "Unknown")
            )
        )

        confidence = result.get(
            "confidence",
            result.get("probability", None)
        )

        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        if str(label).lower() in [
            "fake",
            "deepfake",
            "ai",
            "ai-generated"
        ]:

            st.error(
                f"🚨 FAKE / AI-GENERATED"
            )

        elif str(label).lower() in [
            "real",
            "genuine"
        ]:

            st.success(
                f"✅ REAL IMAGE"
            )

        else:

            st.info(
                f"Prediction: {label}"
            )

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        if confidence is not None:

            try:

                confidence_value = float(confidence)

                if confidence_value <= 1:
                    confidence_value *= 100

                st.metric(
                    "Confidence",
                    f"{confidence_value:.2f}%"
                )

                st.progress(
                    min(confidence_value / 100, 1.0)
                )

            except (ValueError, TypeError):

                st.write(
                    f"Confidence: {confidence}"
                )

        # --------------------------------------------------
        # Grad-CAM Heatmap
        # --------------------------------------------------

        heatmap = result.get("heatmap")

        if heatmap is not None:

            st.subheader(
                "🔥 Grad-CAM Heatmap"
            )

            try:

                st.image(
                    heatmap,
                    caption="Regions influencing the model's decision",
                    use_container_width=True
                )

            except Exception as e:

                st.warning(
                    f"Could not display heatmap: {e}"
                )
