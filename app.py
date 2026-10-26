import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px

from PIL import Image, ImageOps
from pathlib import Path


# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Pneumonia CNN | X-ray Analysis",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------- PATHS & MODEL ----------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_pneumonia_cnn.keras"

CLASS_NAMES = {
    0: "Normal",
    1: "Bacterial",
    2: "Viral",
}


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# ---------------- CUSTOM STYLING ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');

    :root {
        --glass: rgba(255, 255, 255, 0.10);
        --glass-strong: rgba(255, 255, 255, 0.16);
        --glass-border: rgba(255, 255, 255, 0.18);
        --text: #FFFFFF;
        --text-soft: rgba(255, 255, 255, 0.70);
        --text-faint: rgba(255, 255, 255, 0.50);
        --blue: #3FA7F5;
        --violet: #7C6CF2;
        --mauve: #C98FB0;
    }

    html, body, [class*="css"], .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* Twilight forest-style background */
    .stApp {
        background:
            radial-gradient(ellipse 60% 45% at 18% 0%, rgba(110, 112, 255, 0.45), transparent 70%),
            radial-gradient(ellipse 45% 40% at 88% 8%, rgba(63, 140, 246, 0.35), transparent 70%),
            radial-gradient(ellipse 50% 40% at 8% 95%, rgba(96, 130, 60, 0.40), transparent 70%),
            radial-gradient(ellipse 50% 45% at 92% 92%, rgba(120, 80, 220, 0.45), transparent 70%),
            linear-gradient(165deg, #0A0E2C 0%, #141B4D 45%, #101F3C 75%, #13230F 100%);
        background-attachment: fixed;
        color: var(--text);
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    /* Global text colours */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4,
    .stApp p, .stApp label, .stApp li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stCaptionContainer"] {
        color: var(--text);
    }

    [data-testid="stCaptionContainer"],
    .stApp small {
        color: var(--text-faint) !important;
    }

    hr, [data-testid="stDivider"] {
        border-color: rgba(255, 255, 255, 0.12) !important;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(18, 22, 64, 0.55);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-right: 1px solid var(--glass-border);
    }

    [data-testid="stSidebar"] h1 {
        font-weight: 300;
        letter-spacing: -0.5px;
    }

    [data-testid="stSidebar"] h3 {
        font-weight: 400;
        font-size: 18px;
    }

    /* Hero card */
    .hero {
        position: relative;
        overflow: hidden;
        background: linear-gradient(135deg, rgba(120, 130, 220, 0.38), rgba(40, 48, 110, 0.35));
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(28px);
        -webkit-backdrop-filter: blur(28px);
        padding: 40px 36px;
        border-radius: 32px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 20px 60px rgba(5, 8, 40, 0.45);
    }

    .hero::after {
        content: "";
        position: absolute;
        right: -60px;
        top: -80px;
        width: 380px;
        height: 380px;
        background: radial-gradient(circle,
            rgba(80, 150, 255, 0.55) 0%,
            rgba(200, 80, 140, 0.30) 40%,
            transparent 70%);
        filter: blur(20px);
        pointer-events: none;
    }

    .hero h1 {
        position: relative;
        z-index: 1;
        color: white;
        font-size: 3.4rem;
        font-weight: 300;
        line-height: 1.05;
        letter-spacing: -1.5px;
        margin-bottom: 14px;
    }

    .hero p {
        position: relative;
        z-index: 1;
        color: var(--text-soft);
        font-size: 16px;
        max-width: 520px;
        margin-bottom: 0;
    }

    /* Metric cards */
    .metric-card {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        padding: 22px;
        border-radius: 24px;
        min-height: 120px;
        box-shadow: 0 12px 36px rgba(5, 8, 40, 0.35);
    }

    .metric-label {
        color: var(--text-faint);
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #FFFFFF;
        font-size: 30px;
        font-weight: 400;
        letter-spacing: -0.5px;
    }

    /* Make the first metric card the blue "visit" style card */
    [data-testid="stHorizontalBlock"] > div:first-child .metric-card {
        background: linear-gradient(135deg, #4FB4F7 0%, #2E7BE6 100%);
        border: none;
    }

    [data-testid="stHorizontalBlock"] > div:first-child .metric-card .metric-label {
        color: rgba(255, 255, 255, 0.80);
    }

    .section-title {
        color: #FFFFFF;
        font-size: 22px;
        font-weight: 400;
        letter-spacing: -0.3px;
        margin: 12px 0;
    }

    /* Disclaimer -> mauve alert card */
    .disclaimer {
        position: relative;
        background: linear-gradient(135deg, rgba(140, 110, 185, 0.80), rgba(215, 140, 150, 0.72));
        border: 1px solid rgba(255, 255, 255, 0.22);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 22px;
        padding: 18px 20px 18px 44px;
        color: #FFFFFF;
        font-size: 14px;
        box-shadow: 0 12px 36px rgba(60, 20, 90, 0.35);
    }

    .disclaimer::before {
        content: "";
        position: absolute;
        left: 18px;
        top: 22px;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background: #FFFFFF;
        box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
    }

    .disclaimer strong {
        font-weight: 600;
    }

    /* Buttons (pill shaped) */
    .stButton > button,
    [data-testid="stFileUploaderDropzone"] button {
        border-radius: 999px;
        border: 1px solid var(--glass-border);
        background: var(--glass-strong);
        color: #FFFFFF;
        padding: 0.6rem 1.4rem;
        font-weight: 500;
        backdrop-filter: blur(14px);
        transition: all 0.2s ease;
    }

    .stButton > button:hover,
    [data-testid="stFileUploaderDropzone"] button:hover {
        border-color: rgba(255, 255, 255, 0.45);
        color: #FFFFFF;
        background: rgba(255, 255, 255, 0.24);
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4FB4F7 0%, #7C6CF2 100%);
        border: none;
        box-shadow: 0 10px 28px rgba(80, 120, 240, 0.45);
    }

    .stButton > button[kind="primary"]:hover {
        filter: brightness(1.1);
        box-shadow: 0 12px 34px rgba(80, 120, 240, 0.6);
    }

    .stButton > button:disabled {
        background: rgba(255, 255, 255, 0.08);
        color: rgba(255, 255, 255, 0.40);
        box-shadow: none;
        border: 1px solid rgba(255, 255, 255, 0.10);
    }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background: var(--glass);
        border: 1px dashed rgba(255, 255, 255, 0.35);
        border-radius: 24px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
    }

    [data-testid="stFileUploaderDropzone"] * {
        color: var(--text-soft);
    }

    [data-testid="stFileUploaderDropzone"] button {
        color: #FFFFFF;
    }

    [data-testid="stFileUploaderFile"] {
        background: var(--glass);
        border-radius: 16px;
        padding: 6px 10px;
    }

    /* Images */
    [data-testid="stImage"] img {
        border-radius: 24px;
        border: 1px solid var(--glass-border);
        box-shadow: 0 16px 48px rgba(5, 8, 40, 0.5);
    }

    /* Alerts (info / error) */
    [data-testid="stAlert"] {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 20px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        color: #FFFFFF;
    }

    [data-testid="stAlert"] * {
        color: #FFFFFF;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 22px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        overflow: hidden;
    }

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary * {
        color: #FFFFFF;
    }

    /* Chart & dataframe containers */
    [data-testid="stPlotlyChart"] {
        background: var(--glass);
        border: 1px solid var(--glass-border);
        border-radius: 26px;
        padding: 12px;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
    }

    [data-testid="stDataFrame"] {
        border-radius: 18px;
        overflow: hidden;
    }

    [data-testid="stSpinner"] * {
        color: var(--text-soft);
    }
</style>
""", unsafe_allow_html=True)


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("🫁 Pneumonia CNN")
    st.caption("Chest X-ray classification dashboard")
    st.divider()

    st.subheader("Model information")
    st.write("**Architecture:** Custom CNN")
    st.write("**Input:** 224 × 224 grayscale")
    st.write("**Output:** 3 classes")

    st.divider()
    st.caption("Local prototype • Model inference")


# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
    <h1>Chest X-ray Analysis</h1>
    <p>
        Upload a chest X-ray to explore predictions from your trained
        convolutional neural network.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
    <strong>Research prototype:</strong> This application is for educational
    demonstration only. It is not clinically validated and must not be used
    to diagnose, exclude, or make treatment decisions about pneumonia.
</div>
""", unsafe_allow_html=True)

st.write("")


# ---------------- LOAD MODEL ----------------
try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# ---------------- UPLOAD ----------------
st.markdown(
    '<div class="section-title">Upload chest X-ray</div>',
    unsafe_allow_html=True
)

left, right = st.columns([1, 1.3], gap="large")

with left:
    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"],
        help="Upload a chest X-ray image in PNG or JPEG format."
    )

    st.caption("Supported formats: PNG, JPG, JPEG")

    analyze = st.button(
        "Analyze X-ray",
        type="primary",
        use_container_width=True,
        disabled=uploaded_file is None
    )


# ---------------- PREPROCESSING ----------------
def preprocess_image(image):
    image = ImageOps.exif_transpose(image)
    image = image.convert("L")
    image = image.resize((224, 224))

    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=-1)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


with right:
    st.markdown(
        '<div class="section-title">Image preview</div>',
        unsafe_allow_html=True
    )

    if uploaded_file is not None:
        try:
            preview = Image.open(uploaded_file)
            preview = ImageOps.exif_transpose(preview).convert("RGB")
            st.image(
                preview,
                caption="Uploaded chest X-ray",
                use_container_width=True
            )
        except Exception:
            st.error("The uploaded file could not be opened.")


# ---------------- PREDICTION ----------------
if analyze and uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        input_array = preprocess_image(image)

        with st.spinner("Running model inference..."):
            probabilities = model.predict(input_array, verbose=0)[0]

        predicted_class = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_class]) * 100

        st.divider()
        st.markdown(
            '<div class="section-title">Prediction results</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Predicted class</div>
                <div class="metric-value">{CLASS_NAMES[predicted_class]}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model confidence</div>
                <div class="metric-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Classes evaluated</div>
                <div class="metric-value">3</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown(
            '<div class="section-title">Class probability distribution</div>',
            unsafe_allow_html=True
        )

        results_df = pd.DataFrame({
            "Class": [CLASS_NAMES[i] for i in range(3)],
            "Probability (%)": probabilities * 100
        })

        fig = px.bar(
            results_df,
            x="Class",
            y="Probability (%)",
            text=results_df["Probability (%)"].map(
                lambda value: f"{value:.2f}%"
            ),
            color="Class",
            title="Model output probabilities",
            color_discrete_sequence=["#4FB4F7", "#7C6CF2", "#D98FB0"]
        )

        fig.update_traces(
            marker_line_width=0,
            textfont=dict(color="white", size=14),
            textposition="outside",
            cliponaxis=False
        )

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif", color="white"),
            showlegend=False,
            yaxis_range=[0, 100],
            yaxis=dict(gridcolor="rgba(255,255,255,0.12)", zeroline=False),
            xaxis=dict(gridcolor="rgba(255,255,255,0.0)"),
            bargap=0.45,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        st.plotly_chart(fig, use_container_width=True)

        with st.expander("View numerical probabilities"):
            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True
            )

        st.info(
            "These values are model output probabilities, not verified "
            "clinical probabilities. A high score does not establish "
            "that a prediction is correct."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


# ---------------- FOOTER ----------------
st.divider()
st.caption(
    "Pneumonia CNN • Educational machine-learning project • "
    "Not for clinical use"
)