import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------
# PAGE CONFIG
# ---------------------------------
st.set_page_config(
    page_title="PotatoScan — Disease Detector",
    page_icon="🥔",
    layout="centered"
)

# ---------------------------------
# CSS — BOTANICAL FIELD JOURNAL
# ---------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=DM+Mono:wght@300;400&family=Lato:wght@300;400&display=swap');

:root {
    --soil:     #1C1208;
    --bark:     #2E1F0A;
    --moss:     #3B4A2F;
    --sage:     #6B7F58;
    --leaf:     #8FAF6A;
    --lime:     #B8D48A;
    --cream:    #F0E8D0;
    --parchment:#E8DCC0;
    --rust:     #C4581A;
    --blight:   #8B2500;
    --amber:    #D4831A;
    --gold:     #E8B84B;
}

html, body, [class*="css"] {
    background-color: var(--soil);
    color: var(--cream);
    font-family: 'Lato', sans-serif;
    font-weight: 300;
}

.stApp {
    background: var(--soil);
    background-image:
        radial-gradient(ellipse at 0% 0%, rgba(139,175,106,0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 100% 100%, rgba(196,88,26,0.04) 0%, transparent 55%),
        url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='400' height='400'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.75' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='400' height='400' filter='url(%23n)' opacity='0.035'/%3E%3C/svg%3E");
}

/* ---- HEADER ---- */
.field-header {
    padding: 2.5rem 0 0.5rem 0;
    text-align: center;
    position: relative;
}

.field-header .eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.4em;
    color: var(--sage);
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}

.field-header h1 {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--cream);
    line-height: 1.1;
    margin: 0 0 0.3rem 0;
    letter-spacing: -0.01em;
}

.field-header h1 span {
    color: var(--leaf);
    font-style: italic;
}

.field-header .tagline {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 0.95rem;
    color: var(--sage);
    margin-bottom: 1.5rem;
}

.ornament {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1rem;
    margin: 0.5rem 0 1.8rem 0;
}

.ornament-line {
    flex: 1;
    max-width: 120px;
    height: 1px;
    background: linear-gradient(to right, transparent, var(--sage));
}

.ornament-line.right {
    background: linear-gradient(to left, transparent, var(--sage));
}

.ornament-diamond {
    width: 6px; height: 6px;
    background: var(--leaf);
    transform: rotate(45deg);
}

/* ---- UPLOAD ZONE ---- */
.upload-heading {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.3em;
    color: var(--sage);
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

[data-testid="stFileUploader"] {
    background: rgba(46,31,10,0.6) !important;
    border: 1px solid rgba(107,127,88,0.35) !important;
    border-radius: 2px !important;
    transition: all 0.3s ease !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: rgba(143,175,106,0.6) !important;
    background: rgba(46,31,10,0.8) !important;
}

/* ---- IMAGE ---- */
[data-testid="stImage"] {
    border: 1px solid rgba(107,127,88,0.3);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}

/* ---- BUTTON ---- */
.stButton > button {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.25em !important;
    text-transform: uppercase !important;
    color: var(--soil) !important;
    background: var(--leaf) !important;
    border: none !important;
    border-radius: 1px !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
    font-weight: 400 !important;
    transition: all 0.25s ease !important;
}

.stButton > button:hover {
    background: var(--lime) !important;
    box-shadow: 0 4px 20px rgba(143,175,106,0.3) !important;
    transform: translateY(-1px) !important;
}

/* ---- RESULT CARD ---- */
.result-card {
    margin: 1.8rem 0 1rem 0;
    padding: 1.8rem 2rem;
    border-radius: 2px;
    position: relative;
    animation: fadeUp 0.5s ease;
}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

.result-healthy {
    background: rgba(59,74,47,0.25);
    border: 1px solid rgba(143,175,106,0.4);
    border-left: 3px solid var(--leaf);
}

.result-early {
    background: rgba(212,131,26,0.1);
    border: 1px solid rgba(212,131,26,0.35);
    border-left: 3px solid var(--amber);
}

.result-late {
    background: rgba(139,37,0,0.15);
    border: 1px solid rgba(196,88,26,0.4);
    border-left: 3px solid var(--rust);
}

.result-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}

.result-healthy .result-label { color: var(--sage); }
.result-early   .result-label { color: var(--amber); }
.result-late    .result-label { color: var(--rust); }

.result-name {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 0.3rem;
}

.result-healthy .result-name { color: var(--lime); }
.result-early   .result-name { color: var(--gold); }
.result-late    .result-name { color: var(--rust); }

.result-conf {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: var(--sage);
    margin-bottom: 1rem;
}

/* ---- PROB BARS ---- */
.prob-section-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.58rem;
    letter-spacing: 0.3em;
    color: var(--sage);
    text-transform: uppercase;
    margin: 1.6rem 0 0.8rem 0;
    border-bottom: 1px solid rgba(107,127,88,0.2);
    padding-bottom: 0.4rem;
}

.prob-row {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.7rem;
}

.prob-name {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: var(--parchment);
    width: 100px;
    flex-shrink: 0;
}

.prob-bar-bg {
    flex: 1;
    height: 4px;
    background: rgba(107,127,88,0.15);
    border-radius: 2px;
    overflow: hidden;
}

.prob-bar-fill-healthy { background: var(--leaf); }
.prob-bar-fill-early   { background: var(--amber); }
.prob-bar-fill-late    { background: var(--rust); }

.prob-pct {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: var(--sage);
    width: 42px;
    text-align: right;
    flex-shrink: 0;
}

/* ---- DISEASE INFO ---- */
.info-block {
    margin: 1.5rem 0 0 0;
    padding: 1.2rem 1.5rem;
    background: rgba(28,18,8,0.5);
    border-top: 1px solid rgba(107,127,88,0.2);
}

.info-block h4 {
    font-family: 'Playfair Display', serif;
    font-style: italic;
    font-size: 1rem;
    margin: 0 0 0.8rem 0;
}

.result-healthy .info-block h4 { color: var(--leaf); }
.result-early   .info-block h4 { color: var(--gold); }
.result-late    .info-block h4 { color: var(--rust); }

.info-cols {
    display: flex;
    gap: 2rem;
}

.info-col-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--sage);
    margin-bottom: 0.4rem;
}

.info-col ul {
    margin: 0;
    padding-left: 1rem;
    font-size: 0.78rem;
    color: var(--parchment);
    line-height: 1.7;
}

/* ---- FOOTER ---- */
.field-footer {
    text-align: center;
    padding: 2.5rem 0 1rem 0;
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem;
    letter-spacing: 0.3em;
    color: rgba(107,127,88,0.4);
    text-transform: uppercase;
}

/* ---- HIDE STREAMLIT CHROME ---- */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 0 !important; max-width: 700px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# LOAD MODEL
# ---------------------------------
@st.cache_resource
def load_detection_model():
    return tf.keras.models.load_model(
        "potato_model (1).keras", compile=False
    )

model = load_detection_model()

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# ---------------------------------
# HEADER
# ---------------------------------
st.markdown("""
<div class="field-header">
    <div class="eyebrow">Agricultural AI · Field Analysis System</div>
    <h1>Potato<span>Scan</span></h1>
    <div class="tagline">CNN-powered leaf disease identification</div>
    <div class="ornament">
        <div class="ornament-line"></div>
        <div class="ornament-diamond"></div>
        <div class="ornament-line right"></div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------
# UPLOADER
# ---------------------------------
st.markdown('<div class="upload-heading">// Leaf Sample</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a potato leaf photograph",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# ---------------------------------
# PREDICTION
# ---------------------------------
def predict(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    preds = model.predict(arr, verbose=0)
    idx = int(np.argmax(preds[0]))
    return CLASS_NAMES[idx], float(preds[0][idx]), preds[0]

# ---------------------------------
# MAIN
# ---------------------------------
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    st.markdown("<br>", unsafe_allow_html=True)
    st.image(img, caption="", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⬡  Analyse Leaf Sample"):
        with st.spinner("Running neural classification..."):
            predicted_class, confidence, probs = predict(img)

        # Map class to style
        style_map = {
            "Healthy":      ("healthy", "✦ No Pathogen Detected"),
            "Early Blight": ("early",   "⚠ Pathogen Identified"),
            "Late Blight":  ("late",    "✦ Critical Infection"),
        }
        css_class, status_label = style_map[predicted_class]

        # Result card
        prob_bars = ""
        bar_color_map = {
            "Early Blight": "early",
            "Late Blight":  "late",
            "Healthy":      "healthy"
        }
        for i, name in enumerate(CLASS_NAMES):
            pct = probs[i] * 100
            color = bar_color_map[name]
            prob_bars += f"""
            <div class="prob-row">
                <div class="prob-name">{name}</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill-{color}" style="width:{pct:.1f}%;height:100%;border-radius:2px;"></div>
                </div>
                <div class="prob-pct">{pct:.1f}%</div>
            </div>
            """

        # Disease details
        disease_info = {
            "Healthy": (
                "Leaf is in good condition",
                ["No lesions visible", "Uniform green colour", "Normal growth pattern"],
                ["Continue regular care", "Monitor for early signs", "Maintain soil health"]
            ),
            "Early Blight": (
                "Alternaria solani detected",
                ["Brown concentric-ring spots", "Yellowing leaf margins", "Lower leaves affected first"],
                ["Apply copper fungicide", "Remove infected foliage", "Improve air circulation"]
            ),
            "Late Blight": (
                "Phytophthora infestans detected",
                ["Water-soaked dark lesions", "White mould on undersides", "Rapid tissue collapse"],
                ["Immediate fungicide treatment", "Isolate affected plants", "Avoid overhead watering"]
            ),
        }

        disease_title, symptoms, actions = disease_info[predicted_class]

        symptoms_html = "".join(f"<li>{s}</li>" for s in symptoms)
        actions_html  = "".join(f"<li>{a}</li>" for a in actions)

        st.markdown(f"""
        <div class="result-card result-{css_class}">
            <div class="result-label">{status_label}</div>
            <div class="result-name">{predicted_class}</div>
            <div class="result-conf">Confidence: {confidence*100:.2f}% &nbsp;·&nbsp; Model: CNN 224×224</div>

            <div class="prob-section-title">// Class Probabilities</div>
            {prob_bars}

            <div class="info-block">
                <h4>{disease_title}</h4>
                <div class="info-cols">
                    <div>
                        <div class="info-col-label">Symptoms</div>
                        <ul>{symptoms_html}</ul>
                    </div>
                    <div>
                        <div class="info-col-label">Recommended Action</div>
                        <ul>{actions_html}</ul>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------
# FOOTER
# ---------------------------------
st.markdown("""
<div class="field-footer">
    ◈ &nbsp; Built with TensorFlow · CNN · Streamlit &nbsp; ◈
</div>
""", unsafe_allow_html=True)
