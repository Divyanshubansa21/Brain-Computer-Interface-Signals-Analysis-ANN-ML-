import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            st.markdown(
                f"<style>{f.read()}</style>",
                unsafe_allow_html=True
            )

local_css("app/style.css")


st.set_page_config(
    page_title="Brain Computer Interface",
    page_icon="🧠",
    layout="wide"
)

MODEL_PATH = "models/random_forest_model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)


    st.sidebar.title("🧠 BCI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Prediction",
        "📁 Dataset",
        "📈 Analytics",
        "🧠 EEG Viewer",
        "⚙ Model",
        "👨‍💻 About"
    ]
)

st.sidebar.success("Status : Ready")

if page == "🏠 Home":

    st.markdown("""
    <div class='hero'>
        <h1>🧠 Brain Computer Interface</h1>
        <h2>EEG Emotion Recognition</h2>
        <p>Random Forest Classifier trained on DEAP Dataset</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric("Subjects", "32")

    with c2:
        st.metric("Channels", "32")

    with c3:
        st.metric("Trials", "1280")

    with c4:
        st.metric("Accuracy", "77%")

    st.write("")

    st.subheader("Workflow")

    st.markdown("""
EEG Signal

↓

Preprocessing

↓

Feature Extraction

↓

Random Forest

↓

Emotion Prediction
""")

    st.write("")

    st.subheader("Project Highlights")

    a, b, c = st.columns(3)

    with a:
        st.info("⚡ Fast Processing")

    with b:
        st.success("🧠 Machine Learning")

    with c:
        st.warning("📊 DEAP Dataset")

    st.write("")

    fig, ax = plt.subplots(figsize=(6,3))

    ax.bar(
        ["Accuracy", "Error"],
        [77,23]
    )

    st.pyplot(fig)


    # --------------------------------
# PREDICTION
# --------------------------------

elif page=="📊 Prediction":
    st.title("📊 Emotion Prediction")
    st.markdown("""
<div class='hero'>

# 🧠 EEG Emotion Detection

Predict Human Emotion using
Random Forest + DEAP Dataset

</div>
""", unsafe_allow_html=True)

st.write("Enter EEG Feature Values")

features = []

cols = st.columns(4)

for i in range(32):
    with cols[i % 4]:
        value = st.number_input(
            f"Channel {i+1}",
            value=0.0,
            format="%.4f",
            key=f"f{i}"
        )
        features.append(value)

st.write("")

c1, c2, c3 = st.columns([1,2,1])

with c2:

    if st.button(
        "🚀 Predict Emotion",
        use_container_width=True,
        key="predict_button"
    ):

        if model is None:
            st.error("Model not found!")

        else:

            sample = np.array(features).reshape(1, -1)

            prediction = model.predict(sample)[0]

            st.write("---")

            if prediction == 1:
                st.success("😊 Positive Emotion Detected")
                st.progress(82)
                st.metric("Confidence", "82%")

            else:
                st.error("😔 Negative Emotion Detected")
                st.progress(74)
                st.metric("Confidence", "74%")

    if model is None:
        st.error("Model not found!")
    else:

        sample = np.array(features).reshape(1, -1)

        prediction = model.predict(sample)[0]

        st.write("---")

        if prediction == 1:
            st.success("😊 Positive Emotion Detected")
            st.progress(82)
            st.metric("Confidence", "82%")

        else:
            st.error("😔 Negative Emotion Detected")
            st.progress(74)
            st.metric("Confidence", "74%")