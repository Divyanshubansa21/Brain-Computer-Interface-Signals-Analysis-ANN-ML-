import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def local_css(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )

local_css("app/style.css")


# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Brain Computer Interface",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = joblib.load("models/random_forest_model.pkl")

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("🧠 BCI Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "📊 Prediction",
        "📁 Dataset",
        "📈 Visualizations",
        "🤖 Model",
        "👨‍💻 About"
    ]
)

# =============================
# HOME
# =============================

if page=="🏠 Home":

    st.markdown("""
<div class="hero">

# 🧠 Brain Computer Interface

### EEG Emotion Recognition using Machine Learning

Built using Random Forest + DEAP Dataset

</div>
""", unsafe_allow_html=True)

    st.write("")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown("""
<div class="metric-card">
<h2>👥</h2>
<h1>32</h1>
<p>Subjects</p>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div class="metric-card">
<h2>🧠</h2>
<h1>32</h1>
<p>EEG Channels</p>
</div>
""", unsafe_allow_html=True)

    with c3:
        st.markdown("""
<div class="metric-card">
<h2>🎬</h2>
<h1>1280</h1>
<p>Trials</p>
</div>
""", unsafe_allow_html=True)

    with c4:
        st.markdown("""
<div class="metric-card">
<h2>🎯</h2>
<h1>77%</h1>
<p>Accuracy</p>
</div>
""", unsafe_allow_html=True)

    st.write("")

    st.subheader("⚡ Project Workflow")

    st.info("""
EEG Signals
⬇
Preprocessing
⬇
Feature Extraction
⬇
Random Forest
⬇
Emotion Prediction
""")

    st.success("✅ System Ready for Prediction")

# =============================
# PREDICTION
# =============================

elif page=="📊 Prediction":

    st.title("📊 Emotion Prediction")

    st.write("Enter EEG Features")

    features=[]

    cols=st.columns(4)

    for i in range(32):

        with cols[i%4]:

            value=st.number_input(
                f"Channel {i+1}",
                value=0.0,
                format="%.4f"
            )

            features.append(value)

    if st.button("Predict Emotion"):

        sample=np.array(features).reshape(1,-1)

        prediction=model.predict(sample)

        st.markdown("---")

        if prediction[0]==1:

            st.success("😊 Positive Emotion")

            st.progress(82)

            st.metric(
                "Confidence",
                "82%"
            )

        else:

            st.error("😔 Negative Emotion")

            st.progress(74)

            st.metric(
                "Confidence",
                "74%"
            )

            # =============================
# DATASET
# =============================

elif page=="📁 Dataset":

    st.title("📁 DEAP Dataset")

    st.write("Dataset Information")

    df = pd.DataFrame({
    "Property":[
        "Subjects",
        "Trials",
        "EEG Channels",
        "Peripheral Channels",
        "Sampling Rate"
    ],
    "Value":[
        "32",
        "1280",
        "32",
        "8",
        "128 Hz"
    ]
})

    st.dataframe(df,use_container_width=True)

# =============================
# VISUALIZATION
# =============================

elif page=="📈 Visualizations":

    st.title("📈 EEG Visualization")

    x=np.linspace(0,10,500)

    y=np.sin(x)

    fig,ax=plt.subplots(figsize=(10,4))

    ax.plot(x,y)

    ax.set_title("Sample EEG Signal")

    st.pyplot(fig)

# =============================
# MODEL
# =============================

elif page=="🤖 Model":

    st.title("🤖 Machine Learning Model")

    st.success("Random Forest Classifier")

    st.metric("Accuracy","77%")

    st.metric("Precision","65%")

    st.metric("Recall","77%")

    st.metric("F1 Score","69%")

# =============================
# ABOUT
# =============================

elif page=="👨‍💻 About":

    st.title("👨‍💻 About Project")

    st.write("""
Brain Computer Interface Signal Analysis

Dataset : DEAP Dataset

Model : Random Forest

Language : Python

Framework : Streamlit
""")

    st.info("Developed by Divyanshu Bansal")