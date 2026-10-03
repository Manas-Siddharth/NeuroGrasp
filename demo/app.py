import os
import sys
import time
import math
import numpy as np
from PIL import Image
import cv2
import streamlit as st

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.inference import GraspPredictor
from src.utils import draw_grasp

st.set_page_config(
    page_title="NeuroGrasp Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 NeuroGrasp: Deep Learning-Based Robotic Grasp Detection")
st.markdown("""
Predict feasible planar robotic grasp parameters $(x, y, w, h, \\theta, \\text{confidence})$ from RGB camera images in real-time.
""")

st.sidebar.header("⚙️ Model Configuration")
checkpoint_path = st.sidebar.text_input("Checkpoint Path", value="checkpoints/best_model.pth")
conf_threshold = st.sidebar.slider("Confidence Display Threshold", 0.0, 1.0, 0.1, 0.05)

@st.cache_resource
def load_predictor(path):
    if not os.path.exists(path):
        return None
    return GraspPredictor(checkpoint_path=path)

predictor = load_predictor(checkpoint_path)

if predictor is None:
    st.error(f"❌ Model checkpoint not found at `{checkpoint_path}`. Please train the model first using `python -m src.train`.")
    st.stop()

st.sidebar.success("✅ Model loaded successfully!")

# Upload / Sample selection
col_upload, col_sample = st.columns(2)

with col_upload:
    uploaded_file = st.file_uploader("Upload an RGB Image", type=["jpg", "png", "jpeg"])

with col_sample:
    sample_dir = "data/sample_dataset"
    sample_images = []
    if os.path.exists(sample_dir):
        sample_images = [f for f in os.listdir(sample_dir) if f.endswith("_RGB.png")]
    
    selected_sample = st.selectbox("Or select a Sample Dataset Image", options=["None"] + sample_images)

# Determine input image
input_image_pil = None
image_source_label = ""

if uploaded_file is not None:
    input_image_pil = Image.open(uploaded_file).convert("RGB")
    image_source_label = f"Uploaded: {uploaded_file.name}"
elif selected_sample != "None":
    sample_path = os.path.join(sample_dir, selected_sample)
    input_image_pil = Image.open(sample_path).convert("RGB")
    image_source_label = f"Sample: {selected_sample}"

if input_image_pil is not None:
    st.subheader(f"🖼️ Input Image: {image_source_label}")
    
    col_raw, col_pred = st.columns(2)

    with col_raw:
        st.image(input_image_pil, caption="Raw Input RGB Image", use_container_width=True)

    with st.spinner("🤖 Computing grasp prediction..."):
        t_start = time.time()
        result = predictor.predict(input_image_pil)
        latency_ms = (time.time() - t_start) * 1000.0

    with col_pred:
        annotated_rgb = result["annotated_image"]
        st.image(annotated_rgb, caption="Predicted Robotic Grasp Overlay", use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Predicted Grasp Parameters")

    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Grasp Center X", f"{result['x']:.1f} px")
    m2.metric("Grasp Center Y", f"{result['y']:.1f} px")
    m3.metric("Gripper Width (w)", f"{result['width']:.1f} px")
    m4.metric("Gripper Height (h)", f"{result['height']:.1f} px")
    m5.metric("Grasp Angle (θ)", f"{result['angle_deg']:.1f}°")
    m6.metric("Confidence Score", f"{result['confidence']:.4f}")
    m7.metric("Inference Latency", f"{latency_ms:.1f} ms")

    if result['confidence'] >= conf_threshold:
        st.success(f"✅ Feasible Grasp Detected! Confidence {result['confidence']:.2f} >= threshold {conf_threshold:.2f}")
    else:
        st.warning(f"⚠️ Low Confidence Grasp Detected! Confidence {result['confidence']:.2f} < threshold {conf_threshold:.2f}")

else:
    st.info("👈 Please upload an RGB image or select a sample image from the options above to detect robotic grasps.")
