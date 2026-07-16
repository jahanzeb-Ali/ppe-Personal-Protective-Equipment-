import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="Object Detection", layout="wide")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# --- Sidebar: controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    conf = st.slider("Confidence threshold", 0.0, 1.0, 0.25, 0.05)

# --- Main area ---
st.title("Object Detection")

if uploaded_file is None:
    st.info("Upload an image from the sidebar to run detection.")
else:
    image = Image.open(uploaded_file).convert("RGB")

    results = model.predict(np.array(image), conf=conf, save=False)
    annotated = results[0].plot()[:, :, ::-1]   # BGR -> RGB

    # Side-by-side so output is visible without scrolling
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image, use_container_width=True)
    with col2:
        st.subheader("Detections")
        st.image(annotated, use_container_width=True)

    # Detection summary
    boxes = results[0].boxes
    names = results[0].names
    if len(boxes) == 0:
        st.warning("No objects detected. Try lowering the confidence threshold.")
    else:
        counts = {}
        for c in boxes.cls.tolist():
            label = names[int(c)]
            counts[label] = counts.get(label, 0) + 1

        st.subheader(f"Found {len(boxes)} object(s)")
        cols = st.columns(len(counts))
        for col, (label, n) in zip(cols, counts.items()):
            col.metric(label, n)