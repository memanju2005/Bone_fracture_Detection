import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
from PIL import Image
from predictions import predict

# Ensure the temp directory exists
if not os.path.exists("temp"):
    os.makedirs("temp")

st.set_page_config(page_title="Bone Fracture Detection", layout="centered", page_icon="🦴")

# --- CUSTOM CSS FOR STRUCTURE & STYLE ---
st.markdown("""
    <style>
    .header-style {
        text-align: center;
        padding-bottom: 1rem;
        border-bottom: 2px solid #ccc;
        margin-bottom: 2rem;
    }
    .footer-style {
        text-align: center;
        margin-top: 4rem;
        padding-top: 1rem;
        border-top: 1px solid #ccc;
        font-size: 0.95rem;
        color: #555;
    }
    .creator-text {
        color: #2E86C1;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# --- HEADER SECTION ---
st.markdown('<div class="header-style">', unsafe_allow_html=True)
st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image of a bone to analyze and detect fractures.", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# --- UPLOAD SECTION ---
st.markdown("### 📤 Upload Image")
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "png", "jpeg"])

st.markdown("---")

if uploaded_file is not None:
    # --- RESULT SECTION ---
    st.markdown("### 🔍 Analysis Results")
    col1, col2 = st.columns([1, 1], gap="large")
    
    with col1:
        st.subheader("Uploaded X-Ray")
        image = Image.open(uploaded_file)
        st.image(image, use_container_width=True, caption="Selected Image")
    
    # Save the file temporarily because Keras load_img expects a file path
    temp_path = os.path.join("temp", uploaded_file.name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    with col2:
        st.subheader("Prediction")
        with st.spinner("Analyzing image..."):
            # 1. Predict Bone Type (Elbow, Hand, Shoulder)
            bone_type = predict(temp_path, model="Parts")
            
            # 2. Predict Fracture or Normal
            fracture_status = predict(temp_path, model=bone_type)
            
        st.info(f"**Bone Type Identified:** {bone_type}")
        
        if fracture_status.lower() == 'fractured':
            st.error(f"**Diagnosis:** {fracture_status.capitalize()} 🚨")
        else:
            st.success(f"**Diagnosis:** {fracture_status.capitalize()} ✅")
            
    # Clean up the temporary file
    try:
        os.remove(temp_path)
    except Exception as e:
        pass

# --- FOOTER SECTION ---
st.markdown("""
    <div class="footer-style">
        <p>Project is created by <span class="creator-text">Manjunath kotabagi</span></p>
    </div>
""", unsafe_allow_html=True)
