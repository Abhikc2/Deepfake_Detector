"""
Deepfake Detector — Streamlit Application.

A premium, dark-themed web app that lets users upload videos and
detect whether they are real or AI-generated deepfakes using a
CNN+LSTM model with YOLO person detection and MTCNN face extraction.

Run:
    streamlit run app/app.py
"""

import os
import sys
import time
import tempfile
import logging
import base64
from pathlib import Path

import streamlit as st
import cv2
import numpy as np
import torch
from PIL import Image

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.video_to_frames import extract_frames, validate_video_file, get_video_info
from utils.yolo_detector import YOLOPersonDetector
from utils.face_extractor import FaceExtractor
from models.cnn_lstm import DeepfakeDetector
from app.image_detector import detect_image
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ─── Config ──────────────────────────────────────────────────────────────────
APP_TITLE = "🛡️ DeepGuard AI"
APP_SUBTITLE = "Advanced Deepfake Detection System"
MAX_FILE_SIZE_MB = 200
SEQUENCE_LENGTH = 15
SAMPLE_RATE = 5
MAX_FRAMES = 80
IMAGE_SIZE = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepGuard AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ────────────────────────────────────────────── */
    :root {
        --bg-primary: #080c14;
        --bg-secondary: #0b1120;
        --bg-card: #111827;
        --bg-card-hover: #1a2540;
        --accent-blue: #0ea5e9;
        --accent-blue-dark: #0369a1;
        --accent-blue-light: #38bdf8;
        --accent-gradient: linear-gradient(135deg, #0ea5e9 0%, #0369a1 50%, #023e6b 100%);
        --text-primary: #ffffff;
        --text-secondary: #cccccc;
        --text-muted: #94a3b8;
        --border-subtle: rgba(14, 165, 233, 0.15);
        --border-accent: rgba(14, 165, 233, 0.3);
        --success: #22c55e;
        --danger: #ef4444;
        --warning: #f59e0b;
        --shadow-accent: 0 0 30px rgba(14, 165, 233, 0.2);
        --shadow-subtle: 0 4px 16px rgba(0, 0, 0, 0.5);
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
        color: var(--text-primary) !important;
    }

    /* ── Header ────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 1rem 0 1.5rem;
        margin-bottom: 2rem;
        border-bottom: 2px solid var(--border-accent);
    }
    .app-header .logo-img {
        width: 160px;
        height: 160px;
        margin: 0 auto 0.5rem;
        display: block;
        filter: drop-shadow(0 0 25px rgba(14, 165, 233, 0.4));
    }
    .app-header h1 {
        font-size: 4rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
        animation: glowPulse 4.5s ease-in-out infinite;
    }
    
    @keyframes glowPulse {
    0%, 100% {
        text-shadow: 
            0 0 10px rgba(14,165,233,0.1),
            0 0 20px rgba(3,105,161,0.1);
        }
        50% {
            text-shadow: 
                0 0 20px rgba(14,165,233,0.4),
                0 0 40px rgba(3,105,161,0.3);
        }
    }

    .app-header p {
        color: var(--text-secondary);
        font-size: 1.3rem;
        font-weight: 400;
        letter-spacing: 0.05em;
    }

    /* ── Cards ──────────────────────────────────────────────── */
    .glass-card {
        border-radius: 5px;
        padding: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid var(--border-accent);
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        background: var(--bg-card-hover);
        transform: translateY(-4px);
        box-shadow: 0 10px 35px rgba(14, 165, 233, 0.2);
        border-color: var(--accent-blue);
    }

    /* ── Result badges ─────────────────────────────────────── */
    .result-real {
        background: linear-gradient(135deg, #0f3b2e, #1a5c47);
        border: 2px solid var(--success);
        border-radius: 18px;
        padding: 0.1rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
        box-shadow: 0 0 25px rgba(34, 197, 94, 0.15);
    }
    .result-fake {
        background: linear-gradient(135deg, #3b0000, #5c1a1a);
        border: 2px solid var(--danger);
        border-radius: 18px;
        padding: 0.1rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
        box-shadow: 0 0 25px rgba(239, 68, 68, 0.2);
    }
    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.1rem 0;
        letter-spacing: -0.01em;
    }
    .result-confidence {
        font-size: 1.67rem;
        color: var(--text-secondary);
        font-weight: 500;
    }

    /* ── Progress Indicators ───────────────────────────────── */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.8rem 1.2rem;
        border-radius: 12px;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        color: var(--text-primary);
        transition: all 0.3s ease;
    }
    .step-active {
        background: rgba(14, 165, 233, 0.12);
        border-left: 4px solid var(--accent-blue);
    }
    .step-done {
        background: rgba(34, 197, 94, 0.12);
        border-left: 4px solid var(--success);
        color: var(--text-secondary);
    }

    /* ── Sidebar ───────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
        border-right: 1px solid var(--border-subtle);
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
    }

    /* ── Buttons ───────────────────────────────────────────── */
    .stButton > button {
        background: var(--accent-gradient) !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        padding: 0.7rem 1.5rem !important;
        border-radius: 12px !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        box-shadow: 0 4px 15px rgba(14, 165, 233, 0.25) !important;
        letter-spacing: 0.02em !important;
    }
    .stButton > button:hover {
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.4) !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: 0 2px 10px rgba(14, 165, 233, 0.3) !important;
    }

    /* ── Metric cards ──────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1.2rem;
        margin: 1.2rem 0;
    }
    .metric-card {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 14px;
        padding: 1.2rem;
        text-align: center;
        transition: all 0.35s ease;
        box-shadow: var(--shadow-subtle);
    }
    .metric-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-3px);
        box-shadow: var(--shadow-accent);
    }
    .metric-card .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .metric-card .metric-label {
        font-size: 0.75rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.4rem;
        font-weight: 600;
    }

    /* ── Animations ────────────────────────────────────────── */
    @keyframes fadeIn {
        from { 
            opacity: 0; 
            transform: translateY(20px);
        }
        to { 
            opacity: 1; 
            transform: translateY(0);
        }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }
    @keyframes glow {
        0%, 100% { box-shadow: 0 0 20px rgba(14, 165, 233, 0.2); }
        50% { box-shadow: 0 0 30px rgba(14, 165, 233, 0.4); }
    }
    .pulse { animation: pulse 2s ease-in-out infinite; }
    .glow { animation: glow 2s ease-in-out infinite; }

    /* ── File uploader ─────────────────────────────────────── */
    .stFileUploader > div {
        border: 2px dashed var(--border-accent) !important;
        border-radius: 16px !important;
        background: rgba(14, 165, 233, 0.04) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        padding: 2rem !important;
    }
    .stFileUploader > div:hover {
        border-color: var(--accent-blue) !important;
        background: rgba(14, 165, 233, 0.08) !important;
        box-shadow: var(--shadow-accent) !important;
    }

    /* ── Input fields ──────────────────────────────────────── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea,
    .stSelectbox > div > div > select {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        color: var(--text-primary) !important;
        transition: all 0.3s ease !important;
    }
    .stTextInput > div > div > input:hover,
    .stTextArea > div > div > textarea:hover,
    .stSelectbox > div > div > select:hover {
        border-color: var(--border-accent) !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus,
    .stSelectbox > div > div > select:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: 0 0 0 2px rgba(14, 165, 233, 0.1) !important;
    }

    /* ── Disclaimer ────────────────────────────────────────── */
    .disclaimer {
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid var(--border-accent);
        border-radius: 12px;
        padding: 1.2rem;
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.6;
        box-shadow: var(--shadow-subtle);
    }
            
    /* ── Sidebar Card (Unified Style) ───────────────────────── */
    .sidebar-card {
        background: rgba(14, 165, 233, 0.06);
        border: 1px solid var(--border-accent);
        border-radius: 16px;
        padding: 1.2rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 6px 25px rgba(14, 165, 233, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .sidebar-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 35px rgba(14, 165, 233, 0.2);
        border-color: var(--accent-blue);
    }

    /* Center headings */
    .sidebar-title {
        text-align: center;
        font-weight: 800;
        margin-bottom: 0.8rem;
        font-size: 1.4rem;
    }

    /* Optional: smaller text */
    .sidebar-content {
        font-size: 0.95rem;
        line-height: 1.6;
        text-align: center;
        color: var(--text-secondary);
    }

    /* ── Expander ──────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    .streamlit-expanderHeader:hover {
        border-color: var(--border-accent) !important;
        background: var(--bg-card-hover) !important;
    }

    /* ── Tab styling ───────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        border-bottom: 2px solid var(--border-subtle);
    }
    .stTabs [role="tablist"] button {
        color: var(--text-muted) !important;
    }
    .stTabs [role="tablist"] button[aria-selected="true"] {
        color: var(--accent-blue) !important;
        border-bottom: 3px solid var(--accent-blue) !important;
    }

    /* ── Hide streamlit branding ───────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ── Text styling enhancements ────────────────────────── */
    h1, h2, h3, h4, h5, h6 {
        color: var(--text-primary) !important;
    }

    p, span, div {
        color: inherit !important;
    }
            
    .gradient-text {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(135deg, #38bdf8, #0ea5e9, #0369a1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ── Sidebar logo ──────────────────────────────────────── */
    .sidebar-logo {
        display: block;
        margin: 0 auto;
        width: 120px;
        height: 120px;
        filter: drop-shadow(0 0 15px rgba(14, 165, 233, 0.35));
    }
            
    /* ── Hide scrollbar ─────────────────────────────────────── */
    ::-webkit-scrollbar {
        display: none;
    }

    * {
        scrollbar-width: none;
    }
            
    /* ── Fix Streamlit video size ─────────────────────────────────────── */
    video {
        width: 100% !important;
        max-width: 560px !important;
        height: 430px !important;
        object-fit: contain !important;
        border-radius: 6px !important;
        display: block;
        margin: auto;
    }
            
    /* ── Upload box ─────────────────────────────────────── */
    .stFileUploader > div {
        padding: 0.35rem !important;
    }

    /* ── Top drag area ─────────────────────────────────────── */
    .stFileUploader [data-testid="stFileUploadDropzone"] {
        border: 1px dashed rgba(14,165,233,0.3) !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Uploaded file box  ─────────────────────────────────────── */
    .stFileUploader [data-testid="stFileUploadDropzone"] + div {
        margin-top: 0.8rem !important;
        background: rgba(14,165,233,0.05);
        border: 1px solid rgba(14,165,233,0.2);
        border-radius: 12px;
        padding: 0.6rem 0.8rem;
    }
    
    /* ── Hide sidebar collapse button ──────────────────────────── */
    button[kind="headerNoPadding"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #38bdf8, #0ea5e9) !important;
        box-shadow: 0 0 8px rgba(14,165,233,0.6);
    }
    
    hr {
        position: relative;
        border: none;
        height: 2px;
        overflow: hidden;
        background: rgba(14,165,233,0.2);
    }

    /* ── Moving glow strip ─────────────────────────────────────── */
    hr::after {
        content: "";
        position: absolute;
        top: 0;
        left: -40%;
        width: 40%;
        height: 100%;
        background: linear-gradient(
            90deg,
            transparent,
            #0ea5e9,
            #ffffff,
            #0ea5e9,
            transparent
        );
        filter: blur(4px);
        animation: moveGlow 2.5s linear infinite;
    }

    /* Animation */
    @keyframes moveGlow {
        0% { left: -40%; }
        100% { left: 100%; }
    }
    
   /* ──  Hide floating link icons ─────────────────────────────────────── */
    a[href^="#"] {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #0ea5e9, transparent) !important;
        box-shadow: 0 0 8px rgba(14, 165, 233, 0.5) !important;
        margin-top: -0.1rem !important;
        margin-bottom: 1.3rem !important;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cached Model Loaders ───────────────────────────────────────────────────
@st.cache_resource
def load_yolo_detector():
    """Load and cache YOLO person detector."""
    return YOLOPersonDetector(model_name="yolov8n.pt", confidence_threshold=0.5)


@st.cache_resource
def load_face_extractor():
    """Load and cache MTCNN face extractor."""
    return FaceExtractor(target_size=(IMAGE_SIZE, IMAGE_SIZE))


@st.cache_resource
def load_deepfake_model():
    """
    Load the CNN+LSTM model.
    If a trained checkpoint exists, load its weights; otherwise use random weights.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepfakeDetector(
        backbone="resnet18",
        pretrained=True,
        feature_dim=512,
        lstm_hidden=256,
        lstm_layers=2,
        num_classes=2,
    ).to(device)

    checkpoint_path = Path(__file__).resolve().parent.parent / "checkpoints" / "best_model.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        st.sidebar.success("Trained model loaded")
    else:
        st.sidebar.warning("No trained model found — using untrained weights")

    model.eval()
    return model, device


@st.cache_resource
def load_hf_image_model():
    """Load the pretrained Image Classifier from HuggingFace."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = "dima806/deepfake_vs_real_image_detection"

    try:
        processor = AutoImageProcessor.from_pretrained(model_id)
        model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
        model.eval()
        st.sidebar.success("Deepfake Image Model loaded ✓")
    except Exception as e:
        st.sidebar.error(f"Failed to load image model: {e}")
        raise

    return model, processor, device


def get_inference_transforms():
    """Get transforms for inference."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─── Inference Pipeline ─────────────────────────────────────────────────────
def run_inference(video_path: str) -> dict:
    """
    Full inference pipeline:
      Video → Frames → YOLO → MTCNN → CNN+LSTM → Prediction.

    Returns dict with prediction, confidence, faces, and timing info.
    """
    result = {
        "prediction": None,
        "confidence": 0.0,
        "real_prob": 0.0,
        "fake_prob": 0.0,
        "num_frames": 0,
        "num_faces": 0,
        "face_images": [],
        "timings": {},
    }

    # Load models
    yolo = load_yolo_detector()
    face_ext = load_face_extractor()
    model, device = load_deepfake_model()
    transform = get_inference_transforms()

    progress = st.progress(0, text="Initializing pipeline…")

    # Step 1: Extract frames
    t0 = time.time()
    progress.progress(10, text="Extracting frames from video…")

    frames = extract_frames(
        video_path,
        output_dir=None,
        sample_rate=SAMPLE_RATE,
        max_frames=MAX_FRAMES,
    )
    result["num_frames"] = len(frames)
    result["timings"]["frame_extraction"] = time.time() - t0

    if len(frames) == 0:
        progress.progress(100, text="❌ No frames extracted")
        return result

    # Step 2: YOLO + MTCNN → face crops
    t1 = time.time()
    face_tensors = []

    for i, frame in enumerate(frames):
        pct = 20 + int(60 * (i / len(frames)))
        progress.progress(pct, text=f"Processing frame {i+1}/{len(frames)} — YOLO + Face Detection…")

        # YOLO person detection
        person_crops = yolo.detect_persons(frame)
        if not person_crops:
            person_crops = [frame]

        # MTCNN face extraction
        for crop in person_crops:
            face = face_ext.extract_face(crop)
            if face is not None:
                result["face_images"].append(face.copy())
                face_tensor = transform(face)
                face_tensors.append(face_tensor)
                break  # one face per frame

    result["num_faces"] = len(face_tensors)
    result["timings"]["detection"] = time.time() - t1

    if len(face_tensors) == 0:
        progress.progress(100, text="❌ No faces detected in the video")
        return result

    # Step 3: Build sequence and run model
    t2 = time.time()
    progress.progress(85, text="Running AI classification…")

    # Pad or truncate to SEQUENCE_LENGTH
    if len(face_tensors) < SEQUENCE_LENGTH:
        while len(face_tensors) < SEQUENCE_LENGTH:
            face_tensors.append(face_tensors[-1])
    face_tensors = face_tensors[:SEQUENCE_LENGTH]

    # Stack into tensor: (1, seq_len, C, H, W)
    sequence = torch.stack(face_tensors).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = model.predict_proba(sequence)

    real_prob = probs[0][0].item()
    fake_prob = probs[0][1].item()

    result["real_prob"] = real_prob
    result["fake_prob"] = fake_prob
    result["prediction"] = "REAL" if real_prob > fake_prob else "FAKE"
    result["confidence"] = max(real_prob, fake_prob)
    result["timings"]["inference"] = time.time() - t2
    result["timings"]["total"] = time.time() - t0

    progress.progress(100, text="Analysis complete!")
    time.sleep(0.5)
    progress.empty()

    return result


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def _get_logo_base64():
    """Load logo and return base64 string for embedding."""
    logo_path = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
    if logo_path.exists():
        with open(logo_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


def render_sidebar():
    with st.sidebar:
        logo_b64 = _get_logo_base64()
        if logo_b64:
            st.markdown(f"""
            <div style="text-align:center; padding: 0.3rem 0;">
                <img src="data:image/png;base64,{logo_b64}" class="sidebar-logo" alt="DeepGuard AI">
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center; padding: 0.1rem 0;">
            <h2 class="gradient-text" style="margin: 0.1rem 0; font-size: 2.3rem;">
                DeepGuard AI
            </h2>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div class="sidebar-card">
        <div class="sidebar-title">[⁉️]  How It Works</div>
        <div class="sidebar-content">
            <strong>🎬 Video:</strong> Upload → Frames → YOLO → MTCNN → CNN+LSTM → Verdict<br><br>
            <strong>🖼️ Image:</strong> Upload → HuggingFace Model → Verdict
        </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()

        st.markdown("""
        <div class="sidebar-card">
        <div class="sidebar-title">[🏗️]  Architecture</div>
        <div class="sidebar-content">
            <strong>Video:</strong> Input → YOLO → MTCNN<br>
            → ResNet18 → LSTM → Verdict<br><br>
            <strong>Image:</strong> Full Image<br>
            → Deepfake Image Classifier → Verdict
        </div>
        </div>
    """, unsafe_allow_html=True)

        st.divider()

        device = "🟢 CUDA (GPU)" if torch.cuda.is_available() else "🔵 CPU"
        st.markdown(f"""
        <div class="sidebar-card">
        <div class="sidebar-title">[⚙️]  Settings</div>
        <div class="sidebar-content">
            <strong>Device:</strong> {device}<br>
            <strong>Sequence Length:</strong> {SEQUENCE_LENGTH}<br>
            <strong>Sample Rate:</strong> Every {SAMPLE_RATE}th frame<br>
            <strong>Max File Size:</strong> {MAX_FILE_SIZE_MB} MB
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("""
        <div class="sidebar-card">
        <div class="sidebar-title">[⚠️]  Disclaimer</div>
        <div class="sidebar-content">
            This tool is for educational and research purposes only. Results should not be used as definitive proof. No uploaded data is permanently stored. 
        </div>
        </div>
        """, unsafe_allow_html=True)

# ─── Main Content ────────────────────────────────────────────────────────────
def render_image_tab():
    """Render the Image Detection tab content."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin-bottom:0.5rem;">
            <span style="font-size:1.5rem; color:var(--text-secondary);">
                Upload an image to analyze...
            </span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_image = st.file_uploader(
            "Upload Image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
            help="Supported formats: JPG, JPEG, PNG",
            key="image_uploader",
        )

    if uploaded_image is not None:
        # Save to temp file
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_image.name)

        with open(tmp_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        try:
            # Preview
            pil_preview = Image.open(tmp_path)
            img_width, img_height = pil_preview.size
            file_size_kb = os.path.getsize(tmp_path) / 1024

            left, right = st.columns([2, 3])

            with left:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">Image Preview</h3>
                </div>
                """, unsafe_allow_html=True)
                st.image(pil_preview, use_container_width=True)

            with right:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">Image Information</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{img_width}×{img_height}</div>
                        <div class="metric-label">Resolution</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{file_size_kb:.0f} KB</div>
                        <div class="metric-label">File Size</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{uploaded_image.name.rsplit('.', 1)[-1].upper()}</div>
                        <div class="metric-label">Format</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                analyze_img_btn = st.button(
                    "🔍 Analyze Image for Deepfakes",
                    use_container_width=True,
                    type="primary",
                    key="analyze_image_btn",
                )

            if analyze_img_btn:
                st.divider()

                # Load model
                model, processor, device = load_hf_image_model()

                with st.spinner("Analyzing image with Deepfake Classifier…"):
                    result = detect_image(
                        model=model,
                        processor=processor,
                        image_path=tmp_path,
                        device=device,
                    )

                if result["error"]:
                    st.error(f"❌ **Error:** {result['error']}")
                    return

                if result["prediction"] is None:
                    st.error("❌ **Analysis failed.** Could not process the image.")
                    return

                # ── Display Results ──────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                with res_col2:
                    is_real = result["prediction"] == "REAL"
                    css_class = "result-real" if is_real else "result-fake"
                    emoji = "✅" if is_real else "🚨"
                    color = "#0FC52A" if is_real else "#ef4444"
                    label = result["prediction"]
                    conf = result["confidence"] * 100

                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="font-size:3rem;">{emoji}</div>
                        <div class="result-label" style="color:{color};">{label}</div>
                        <div class="result-confidence">
                            Confidence: <strong>{conf:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Fallback notice ──────────────────────────────────
                if result["used_full_image"]:
                    st.info("ℹ️ No face detected — analysis was performed on the full image.")

                # ── Probability breakdown ────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                prob_left, prob_right = st.columns(2)
                with prob_left:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-muted);
                            text-transform:uppercase; letter-spacing:0.05em;">
                            Real Probability
                        </div>
                        <div style="font-size:2rem; font-weight:700; color:#10b981;
                            margin:0.5rem 0;">
                            {result['real_prob']*100:.1f}%
                        </div>
                        <div style="background:rgba(16,185,129,0.1); border-radius:999px;
                            height:8px; overflow:hidden;">
                            <div style="background:#10b981; height:100%;
                                width:{result['real_prob']*100}%;
                                border-radius:999px; transition: width 1s ease;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with prob_right:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-muted);
                            text-transform:uppercase; letter-spacing:0.05em;">
                            Fake Probability
                        </div>
                        <div style="font-size:2rem; font-weight:700; color:#ef4444;
                            margin:0.5rem 0;">
                            {result['fake_prob']*100:.1f}%
                        </div>
                        <div style="background:rgba(239,68,68,0.1); border-radius:999px;
                            height:8px; overflow:hidden;">
                            <div style="background:#ef4444; height:100%;
                                width:{result['fake_prob']*100}%;
                                border-radius:999px; transition: width 1s ease;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Processing stats ─────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">[⚡] Processing Details</h3>
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{result['num_faces']}</div>
                            <div class="metric-label">Faces Analyzed</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('total', 0):.2f}s</div>
                            <div class="metric-label">Total Time</div>
                        </div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('face_extraction', 0):.2f}s</div>
                            <div class="metric-label">Face Extraction</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('inference', 0):.2f}s</div>
                            <div class="metric-label">Model Inference</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Face gallery ─────────────────────────────────────
                if result["face_images"] and not result["used_full_image"]:
                    with st.expander("[👁️]  Detected Faces (click to expand)", expanded=False):
                        cols = st.columns(min(5, len(result["face_images"])))
                        for idx, face_img in enumerate(result["face_images"][:10]):
                            with cols[idx % len(cols)]:
                                pil_face = Image.fromarray(face_img)
                                st.image(pil_face, caption=f"Face {idx+1}", use_container_width=True)

        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass


def render_video_tab():
    """Render the Video Detection tab content (original pipeline)."""
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin-bottom:0.5rem;">
            <span style="font-size:1.5rem; color:var(--text-secondary);">
                Upload a video to analyze...
            </span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            help="Supported formats: MP4, AVI, MOV",
            key="video_uploader",
        )

    if uploaded_file is not None:
        # Save to temp file
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)

        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        try:
            # Validate
            is_valid, msg = validate_video_file(tmp_path)

            if not is_valid:
                st.error(f"❌ {msg}")
                return

            # Video info
            info = get_video_info(tmp_path)

            # Layout: video preview + info
            left, right = st.columns([2, 3])

            with left:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">Video Preview</h3>
                </div>
                """, unsafe_allow_html=True)
                st.video(tmp_path)

            with right:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">Video Information</h3>
                </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{info['frame_count']}</div>
                        <div class="metric-label">Total Frames</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{info['fps']:.0f}</div>
                        <div class="metric-label">FPS</div>
                    </div>
                </div>
                <div class="metric-row">
                    <div class="metric-card">
                        <div class="metric-value">{info['width']}×{info['height']}</div>
                        <div class="metric-label">Resolution</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">{info['duration_sec']:.1f}s</div>
                        <div class="metric-label">Duration</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                file_size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
                st.markdown(f"""
                <div class="metric-row">
                    <div class="metric-card" style="flex:1;">
                        <div class="metric-value">{file_size_mb:.1f} MB</div>
                        <div class="metric-label">File Size</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Analyze button
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
            with col_btn2:
                analyze_btn = st.button(
                    "Analyze Video for Deepfakes",
                    use_container_width=True,
                    type="primary",
                )

            if analyze_btn:
                st.divider()

                # Run inference
                result = run_inference(tmp_path)

                if result["prediction"] is None:
                    if result["num_faces"] == 0:
                        st.error("❌ **No faces detected.** The video must contain visible human faces for analysis.")
                    else:
                        st.error("❌ **Analysis failed.** The video could not be processed.")
                    return

                # ── Display Results ──────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                with res_col2:
                    is_real = result["prediction"] == "REAL"
                    css_class = "result-real" if is_real else "result-fake"
                    emoji = "✅" if is_real else "🚨"
                    color = "#0FC52A" if is_real else "#ef4444"
                    label = result["prediction"]
                    conf = result["confidence"] * 100

                    st.markdown(f"""
                    <div class="{css_class}">
                        <div style="font-size:3rem;">{emoji}</div>
                        <div class="result-label" style="color:{color};">{label}</div>
                        <div class="result-confidence">
                            Confidence: <strong>{conf:.1f}%</strong>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Probability breakdown ────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                prob_left, prob_right = st.columns(2)
                with prob_left:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-muted);
                            text-transform:uppercase; letter-spacing:0.05em;">
                            Real Probability
                        </div>
                        <div style="font-size:2rem; font-weight:700; color:#10b981;
                            margin:0.5rem 0;">
                            {result['real_prob']*100:.1f}%
                        </div>
                        <div style="background:rgba(16,185,129,0.1); border-radius:999px;
                            height:8px; overflow:hidden;">
                            <div style="background:#10b981; height:100%;
                                width:{result['real_prob']*100}%;
                                border-radius:999px; transition: width 1s ease;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with prob_right:
                    st.markdown(f"""
                    <div class="glass-card" style="text-align:center;">
                        <div style="font-size:0.85rem; color:var(--text-muted);
                            text-transform:uppercase; letter-spacing:0.05em;">
                            Fake Probability
                        </div>
                        <div style="font-size:2rem; font-weight:700; color:#ef4444;
                            margin:0.5rem 0;">
                            {result['fake_prob']*100:.1f}%
                        </div>
                        <div style="background:rgba(239,68,68,0.1); border-radius:999px;
                            height:8px; overflow:hidden;">
                            <div style="background:#ef4444; height:100%;
                                width:{result['fake_prob']*100}%;
                                border-radius:999px; transition: width 1s ease;">
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Processing stats ─────────────────────────────────
                st.markdown("<br>", unsafe_allow_html=True)

                st.markdown(f"""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">[⚡] Processing Details</h3>
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{result['num_frames']}</div>
                            <div class="metric-label">Frames Sampled</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['num_faces']}</div>
                            <div class="metric-label">Faces Detected</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('total', 0):.1f}s</div>
                            <div class="metric-label">Total Time</div>
                        </div>
                    </div>
                    <div class="metric-row">
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('frame_extraction', 0):.2f}s</div>
                            <div class="metric-label">Frame Extraction</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('detection', 0):.2f}s</div>
                            <div class="metric-label">YOLO + MTCNN</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">{result['timings'].get('inference', 0):.2f}s</div>
                            <div class="metric-label">Model Inference</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # ── Face gallery (expandable) ────────────────────────
                if result["face_images"]:
                    if "selected_face" not in st.session_state:
                        st.session_state.selected_face = None

                    with st.expander("[👁️]  Detected Faces (click to expand)", expanded=False):
                        cols = st.columns(min(5, len(result["face_images"])))
                        for idx, face_img in enumerate(result["face_images"][:10]):
                            with cols[idx % len(cols)]:
                                pil_face = Image.fromarray(face_img)
                                st.image(pil_face, caption=f"Frame {idx+1}", use_container_width=True)

                    # Full-size viewer
                    if st.session_state.selected_face is not None:
                        idx = st.session_state.selected_face
                        face_img = result["face_images"][idx]
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div class="glass-card" style="text-align:center;">
                            <h3 style="color:var(--text-primary); margin-top:0;">
                                🔍 Frame {idx+1} — Full View
                            </h3>
                        </div>
                        """, unsafe_allow_html=True)

        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass


def render_main():
    # Header with logo
    logo_b64 = _get_logo_base64()
    logo_html = ""
    if logo_b64:
        logo_html = f'<img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="DeepGuard AI">'

    st.markdown(f"""
    <div class="app-header">
        {logo_html}
        <h1>DeepGuard AI</h1>
        <p>Deepfake Detection & Security</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Tabbed interface ──────────────────────────────────────────────────
    video_tab, image_tab = st.tabs(["🎬  Video Detection", "🖼️  Image Detection"])

    with video_tab:
        render_video_tab()

    with image_tab:
        render_image_tab()


# ─── Run App ─────────────────────────────────────────────────────────────────
render_sidebar()
render_main()