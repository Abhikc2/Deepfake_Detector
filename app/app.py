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
from torchvision import transforms

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
        --bg-primary: #0a0a0f;
        --bg-secondary: #12121a;
        --bg-card: #1a1a2e;
        --bg-card-hover: #222240;
        --accent-blue: #4f8cff;
        --accent-purple: #7c3aed;
        --accent-gradient: linear-gradient(135deg, #4f8cff 0%, #7c3aed 50%, #ec4899 100%);
        --text-primary: #f0f0f5;
        --text-secondary: #a0a0b5;
        --text-muted: #6b6b80;
        --border-subtle: rgba(255, 255, 255, 0.06);
        --success: #10b981;
        --danger: #ef4444;
        --warning: #f59e0b;
        --shadow-glow: 0 0 40px rgba(79, 140, 255, 0.15);
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ── Header ────────────────────────────────────────────── */
    .app-header {
        text-align: center;
        padding: 2rem 0 1.5rem;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {
        font-size: 3rem;
        font-weight: 800;
        background: var(--accent-gradient);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: -0.02em;
    }
    .app-header p {
        color: var(--text-secondary);
        font-size: 1.1rem;
        font-weight: 300;
        letter-spacing: 0.04em;
    }

    /* ── Cards ──────────────────────────────────────────────── */
    .glass-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        backdrop-filter: blur(12px);
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        background: var(--bg-card-hover);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ── Result badges ─────────────────────────────────────── */
    .result-real {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #10b981;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }
    .result-fake {
        background: linear-gradient(135deg, #7f1d1d, #991b1b);
        border: 1px solid #ef4444;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
    }
    .result-label {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 0.5rem 0;
    }
    .result-confidence {
        font-size: 1.2rem;
        color: var(--text-secondary);
    }

    /* ── Progress indicators ───────────────────────────────── */
    .step-indicator {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: var(--text-primary);
    }
    .step-active {
        background: rgba(79, 140, 255, 0.1);
        border-left: 3px solid var(--accent-blue);
    }
    .step-done {
        background: rgba(16, 185, 129, 0.1);
        border-left: 3px solid var(--success);
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

    /* ── Metric cards ──────────────────────────────────────── */
    .metric-row {
        display: flex;
        gap: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        flex: 1;
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    .metric-card .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: var(--accent-blue);
    }
    .metric-card .metric-label {
        font-size: 0.8rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }

    /* ── Animations ────────────────────────────────────────── */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    .pulse { animation: pulse 2s ease-in-out infinite; }

    /* ── File uploader ─────────────────────────────────────── */
    .stFileUploader > div {
        border: 2px dashed rgba(79, 140, 255, 0.3) !important;
        border-radius: 16px !important;
        background: rgba(79, 140, 255, 0.03) !important;
        transition: all 0.3s ease;
    }
    .stFileUploader > div:hover {
        border-color: rgba(79, 140, 255, 0.6) !important;
        background: rgba(79, 140, 255, 0.06) !important;
    }

    /* ── Disclaimer ────────────────────────────────────────── */
    .disclaimer {
        background: rgba(245, 158, 11, 0.08);
        border: 1px solid rgba(245, 158, 11, 0.2);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        font-size: 0.82rem;
        color: #fbbf24;
        line-height: 1.5;
    }

    /* ── Hide streamlit branding ───────────────────────────── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
        st.sidebar.success("✅ Trained model loaded")
    else:
        st.sidebar.warning("⚠️ No trained model found — using untrained weights (demo mode)")

    model.eval()
    return model, device


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

    progress = st.progress(0, text="🔍 Initializing pipeline…")

    # Step 1: Extract frames
    t0 = time.time()
    progress.progress(10, text="🎬 Extracting frames from video…")

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
        progress.progress(pct, text=f"🤖 Processing frame {i+1}/{len(frames)} — YOLO + Face Detection…")

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
    progress.progress(85, text="🧠 Running AI classification…")

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

    progress.progress(100, text="✅ Analysis complete!")
    time.sleep(0.5)
    progress.empty()

    return result


# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 1rem 0;">
            <span style="font-size: 2.5rem;">🛡️</span>
            <h2 style="margin: 0.3rem 0; background: linear-gradient(135deg, #4f8cff,#7c3aed);
                -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                DeepGuard AI
            </h2>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 📖 How It Works")
        st.markdown("""
        <div class="glass-card" style="font-size:0.85rem; line-height:1.6;">
        <strong>1.</strong> Upload a video file<br>
        <strong>2.</strong> Frames are extracted & sampled<br>
        <strong>3.</strong> <span style="color:#4f8cff">YOLOv8</span> detects persons<br>
        <strong>4.</strong> <span style="color:#7c3aed">MTCNN</span> extracts faces<br>
        <strong>5.</strong> <span style="color:#ec4899">CNN+LSTM</span> classifies the sequence<br>
        <strong>6.</strong> Verdict: <span style="color:#10b981">Real</span> or <span style="color:#ef4444">Fake</span>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### 🏗️ Architecture")
        st.markdown("""
        <div class="glass-card" style="font-size:0.82rem;">
        <code>
        Video → OpenCV<br>
          → YOLOv8 (Person Filter)<br>
          → MTCNN (Face Extract)<br>
          → ResNet18 (Spatial)<br>
          → LSTM (Temporal)<br>
          → Classification
        </code>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### ⚙️ Settings")
        device = "🟢 CUDA (GPU)" if torch.cuda.is_available() else "🔵 CPU"
        st.markdown(f"**Device:** {device}")
        st.markdown(f"**Sequence Length:** {SEQUENCE_LENGTH} frames")
        st.markdown(f"**Sample Rate:** Every {SAMPLE_RATE}th frame")
        st.markdown(f"**Max File Size:** {MAX_FILE_SIZE_MB} MB")

        st.divider()

        st.markdown("""
        <div class="disclaimer">
            ⚠️ <strong>Disclaimer:</strong> This tool is for educational and
            research purposes only. Results should not be used as definitive
            proof. No uploaded data is permanently stored. All temporary files
            are deleted after processing.
        </div>
        """, unsafe_allow_html=True)


# ─── Main Content ────────────────────────────────────────────────────────────
def render_main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1>🛡️ DeepGuard AI</h1>
        <p>Detect AI-generated deepfake videos with state-of-the-art deep learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div style="text-align:center; margin-bottom:0.5rem;">
            <span style="font-size:1rem; color:var(--text-secondary);">
                Upload a video to analyze
            </span>
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload Video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
            help="Supported formats: MP4, AVI, MOV • Max size: 200 MB",
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
            left, right = st.columns([3, 2])

            with left:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">📹 Video Preview</h3>
                </div>
                """, unsafe_allow_html=True)
                st.video(tmp_path)

            with right:
                st.markdown("""
                <div class="glass-card">
                    <h3 style="color:var(--text-primary); margin-top:0;">📊 Video Information</h3>
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
                    "🔍  Analyze Video for Deepfakes",
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
                    color = "#10b981" if is_real else "#ef4444"
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
                    <h3 style="color:var(--text-primary); margin-top:0;">⚡ Processing Details</h3>
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
                    with st.expander("👁️ Detected Faces (click to expand)", expanded=False):
                        cols = st.columns(min(5, len(result["face_images"])))
                        for idx, face_img in enumerate(result["face_images"][:10]):
                            with cols[idx % len(cols)]:
                                pil_face = Image.fromarray(face_img)
                                st.image(pil_face, caption=f"Frame {idx+1}", width=120)

        finally:
            # Clean up temp files
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except Exception:
                pass


# ─── Run App ─────────────────────────────────────────────────────────────────
render_sidebar()
render_main()
