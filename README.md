# 🛡️ DeepGuard AI — Deepfake Detection System

An advanced deepfake detection system that analyzes videos to determine whether they are **real** or **AI-generated fakes** using state-of-the-art deep learning techniques.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🏗️ Architecture

```
Video → OpenCV (Frame Extraction)
         → YOLOv8 (Person Detection)
         → MTCNN (Face Extraction)
         → ResNet18 CNN (Spatial Features)
         → LSTM (Temporal Modeling)
         → Classification (Real / Fake)
```

| Component | Purpose | Technology |
|-----------|---------|-----------|
| Frame Extraction | Video → sampled frames | OpenCV |
| Person Detection | Filter relevant regions | YOLOv8 (Ultralytics) |
| Face Extraction | Detect & align faces | MTCNN (facenet-pytorch) |
| Spatial Features | Per-frame feature vectors | ResNet18 (pretrained) |
| Temporal Model | Sequence pattern analysis | LSTM (2-layer, 256-dim) |
| Classifier | Binary decision | FC layers with dropout |

---

## 📂 Project Structure

```
deepfake-detector/
├── app/
│   └── app.py                  # Streamlit web application
├── models/
│   ├── cnn.py                  # CNN feature extractor (ResNet18)
│   └── cnn_lstm.py             # Full CNN+LSTM detector
├── utils/
│   ├── video_to_frames.py      # Video ingestion & frame extraction
│   ├── yolo_detector.py        # YOLOv8 person detection
│   └── face_extractor.py       # MTCNN face extraction
├── dataset/
│   └── dataset_sequence.py     # Sequence dataset for training
├── scripts/
│   ├── preprocess.py           # End-to-end preprocessing pipeline
│   ├── train.py                # Model training script
│   └── evaluate.py             # Evaluation with metrics
├── data/
│   ├── raw/                    # Raw video files
│   ├── frames/                 # Extracted frames
│   └── faces/                  # Cropped face images
├── checkpoints/                # Saved model weights
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Web App (Demo Mode)

```bash
streamlit run app/app.py
```

> The app works out of the box in **demo mode** (untrained weights). Upload any video with a visible face to test the full pipeline.

### 3. Train a Model (Optional)

#### Step 1: Organize your dataset

```
data/raw/
├── real/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── fake/
    ├── video_101.mp4
    ├── video_102.mp4
    └── ...
```

> Recommended dataset: [FaceForensics++](https://github.com/ondyari/FaceForensics)

#### Step 2: Preprocess videos

```bash
python scripts/preprocess.py --input_dir data/raw --output_dir data/faces --sample_rate 5 --max_frames 80
```

#### Step 3: Train the model

```bash
python scripts/train.py --data_dir data/faces --epochs 15 --batch_size 8 --lr 1e-4
```

#### Step 4: Evaluate

```bash
python scripts/evaluate.py --data_dir data/faces --checkpoint checkpoints/best_model.pth
```

---

## 🔐 Security

| Feature | Status |
|---------|--------|
| File type validation | ✅ |
| File size limit (200 MB) | ✅ |
| Temp file auto-deletion | ✅ |
| No permanent data storage | ✅ |
| Frame count limits | ✅ |
| No user code execution | ✅ |

---

## 🧠 Model Details

- **CNN Backbone:** ResNet18 (ImageNet pretrained) → 512-dim features
- **LSTM:** 2 layers, 256 hidden units, dropout 0.3
- **Classifier:** LayerNorm → FC(256→128) → ReLU → FC(128→2)
- **Input:** Sequence of 15 face crops at 224×224
- **Output:** Binary classification (Real / Fake) with confidence

---

## ⚠️ Limitations

- Requires visible human faces in the video
- Accuracy depends on training data quality and volume
- Processing time scales with video length
- Not suitable for real-time / streaming use
- Results should **not** be treated as legal evidence

---

## 📄 License

This project is for **educational and research purposes**.

---

*Built with ❤️ using PyTorch, YOLOv8, MTCNN, and Streamlit*
