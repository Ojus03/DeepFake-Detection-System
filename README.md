# DeepFake-Detection-System ( Dataset )

DFDC Dataset Link - https://www.kaggle.com/datasets/sanikatiwarekar/deep-fake-detection-dfd-entire-original-dataset

---

# DeepFake Detection System

A deep learning based system for detecting deepfake videos, built as a final year engineering project at DES Pune University.

---

## About

This project presents an end-to-end deepfake video detection pipeline that combines spatial feature extraction with temporal sequence modeling. The system analyzes video frames to identify manipulated or AI-generated faces with high accuracy.

---

## Model Architecture

The system uses a three-stage hybrid architecture:

- **EfficientNetV2B0** — extracts spatial features from individual video frames
- **BiLSTM (Bidirectional LSTM)** — captures temporal dependencies across frame sequences
- **Multi-Head Attention** — focuses on the most discriminative regions and time steps

This combination allows the model to detect both frame-level artifacts and sequence-level inconsistencies that are characteristic of deepfake videos.

---

## Results

| Metric | Score |
|--------|-------|
| Test Accuracy | ~92% |
| Architecture | EfficientNetV2B0 + BiLSTM + Multi-Head Attention |
| Input | Video frames (224x224) |

---

## Tech Stack

- Python 3.x
- TensorFlow / Keras
- OpenCV
- Flask (inference API)
- NumPy

---

## Project Structure

```
DeepFake-Detection-System/
├── app.py              # Flask web application
├── inference.py        # Model inference pipeline
├── main.py             # Main training script
├── visualize.py        # Visualization utilities
├── run.bat             # Windows run script
└── .gitignore
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/Ojus03/DeepFake-Detection-System.git
cd DeepFake-Detection-System
```

**2. Install dependencies**
```bash
pip install tensorflow opencv-python flask numpy
```

**3. Run the application**
```bash
python app.py
```
Or on Windows:
```bash
run.bat
```

---

## Team

| Name | Role |
|------|------|
| Ojus | Model Development & Integration |
| Moin Thange | Data Pipeline & Preprocessing |
| Shriram Bidve | Training & Evaluation |
| Vaibhav Khatavkar | Frontend & Deployment |

**Institution:** School of Engineering and Technology, DES Pune University

---

## Note

Model weights and dataset files are not included in this repository due to size constraints. The architecture and inference code are provided for reference.

