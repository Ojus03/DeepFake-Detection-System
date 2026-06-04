"""
╔══════════════════════════════════════════════════════════════════════╗
║   DEEPFAKE DETECTION — inference.py                                 ║
║   Run on ANY video not in the dataset.                              ║
║   Usage:                                                            ║
║     python inference.py path/to/video.mp4                          ║
║     python inference.py  (interactive prompt)                       ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, sys, gc, time, warnings
warnings.filterwarnings("ignore")

# ── CUDA paths ────────────────────────────────────────────────────────
_CUDA_CANDIDATES = [
    r"D:\Anaconda\envs\tf_gpu\Library\bin",
    r"D:\Anaconda\envs\tf_gpu\bin",
    r"C:\Users\ASUS\miniconda3\envs\tf_gpu\Library\bin",
    r"C:\Users\ASUS\miniconda3\envs\tf_gpu\bin",
    r"C:\ProgramData\Anaconda3\envs\tf_gpu\Library\bin",
    r"C:\ProgramData\Anaconda3\envs\tf_gpu\bin",
]
for p in _CUDA_CANDIDATES:
    if os.path.isdir(p):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "3"

import cv2
import numpy as np
import joblib
import tensorflow as tf

# ── MTCNN ─────────────────────────────────────────────────────────────
try:
    from mtcnn import MTCNN
    _mtcnn = MTCNN()
except ImportError:
    _mtcnn = None

# ── Constants (must match main.py exactly) ────────────────────────────
_ROOT        = os.path.dirname(os.path.abspath(__file__))
SEQ_LENGTH   = 24
IMG_SIZE     = 224
FEAT_DIM     = 1280
UNFREEZE_LAYERS = 30
MODEL_PATH   = os.path.join(_ROOT, "best_model_v3.keras")
SCALER_PATH  = os.path.join(_ROOT, "feature_scaler_v3.pkl")
VIDEO_EXTS   = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def detect_face(frame_rgb):
    if _mtcnn is not None:
        try:
            results = _mtcnn.detect_faces(frame_rgb)
            if results:
                best = max(results, key=lambda r: r["confidence"])
                if best["confidence"] > 0.85:
                    x, y, w, h = best["box"]
                    x, y = max(0, x), max(0, y)
                    pad  = int(min(w, h) * 0.10)
                    H, W = frame_rgb.shape[:2]
                    return (max(0, x-pad), max(0, y-pad),
                            min(W, x+w+pad), min(H, y+h+pad))
        except Exception:
            pass
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad  = int(min(w, h) * 0.10)
        H, W = frame_rgb.shape[:2]
        return (max(0, x-pad), max(0, y-pad),
                min(W, x+w+pad), min(H, y+h+pad))
    return None


def extract_frames(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None, []

    idxs        = np.linspace(0, total - 1, SEQ_LENGTH).astype(int)
    frames      = []
    raw_faces   = []   # stored for Grad-CAM / heatmap overlay

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox      = detect_face(frame_rgb)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face = frame_rgb[y1:y2, x1:x2]
        else:
            face = frame_rgb

        if face.size == 0:
            face = frame_rgb

        raw_faces.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE)))
        face_norm = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        frames.append(face_norm)

    cap.release()
    if not frames:
        return None, []

    while len(frames) < SEQ_LENGTH:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))
        raw_faces.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8))

    return np.array(frames[:SEQ_LENGTH], dtype=np.float32), raw_faces[:SEQ_LENGTH]


def build_extractor():
    from tensorflow.keras.applications import EfficientNetV2B0
    effnet = EfficientNetV2B0(
        weights="imagenet", include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg",
    )
    for layer in effnet.layers:
        layer.trainable = False
    for layer in effnet.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True
    return effnet


def build_classifier():
    from tensorflow.keras.layers import (
        Input, TimeDistributed, Dense, Dropout,
        Bidirectional, LSTM, BatchNormalization,
        MultiHeadAttention, Add, LayerNormalization,
    )
    from tensorflow.keras.models       import Model
    from tensorflow.keras.regularizers import l2

    inp = Input(shape=(SEQ_LENGTH, FEAT_DIM))
    x   = LayerNormalization()(inp)
    x   = TimeDistributed(Dense(256, activation="gelu",
                                kernel_regularizer=l2(1e-4)))(x)
    x   = Bidirectional(LSTM(128, return_sequences=True,
                              dropout=0.20, recurrent_dropout=0.10))(x)
    x   = LayerNormalization()(x)
    attn_out = MultiHeadAttention(num_heads=4, key_dim=64, dropout=0.10)(x, x)
    x   = Add()([x, attn_out])
    x   = LayerNormalization()(x)
    x   = Bidirectional(LSTM(64, dropout=0.20, recurrent_dropout=0.10))(x)
    x   = BatchNormalization()(x)
    x   = Dense(128, activation="gelu", kernel_regularizer=l2(1e-4))(x)
    x   = Dropout(0.35)(x)
    x   = Dense(32,  activation="gelu", kernel_regularizer=l2(1e-4))(x)
    x   = Dropout(0.20)(x)
    out = Dense(1, activation="sigmoid")(x)
    return Model(inp, out)


# ════════════════════════════════════════════════════════════════════════
# GRAD-CAM  — highlights which face regions triggered the detection
# ════════════════════════════════════════════════════════════════════════

def gradcam_on_frame(extractor_gradcam, frame_np):
    """
    Returns a heatmap (H×W, float32 0-1) for a single frame.
    extractor_gradcam: a model that outputs (conv_features, pooled_features).
    """
    frame_batch = np.expand_dims(frame_np, 0)   # (1, H, W, 3)
    with tf.GradientTape() as tape:
        conv_out, pooled = extractor_gradcam(frame_batch, training=False)
        tape.watch(conv_out)
        # Use pooled output as the "score" to differentiate
        score = tf.reduce_mean(pooled)
    grads    = tape.gradient(score, conv_out)[0]          # (h, w, c)
    weights  = tf.reduce_mean(grads, axis=(0, 1))         # (c,)
    cam      = tf.reduce_sum(conv_out[0] * weights, axis=-1).numpy()
    cam      = np.maximum(cam, 0)
    if cam.max() > 0:
        cam  = cam / cam.max()
    cam      = cv2.resize(cam, (IMG_SIZE, IMG_SIZE))
    return cam.astype(np.float32)


def build_gradcam_extractor():
    """EfficientNetV2B0 that outputs both last conv map and pooled vector."""
    from tensorflow.keras.applications import EfficientNetV2B0
    from tensorflow.keras.models       import Model

    base   = EfficientNetV2B0(weights="imagenet", include_top=False,
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))
    pool   = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    return Model(base.input, [base.output, pool])


def apply_heatmap(raw_face_uint8, cam):
    """Blend Grad-CAM heatmap onto original face crop."""
    heat = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    out  = cv2.addWeighted(raw_face_uint8, 0.55, heat, 0.45, 0)
    return out


# ════════════════════════════════════════════════════════════════════════
# FRAME-LEVEL CONFIDENCE
# ════════════════════════════════════════════════════════════════════════

def frame_confidences(extractor, classifier, scaler, frames_np):
    """
    Returns per-frame fake probability by masking one frame at a time
    and observing how confidence changes.
    Fast approximation: replace frame i with zeros and re-predict.
    """
    base_feats = extractor.predict(frames_np, batch_size=8, verbose=0)
    base_feats_scaled = np.clip(
        scaler.transform(base_feats), -5, 5
    ).astype(np.float32)

    # Full sequence confidence (baseline)
    full_seq   = base_feats_scaled[np.newaxis, ...]       # (1, 24, 1280)
    base_conf  = float(classifier.predict(full_seq, verbose=0)[0, 0])

    scores = []
    for i in range(SEQ_LENGTH):
        ablated          = base_feats_scaled.copy()
        ablated[i]       = 0.0
        seq              = ablated[np.newaxis, ...]
        conf             = float(classifier.predict(seq, verbose=0)[0, 0])
        # How much does removing frame i drop the confidence?
        importance       = base_conf - conf
        scores.append(max(0.0, importance))

    # Normalise to 0-1
    arr = np.array(scores)
    if arr.max() > 0:
        arr = arr / arr.max()
    return arr, base_conf


# ════════════════════════════════════════════════════════════════════════
# MAIN PREDICT FUNCTION  (called by app.py and directly)
# ════════════════════════════════════════════════════════════════════════

def predict_video(video_path, save_heatmap_dir=None):
    """
    Full pipeline for one video.
    Returns dict with keys:
        verdict, confidence, label, frame_scores,
        heatmap_frames (list of RGB uint8 arrays), inference_time_s
    """
    t0 = time.time()
    print(f"\n🎬 Processing: {os.path.basename(video_path)}")

    # ── 1. Validate ────────────────────────────────────────────────────
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not video_path.lower().endswith(VIDEO_EXTS):
        raise ValueError(f"Unsupported format. Supported: {VIDEO_EXTS}")

    # ── 2. Load model & scaler ─────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at:\n  {MODEL_PATH}\n"
            "Run main.py first to train and save the model."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at:\n  {SCALER_PATH}\n"
            "Run main.py first."
        )

    print("  Loading model & scaler...")
    classifier = tf.keras.models.load_model(MODEL_PATH)
    scaler     = joblib.load(SCALER_PATH)

    # ── 3. Extract frames ──────────────────────────────────────────────
    print("  Extracting frames + detecting faces...")
    frames_np, raw_faces = extract_frames(video_path)
    if frames_np is None:
        raise RuntimeError("Could not read any frames from the video.")

    # ── 4. Extract EfficientNet features ──────────────────────────────
    print("  Running EfficientNet feature extraction...")
    extractor  = build_extractor()
    base_feats = extractor.predict(frames_np, batch_size=8, verbose=0)

    # ── 5. Scale ───────────────────────────────────────────────────────
    feats_scaled = np.clip(
        scaler.transform(base_feats), -5, 5
    ).astype(np.float32)

    # ── 6. Classify ────────────────────────────────────────────────────
    print("  Classifying sequence...")
    seq        = feats_scaled[np.newaxis, ...]    # (1, 24, 1280)
    confidence = float(classifier.predict(seq, verbose=0)[0, 0])
    verdict    = "FAKE" if confidence > 0.50 else "REAL"
    label      = 1 if confidence > 0.50 else 0

    # ── 7. Frame-level importance ──────────────────────────────────────
    print("  Computing frame-level confidence scores...")
    frame_scores, _ = frame_confidences(extractor, classifier, scaler, frames_np)

    # ── 8. Grad-CAM on top-3 most suspicious frames ────────────────────
    print("  Generating Grad-CAM heatmaps...")
    gradcam_model  = build_gradcam_extractor()
    top3_idx       = np.argsort(frame_scores)[-3:][::-1]
    heatmap_frames = []

    for idx in range(SEQ_LENGTH):
        cam    = gradcam_on_frame(gradcam_model, frames_np[idx])
        vis    = apply_heatmap(raw_faces[idx], cam)
        heatmap_frames.append(vis)

    # ── 9. Save heatmap grid if requested ─────────────────────────────
    if save_heatmap_dir:
        os.makedirs(save_heatmap_dir, exist_ok=True)
        vname  = os.path.splitext(os.path.basename(video_path))[0]
        # Save top-3 heatmap frames
        for rank, fi in enumerate(top3_idx):
            out_path = os.path.join(
                save_heatmap_dir, f"{vname}_heatmap_frame{fi}_rank{rank+1}.png"
            )
            cv2.imwrite(out_path,
                        cv2.cvtColor(heatmap_frames[fi], cv2.COLOR_RGB2BGR))
        print(f"  ✅ Heatmaps saved → {save_heatmap_dir}")

    elapsed = time.time() - t0

    # ── 10. Cleanup ────────────────────────────────────────────────────
    del extractor, gradcam_model, classifier
    gc.collect()
    tf.keras.backend.clear_session()

    result = {
        "verdict"         : verdict,
        "confidence"      : confidence,
        "fake_pct"        : round(confidence * 100, 2),
        "real_pct"        : round((1 - confidence) * 100, 2),
        "label"           : label,
        "frame_scores"    : frame_scores,
        "heatmap_frames"  : heatmap_frames,
        "top3_frames"     : top3_idx.tolist(),
        "inference_time_s": round(elapsed, 2),
    }
    return result


def print_result(result):
    print("\n" + "═" * 50)
    verdict = result["verdict"]
    conf    = result["confidence"]
    bar_len = 30
    filled  = int(conf * bar_len)
    bar     = "█" * filled + "░" * (bar_len - filled)

    print(f"  Verdict    : {'🔴 FAKE' if verdict=='FAKE' else '🟢 REAL'}")
    print(f"  Confidence : [{bar}] {conf*100:.1f}%")
    print(f"  Fake prob  : {result['fake_pct']}%")
    print(f"  Real prob  : {result['real_pct']}%")
    print(f"  Time taken : {result['inference_time_s']}s")
    print(f"\n  Most suspicious frames: {result['top3_frames']}")
    print("═" * 50)


# ════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # GPU memory growth
    for gpu in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass

    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("DeepFake Detection — Inference")
        print("Supported formats:", ", ".join(VIDEO_EXTS))
        video_path = input("\nEnter path to video: ").strip().strip('"')

    result = predict_video(
        video_path,
        save_heatmap_dir=os.path.join(_ROOT, "heatmap_outputs")
    )
    print_result(result)
