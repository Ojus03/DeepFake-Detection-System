"""
╔══════════════════════════════════════════════════════════════════════╗
║   DEEPFAKE DETECTION — app.py  (Gradio UI)                          ║
║   Run:  python app.py                                               ║
║   Then open:  http://localhost:7860                                  ║
║   Uses Gradio — no protobuf conflict with TensorFlow 2.10           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os, gc, time, warnings, tempfile
warnings.filterwarnings("ignore")

# ── CUDA paths ────────────────────────────────────────────────────────
_CUDA_CANDIDATES = [
    r"D:\Anaconda\envs\tf_gpu\Library\bin",
    r"D:\Anaconda\envs\tf_gpu\bin",
    r"C:\Users\ASUS\miniconda3\envs\tf_gpu\Library\bin",
    r"C:\Users\ASUS\miniconda3\envs\tf_gpu\bin",
]
for p in _CUDA_CANDIDATES:
    if os.path.isdir(p):
        os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]              = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"]                   = "3"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import cv2
import numpy as np
import joblib
import tensorflow as tf
import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── MTCNN ─────────────────────────────────────────────────────────────
try:
    from mtcnn import MTCNN
    _mtcnn = MTCNN()
    print("✅ MTCNN loaded")
except ImportError:
    _mtcnn = None
    print("⚠️  MTCNN not found — using Haar fallback")

# ── GPU ───────────────────────────────────────────────────────────────
for gpu in tf.config.list_physical_devices("GPU"):
    try: tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError: pass

# ── Constants ─────────────────────────────────────────────────────────
_ROOT       = os.path.dirname(os.path.abspath(__file__))
SEQ_LENGTH  = 24
IMG_SIZE    = 224
FEAT_DIM    = 1280
UNFREEZE_LAYERS = 30
MODEL_PATH  = os.path.join(_ROOT, "best_model_v4.keras")
SCALER_PATH = os.path.join(_ROOT, "feature_scaler_v4.pkl")

_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ════════════════════════════════════════════════════════════════════════
# LOAD ONCE AT STARTUP
# ════════════════════════════════════════════════════════════════════════
print("Loading model and scaler...")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun main.py first.")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}\nRun main.py first.")

_model  = tf.keras.models.load_model(MODEL_PATH)
_scaler = joblib.load(SCALER_PATH)
print("✅ Model and scaler loaded")

print("Loading EfficientNet extractor...")
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.models       import Model as KModel

_extractor = EfficientNetV2B0(
    weights="imagenet", include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg",
)
for layer in _extractor.layers:                    layer.trainable = False
for layer in _extractor.layers[-UNFREEZE_LAYERS:]: layer.trainable = True

_gc_base  = EfficientNetV2B0(weights="imagenet", include_top=False,
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))
_gc_pool  = tf.keras.layers.GlobalAveragePooling2D()(_gc_base.output)
_gc_model = KModel(_gc_base.input, [_gc_base.output, _gc_pool])
print("✅ All models loaded — ready\n")

# ════════════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════════════

def detect_face(frame_rgb):
    if _mtcnn is not None:
        try:
            res = _mtcnn.detect_faces(frame_rgb)
            if res:
                best = max(res, key=lambda r: r["confidence"])
                if best["confidence"] > 0.85:
                    x, y, w, h = best["box"]
                    x, y = max(0,x), max(0,y)
                    pad  = int(min(w,h)*0.10)
                    H, W = frame_rgb.shape[:2]
                    return max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad)
        except Exception: pass
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, 1.3, 5)
    if len(faces):
        x,y,w,h=faces[0]; pad=int(min(w,h)*0.10)
        H,W=frame_rgb.shape[:2]
        return max(0,x-pad),max(0,y-pad),min(W,x+w+pad),min(H,y+h+pad)
    return None


def extract_frames(video_path):
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    dur   = round(total/fps, 1) if fps > 0 else 0
    if total <= 0:
        cap.release(); return None, [], 0, 0
    idxs      = np.linspace(0, total-1, SEQ_LENGTH).astype(int)
    frames    = []
    raw_faces = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: continue
        fr   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = detect_face(fr)
        face = fr[bbox[1]:bbox[3],bbox[0]:bbox[2]] if bbox else fr
        if face.size == 0: face = fr
        raw_faces.append(cv2.resize(face, (IMG_SIZE, IMG_SIZE)))
        frames.append(cv2.resize(face,(IMG_SIZE,IMG_SIZE)).astype(np.float32)/255.0)
    cap.release()
    while len(frames) < SEQ_LENGTH:
        frames.append(np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.float32))
        raw_faces.append(np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8))
    return np.array(frames[:SEQ_LENGTH],dtype=np.float32), raw_faces[:SEQ_LENGTH], total, dur


def gradcam(frame_np):
    fb = np.expand_dims(frame_np, 0)
    with tf.GradientTape() as tape:
        conv_out, pooled = _gc_model(fb, training=False)
        tape.watch(conv_out)
        score = tf.reduce_mean(pooled)
    grads   = tape.gradient(score, conv_out)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam     = tf.reduce_sum(conv_out[0]*weights, axis=-1).numpy()
    cam     = np.maximum(cam, 0)
    if cam.max() > 0: cam = cam/cam.max()
    return cv2.resize(cam, (IMG_SIZE, IMG_SIZE))


def apply_heatmap(face_uint8, cam):
    heat = cv2.applyColorMap((cam*255).astype(np.uint8), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(face_uint8, 0.55, heat, 0.45, 0)


def make_timeline(arr, verdict, confidence):
    fig, ax = plt.subplots(figsize=(12,3), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    colors = ["#e74c3c" if v>0.5 else "#2ecc71" for v in arr]
    ax.bar(range(1,SEQ_LENGTH+1), arr, color=colors,
           edgecolor="#ffffff22", linewidth=0.5)
    ax.axhline(0.5, color="#ffffff66", linestyle="--", lw=1.2)
    ax.set_xlabel("Frame Index", color="white", fontsize=10)
    ax.set_ylabel("Suspicion Score", color="white", fontsize=10)
    ax.tick_params(colors="white")
    ax.set_xlim([0.5, SEQ_LENGTH+0.5]); ax.set_ylim([0, 1.05])
    for spine in ax.spines.values(): spine.set_edgecolor("#ffffff33")
    ax.set_title(
        f"Frame Suspicion  |  {verdict}  |  {confidence*100:.1f}% fake confidence",
        color="white", fontsize=11, pad=10
    )
    plt.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════
# PREDICT FUNCTION
# ════════════════════════════════════════════════════════════════════════

def predict(video_path):
    if video_path is None:
        empty = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return "⬆️ Upload a video first", empty, empty, empty, None, "—", "—", "—", "—"

    t0 = time.time()

    frames_np, raw_faces, n_frames, duration = extract_frames(video_path)
    if frames_np is None:
        empty = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        return "❌ Could not read video", empty, empty, empty, None, "—", "—", "—", "—"

    base_feats   = _extractor.predict(frames_np, batch_size=8, verbose=0)
    feats_scaled = np.clip(_scaler.transform(base_feats), -5, 5).astype(np.float32)

    confidence = float(_model.predict(feats_scaled[np.newaxis,...], verbose=0)[0,0])
    verdict    = "🔴  DEEPFAKE DETECTED" if confidence > 0.50 else "🟢  AUTHENTIC VIDEO"

    importance = []
    for i in range(SEQ_LENGTH):
        ab = feats_scaled.copy(); ab[i] = 0.0
        c  = float(_model.predict(ab[np.newaxis,...], verbose=0)[0,0])
        importance.append(max(0.0, confidence - c))
    arr = np.array(importance)
    if arr.max() > 0: arr = arr/arr.max()

    top3 = np.argsort(arr)[-3:][::-1]
    heatmaps = []
    for fi in top3:
        cam = gradcam(frames_np[fi])
        heatmaps.append(apply_heatmap(raw_faces[fi], cam))
    while len(heatmaps) < 3:
        heatmaps.append(np.zeros((IMG_SIZE,IMG_SIZE,3),dtype=np.uint8))

    elapsed = round(time.time()-t0, 1)
    fig     = make_timeline(arr, verdict, confidence)

    return (
        verdict,
        heatmaps[0], heatmaps[1], heatmaps[2],
        fig,
        f"{confidence*100:.1f}%",
        f"{(1-confidence)*100:.1f}%",
        f"{elapsed}s",
        f"{n_frames} frames | {duration}s duration",
    )


# ════════════════════════════════════════════════════════════════════════
# GRADIO UI
# ════════════════════════════════════════════════════════════════════════

CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
#verdict { font-size: 1.6rem !important; font-weight: 800 !important;
           text-align: center !important; }
"""

with gr.Blocks(css=CSS, title="DeepFake Detector") as demo:

    gr.Markdown("# 🔍 DeepFake Detection System")
    gr.Markdown("Upload any video to check if it is **real or AI-generated/manipulated**.")

    with gr.Row():
        with gr.Column(scale=1):
            video_in = gr.Video(label="Upload Video")
            run_btn  = gr.Button("🔍  Analyse Video", variant="primary", size="lg")
            gr.Markdown("""
**Supported:** MP4, AVI, MOV, MKV, WEBM, FLV

**Pipeline:**
1. 24 frames extracted evenly
2. MTCNN face detection per frame
3. EfficientNetV2 feature extraction
4. BiLSTM + Attention classification
5. Grad-CAM activation mapping
            """)

        with gr.Column(scale=1):
            verdict_out  = gr.Textbox(label="VERDICT", elem_id="verdict",
                                      interactive=False, lines=2)
            with gr.Row():
                fake_out = gr.Textbox(label="🔴 Fake Probability", interactive=False)
                real_out = gr.Textbox(label="🟢 Real Probability",  interactive=False)
            with gr.Row():
                time_out = gr.Textbox(label="⏱ Time", interactive=False)
                info_out = gr.Textbox(label="📹 Video Info", interactive=False)

    gr.Markdown("### 🔥 Grad-CAM Heatmaps — Top 3 Suspicious Frames")
    gr.Markdown("*Red/yellow = regions the model found suspicious*")
    with gr.Row():
        hm1 = gr.Image(label="Most Suspicious",  image_mode="RGB")
        hm2 = gr.Image(label="2nd Suspicious",   image_mode="RGB")
        hm3 = gr.Image(label="3rd Suspicious",   image_mode="RGB")

    gr.Markdown("### 📊 Frame-level Suspicion Timeline")
    timeline = gr.Plot()

    run_btn.click(
        fn=predict,
        inputs=[video_in],
        outputs=[verdict_out, hm1, hm2, hm3, timeline,
                 fake_out, real_out, time_out, info_out],
    )

    gr.Markdown("---\n*EfficientNetV2B0 + BiLSTM + Multi-Head Attention | MTCNN | 92%+ Test Accuracy*")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860,
                share=False, inbrowser=True)
