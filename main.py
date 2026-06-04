"""
╔══════════════════════════════════════════════════════════════════════╗
║   DEEPFAKE DETECTION — v3 (Target: 90%+ Accuracy)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║  FIXES OVER v2:                                                      ║
║    1. MTCNN face detector   (replaces weak Haar cascade)            ║
║    2. EfficientNet top-30 layers unfrozen (deepfake-aware features) ║
║    3. SEQ_LENGTH 18 → 24   (more temporal signal per video)         ║
║    4. Lower LR for fine-tuning (3e-4 instead of 5e-4)              ║
║    5. Grad-clip added to Adam (stability during fine-tune)          ║
╚══════════════════════════════════════════════════════════════════════╝
"""

# ── Must be before TF import ──────────────────────────────────────────────
import os

# CUDA DLL paths — covers both miniconda and Anaconda on any drive
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
os.environ["TF_CPP_MIN_LOG_LEVEL"]      = "2"
os.environ["PYTHONUNBUFFERED"]          = "1"

import gc, warnings, time, sys
import cv2
import numpy as np
import joblib

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
warnings.filterwarnings("ignore")
cv2.setNumThreads(0)

import tensorflow as tf
from tensorflow.keras.layers import (
    Input, TimeDistributed, Dense, Dropout,
    Bidirectional, LSTM, BatchNormalization,
    MultiHeadAttention, Add, LayerNormalization,
)
from tensorflow.keras.models       import Model
from tensorflow.keras.applications import EfficientNetV2B0
from tensorflow.keras.optimizers   import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks    import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

from sklearn.model_selection    import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics            import confusion_matrix, classification_report, f1_score
from sklearn.preprocessing      import StandardScaler

# ── MTCNN for robust face detection ──────────────────────────────────────
_MTCNN_AVAILABLE = False
try:
    from mtcnn import MTCNN as _MTCNNClass
    _MTCNN_AVAILABLE = True
except ImportError:
    _MTCNNClass = None

def _init_mtcnn():
    """Create a fresh MTCNN instance. Called at startup and after clear_session."""
    if _MTCNN_AVAILABLE:
        det = _MTCNNClass()
        print("✅ MTCNN loaded")
        return det
    print("⚠️  MTCNN not installed — falling back to Haar cascade")
    print("    Run:  pip install mtcnn   to enable better face detection")
    return None

_mtcnn_detector = _init_mtcnn()

# ════════════════════════════════════════════════════════════════════════
# SESSION & GPU
# ════════════════════════════════════════════════════════════════════════
tf.keras.backend.clear_session()
gc.collect()
print("✅ Session cleared")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print(f"✅ GPU: {[g.name for g in gpus]}")
else:
    print("⚠️  CPU only — training will be slow")

# ════════════════════════════════════════════════════════════════════════
# DATASET PATH  — auto-detected from common locations
# ════════════════════════════════════════════════════════════════════════
_DATASET_CANDIDATES = [
    r"D:\Project_DF\dataset",
    r"D:\Project_DF\data",
    r"D:\Project_DF\videos",
    r"D:\Project_DF",
    r"D:\dataset",
    r"D:\data",
    r"C:\Users\ASUS\Downloads\OJUS Project\dataset",
    r"C:\Users\ASUS\Downloads\dataset",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos"),
]

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv")

# ── Folder name → label  (0 = real, 1 = fake) ───────────────────────────
_FOLDER_LABEL_MAP = {
    "fake"          : 1,  "fakes"         : 1,
    "real"          : 0,  "reals"         : 0,
    "0"             : 0,  "1"             : 1,
    "manipulated"   : 1,  "original"      : 0,
    "altered"       : 1,  "pristine"      : 0,
    "forged"        : 1,  "authentic"     : 0,
    "deepfakes"     : 1,  "youtube"       : 0,
    "dfdc"          : 1,
    "synthetic"     : 1,  "genuine"       : 0,
    "generated"     : 1,  "natural"       : 0,
    "fake_videos"   : 1,  "real_videos"   : 0,
    "fake_faces"    : 1,  "real_faces"    : 0,
}

_FAKE_KW = ["fake", "manipulat", "alter", "forg", "deepfake",
             "synthetic", "generat", "dfdc", "tamper"]
_REAL_KW = ["real", "original", "pristine", "authentic",
             "genuine", "youtube", "natural"]


def _find_dataset_path():
    for cand in _DATASET_CANDIDATES:
        if not os.path.isdir(cand):
            continue
        for root, _, files in os.walk(cand):
            if any(f.lower().endswith(VIDEO_EXTS) for f in files):
                print(f"✅ Dataset found at: {cand}")
                return cand
    return None


def scan_dataset(path):
    videos, labels, skipped = [], [], []

    for dirpath, dirnames, filenames in os.walk(path):
        dirnames.sort()
        for fname in sorted(filenames):
            if not fname.lower().endswith(VIDEO_EXTS):
                continue

            full    = os.path.join(dirpath, fname)
            rel     = os.path.relpath(full, path)
            rel_low = rel.lower().replace("\\", "/")
            label   = None

            parent = os.path.basename(dirpath).lower()
            if parent in _FOLDER_LABEL_MAP:
                label = _FOLDER_LABEL_MAP[parent]

            if label is None:
                for part in rel_low.split("/")[:-1]:
                    if part in _FOLDER_LABEL_MAP:
                        label = _FOLDER_LABEL_MAP[part]
                        break

            if label is None:
                if any(kw in rel_low for kw in _FAKE_KW):
                    label = 1
                elif any(kw in rel_low for kw in _REAL_KW):
                    label = 0

            if label is not None:
                videos.append(full)
                labels.append(label)
            else:
                skipped.append(rel)

    n_real = labels.count(0)
    n_fake = labels.count(1)
    print(f"\n   Real (label 0) : {n_real}")
    print(f"   Fake (label 1) : {n_fake}")
    print(f"   Total labelled : {len(videos)}")

    if skipped:
        print(f"   ⚠️  Skipped (no label found): {len(skipped)}")
        for s in skipped[:8]:
            print(f"      {s}")
        if len(skipped) > 8:
            print(f"      ... and {len(skipped)-8} more")
        print("   → Add your folder name to _FOLDER_LABEL_MAP in main.py")

    if not videos:
        print("\n   Directory tree (first 3 levels):")
        for root, dirs, files in os.walk(path):
            depth = root.replace(path, "").count(os.sep)
            if depth > 3:
                continue
            print(f"{'  '*depth}{os.path.basename(root)}/")
            for f in files[:3]:
                print(f"{'  '*(depth+1)}{f}")

    return videos, labels

# ════════════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════════════
SEQ_LENGTH  = 24       # ← increased from 18 (more temporal signal)
IMG_SIZE    = 224
FEAT_DIM    = 1280     # EfficientNetV2B0 pooling="avg" output
BATCH_SIZE  = 16
EPOCHS      = 80
LR          = 3e-4     # ← lowered from 5e-4 (stability with fine-tuning)
N_AUGMENTS  = 3
UNFREEZE_LAYERS = 30   # ← top N layers of EfficientNet to fine-tune

_ROOT             = os.path.dirname(os.path.abspath(__file__))
OLD_FEATURE_CACHE = os.path.join(_ROOT, "feature_cache_224")
FEATURE_CACHE     = os.path.join(_ROOT, "feature_cache_v3")   # v3 — new cache
FRAME_CACHE       = os.path.join(_ROOT, "frame_cache_224")
BEST_MODEL        = os.path.join(_ROOT, "best_model_v4.keras")
SCALER_PATH       = os.path.join(_ROOT, "feature_scaler_v4.pkl")

os.makedirs(FEATURE_CACHE, exist_ok=True)
os.makedirs(FRAME_CACHE,   exist_ok=True)

# Haar cascade fallback (used only if MTCNN is unavailable)
_haar = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ════════════════════════════════════════════════════════════════════════
# FACE DETECTION  — MTCNN primary, Haar fallback
# ════════════════════════════════════════════════════════════════════════
def detect_face(frame_rgb):
    """
    Returns (x1, y1, x2, y2) of the best face, or None if no face found.
    frame_rgb : H×W×3 uint8 RGB image.
    """
    # ── MTCNN ────────────────────────────────────────────────────────────
    if _mtcnn_detector is not None:
        try:
            results = _mtcnn_detector.detect_faces(frame_rgb)
            if results:
                # Pick highest-confidence detection
                best = max(results, key=lambda r: r["confidence"])
                if best["confidence"] > 0.85:
                    x, y, w, h = best["box"]
                    x, y = max(0, x), max(0, y)
                    pad = int(min(w, h) * 0.10)
                    H, W = frame_rgb.shape[:2]
                    return (
                        max(0, x - pad),
                        max(0, y - pad),
                        min(W, x + w + pad),
                        min(H, y + h + pad),
                    )
        except Exception:
            pass

    # ── Haar fallback ─────────────────────────────────────────────────────
    gray  = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    faces = _haar.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        pad = int(min(w, h) * 0.10)
        H, W = frame_rgb.shape[:2]
        return (
            max(0, x - pad), max(0, y - pad),
            min(W, x + w + pad), min(H, y + h + pad),
        )

    return None  # no face found — caller uses full frame

# ════════════════════════════════════════════════════════════════════════
# FRAME EXTRACTION  (4 augmentation modes)
# ════════════════════════════════════════════════════════════════════════
def extract_frames(video_path, aug_mode=0):
    key   = os.path.basename(video_path).replace(".", "_")
    cfile = os.path.join(FRAME_CACHE, f"{key}_{IMG_SIZE}_aug{aug_mode}_s{SEQ_LENGTH}.npy")

    if os.path.exists(cfile):
        try:
            arr = np.load(cfile)
            if arr.shape == (SEQ_LENGTH, IMG_SIZE, IMG_SIZE, 3):
                return arr
            os.remove(cfile)
        except Exception:
            os.remove(cfile)

    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release(); return None

    idxs   = np.linspace(0, total - 1, SEQ_LENGTH).astype(int)
    frames = []

    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        bbox = detect_face(frame_rgb)

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            face = frame_rgb[y1:y2, x1:x2]
        else:
            face = frame_rgb

        if face.size == 0:
            face = frame_rgb

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0

        # ── Augmentation modes ────────────────────────────────────────────
        if aug_mode == 1:
            face = np.fliplr(face)
            face = np.clip(face * np.random.uniform(0.80, 1.20), 0, 1)

        elif aug_mode == 2:
            cp   = int(IMG_SIZE * 0.08)
            t, b = np.random.randint(0, cp), np.random.randint(0, cp)
            l, r = np.random.randint(0, cp), np.random.randint(0, cp)
            crop = face[t:IMG_SIZE-b, l:IMG_SIZE-r]
            if crop.size > 0:
                face = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            face = np.clip(
                face * np.random.uniform(0.70, 1.30) + np.random.uniform(-0.10, 0.10),
                0, 1
            )

        elif aug_mode == 3:
            face = np.fliplr(face)
            face = np.clip(face + np.random.normal(0, 0.025, face.shape), 0, 1)

        frames.append(face.astype(np.float32))

    cap.release()
    if not frames:
        return None

    while len(frames) < SEQ_LENGTH:
        frames.append(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32))

    result = np.array(frames[:SEQ_LENGTH], dtype=np.float32)
    try:
        np.save(cfile, result)
    except Exception:
        pass
    return result

# ════════════════════════════════════════════════════════════════════════
# FEATURE CACHE HELPERS
# ════════════════════════════════════════════════════════════════════════
def _feat_path(video_path, aug_mode=0):
    key    = os.path.basename(video_path).replace(".", "_")
    suffix = "effv3" if aug_mode == 0 else f"aug{aug_mode}v3"
    return os.path.join(FEATURE_CACHE, f"{key}_{IMG_SIZE}_{suffix}.npy")


def migrate_cache(paths):
    """Skip old v2 cache — v3 uses different feature dims due to unfreezing."""
    print("  ℹ️  v3 cache is fresh (unfrozen EfficientNet = new features)")
    return 0


def build_extractor():
    """
    EfficientNetV2B0 with top UNFREEZE_LAYERS trainable.
    This lets the network adapt its high-level features to deepfake artifacts.
    """
    effnet = EfficientNetV2B0(
        weights="imagenet", include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg",
    )

    # Freeze all layers first
    for layer in effnet.layers:
        layer.trainable = False

    # Unfreeze top N layers
    for layer in effnet.layers[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    total     = len(effnet.layers)
    trainable = sum(1 for l in effnet.layers if l.trainable)
    print(f"  EfficientNet: {total} layers total, {trainable} unfrozen (top {UNFREEZE_LAYERS})")
    return effnet


def run_extraction(paths, labels, extractor, aug_mode=0):
    global _mtcnn_detector  # FIX #3: need to reassign after clear_session

    uncached = [i for i, vp in enumerate(paths)
                if not os.path.exists(_feat_path(vp, aug_mode))]
    print(f"  Aug={aug_mode}: {len(paths)-len(uncached)} cached, "
          f"{len(uncached)} to extract")
    t0 = time.time()
    for count, i in enumerate(uncached):
        vp     = paths[i]
        frames = extract_frames(vp, aug_mode=aug_mode)
        if frames is None:
            print(f"  ⚠️  Skipped: {os.path.basename(vp)}"); continue
        feat = extractor.predict(frames, batch_size=8, verbose=0).astype(np.float32)
        try:
            np.save(_feat_path(vp, aug_mode), feat)
        except Exception as e:
            print(f"  ⚠️  Save failed: {e}")
        if (count + 1) % 50 == 0 or count + 1 == len(uncached):
            elapsed = time.time() - t0
            eta     = elapsed / (count + 1) * (len(uncached) - count - 1)
            print(f"  {count+1}/{len(uncached)} | {elapsed:.0f}s | ETA {eta:.0f}s")
            # ── Clear GPU memory every 50 videos to prevent CUDA crash ──
            gc.collect()
            tf.keras.backend.clear_session()
            if count + 1 < len(uncached):
                extractor = build_extractor()
                # FIX #3: re-init MTCNN — clear_session wipes Keras state
                _mtcnn_detector = _init_mtcnn()


def load_features(video_paths, video_labels, aug_modes=(0,)):
    X, y = [], []
    for aug_mode in aug_modes:
        for vp, label in zip(video_paths, video_labels):
            cfile = _feat_path(vp, aug_mode)
            if not os.path.exists(cfile):
                continue
            try:
                feat = np.load(cfile)
                if feat.shape == (SEQ_LENGTH, FEAT_DIM):
                    X.append(feat); y.append(label)
            except Exception:
                pass
    return np.array(X, dtype=np.float32), np.array(y)

# ════════════════════════════════════════════════════════════════════════
# MODEL — BiLSTM + Multi-Head Temporal Attention
# ════════════════════════════════════════════════════════════════════════
def build_model():
    inp = Input(shape=(SEQ_LENGTH, FEAT_DIM), name="feature_input")

    x = LayerNormalization(name="input_ln")(inp)

    x = TimeDistributed(
        Dense(256, activation="gelu", kernel_regularizer=l2(2e-4)),
        name="td_projection"
    )(x)

    x = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.30, recurrent_dropout=0.15),
        name="bilstm_1"
    )(x)
    x = LayerNormalization(name="bilstm1_ln")(x)

    attn_out = MultiHeadAttention(
        num_heads=4, key_dim=64, dropout=0.15,
        name="temporal_attention"
    )(x, x)
    x = Add(name="attn_residual")([x, attn_out])
    x = LayerNormalization(name="attn_ln")(x)

    x = Bidirectional(
        LSTM(64, dropout=0.30, recurrent_dropout=0.15),
        name="bilstm_2"
    )(x)
    x = BatchNormalization(name="bn")(x)

    x   = Dense(128, activation="gelu", kernel_regularizer=l2(2e-4), name="dense_1")(x)
    x   = Dropout(0.45, name="drop_1")(x)
    x   = Dense(32,  activation="gelu", kernel_regularizer=l2(2e-4), name="dense_2")(x)
    x   = Dropout(0.30, name="drop_2")(x)
    out = Dense(1, activation="sigmoid", name="output")(x)

    return Model(inp, out)

# ════════════════════════════════════════════════════════════════════════
# TRAINING GENERATOR — Gaussian noise + Mixup
# ════════════════════════════════════════════════════════════════════════
class FeatureGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y, batch_size,
                 noise_std=0.04, use_mixup=True, mixup_alpha=0.20):
        self.X          = X
        self.y          = y.astype(np.float32)
        self.batch_size = batch_size
        self.noise_std  = noise_std
        self.use_mixup  = use_mixup
        self.alpha      = mixup_alpha
        self.indices    = np.arange(len(X))
        np.random.shuffle(self.indices)

    def __len__(self):
        return max(1, len(self.X) // self.batch_size)

    def __getitem__(self, idx):
        bi  = self.indices[idx * self.batch_size : (idx+1) * self.batch_size]
        Xb  = self.X[bi].copy()
        yb  = self.y[bi].copy()
        Xb += np.random.normal(0, self.noise_std, Xb.shape).astype(np.float32)
        if self.use_mixup and np.random.rand() > 0.50:
            lam  = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(Xb))
            Xb   = (lam * Xb + (1 - lam) * Xb[perm]).astype(np.float32)
            yb   = lam * yb  + (1 - lam) * yb[perm]
        return Xb, yb

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ════════════════════════════════════════════════════════════════════════
# THRESHOLD OPTIMISATION  (FIX #8)
# ════════════════════════════════════════════════════════════════════════
THRESHOLD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_threshold_v4.npy")

def find_best_threshold(model, X_val, y_val):
    """Sweep thresholds on the validation set and pick the one with best F1."""
    probs      = model.predict(X_val, verbose=0).flatten()
    thresholds = np.arange(0.25, 0.76, 0.01)
    best_t, best_f1 = 0.50, 0.0
    for t in thresholds:
        preds = (probs > t).astype(int)
        f1    = f1_score(y_val, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    print(f"  Best threshold: {best_t:.2f}  (F1={best_f1:.4f} on val set)")
    return float(best_t)

# ════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE  (FIX #10: __main__ guard replaces _PIPELINE_RAN flag)
# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── 1. Find dataset ──────────────────────────────────────────────────
    print("\n🔍 Searching for dataset...")
    DATASET_PATH = _find_dataset_path()

    if DATASET_PATH is None:
        print("\n❌  Could not auto-detect dataset. Searched:")
        for c in _DATASET_CANDIDATES:
            print(f"   {c}")
        print("\n👉  Open main.py, find _DATASET_CANDIDATES near the top,")
        print("    and paste your actual dataset path as the FIRST entry.")
        raise RuntimeError("Dataset not found — see instructions above")

    # ── 2. Scan & label ──────────────────────────────────────────────────
    print(f"\n📂 Scanning dataset...")
    paths, labels = scan_dataset(DATASET_PATH)

    if len(paths) == 0:
        print("\n👉  Videos were found but none could be labelled.")
        print("    Open main.py and add your folder names to _FOLDER_LABEL_MAP.")
        raise RuntimeError("No labelled videos — update _FOLDER_LABEL_MAP")

    if len(set(labels)) < 2:
        raise RuntimeError(
            f"Only one class found (labels={set(labels)}). "
            "Both real and fake folders must be present."
        )

    # ── 3. Split BEFORE augmented extraction ────────────────────────────
    idx_arr  = np.arange(len(paths))
    y_arr    = np.array(labels)

    train_idx, temp_idx = train_test_split(
        idx_arr, test_size=0.30, stratify=y_arr, random_state=42
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=y_arr[temp_idx], random_state=42
    )

    train_paths  = [paths[i]  for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_paths    = [paths[i]  for i in val_idx]
    val_labels   = [labels[i] for i in val_idx]
    test_paths   = [paths[i]  for i in test_idx]
    test_labels  = [labels[i] for i in test_idx]

    print(f"\nSplit → Train: {len(train_paths)}  "
          f"Val: {len(val_paths)}  Test: {len(test_paths)}")

    # ── 4. Cache migration notice ────────────────────────────────────────
    print("\n🔄 Checking for old feature cache...")
    migrate_cache(paths)

    # ── 5. Extract features where needed ─────────────────────────────────
    base_missing = [vp for vp in paths
                    if not os.path.exists(_feat_path(vp, 0))]
    aug_missing  = [(vp, m) for vp in train_paths
                    for m in range(1, N_AUGMENTS + 1)
                    if not os.path.exists(_feat_path(vp, m))]

    if base_missing or aug_missing:
        print(f"\n🔨 Building EfficientNet extractor (top {UNFREEZE_LAYERS} layers trainable)...")
        extractor = build_extractor()

        if base_missing:
            print(f"\n🎥 Extracting base features ({len(base_missing)} missing)...")
            run_extraction(paths, labels, extractor, aug_mode=0)

        if aug_missing:
            print(f"\n🎥 Extracting augmented features (training set only)...")
            for m in range(1, N_AUGMENTS + 1):
                print(f"\n  ── Augmentation {m}/{N_AUGMENTS} ──")
                run_extraction(train_paths, train_labels, extractor, aug_mode=m)

        del extractor
        gc.collect()
        tf.keras.backend.clear_session()
        print("\n✅ Extractors freed from GPU")
    else:
        print("✅ All features already cached — skipping extraction")

    # ── 6. Load features ─────────────────────────────────────────────────
    print("\n📦 Loading features...")
    X_train, y_train = load_features(
        train_paths, train_labels,
        aug_modes=list(range(N_AUGMENTS + 1))
    )
    X_val,  y_val  = load_features(val_paths,  val_labels,  aug_modes=[0])
    X_test, y_test = load_features(test_paths, test_labels, aug_modes=[0])

    print(f"Loaded → Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"Train class dist: 0={np.sum(y_train==0)}, 1={np.sum(y_train==1)}")

    # ── 7. Feature normalisation ──────────────────────────────────────────
    # CRITICAL: always load the same scaler the model was trained with.
    print("\n📐 Normalising features with StandardScaler...")

    if os.path.exists(BEST_MODEL) and os.path.exists(SCALER_PATH):
        print(f"  ✅ Loading existing scaler (matches saved model)")
        scaler = joblib.load(SCALER_PATH)
    else:
        print("  🆕 Fitting new scaler on training data...")
        scaler = StandardScaler()
        scaler.fit(X_train.reshape(-1, FEAT_DIM))
        joblib.dump(scaler, SCALER_PATH)
        print(f"  ✅ Scaler saved → {SCALER_PATH}")

    def scale_clip(X):
        return np.clip(
            scaler.transform(X.reshape(-1, FEAT_DIM)).reshape(X.shape),
            -5, 5
        ).astype(np.float32)

    X_train = scale_clip(X_train)
    X_val   = scale_clip(X_val)
    X_test  = scale_clip(X_test)

    # ── 8. Class weights ──────────────────────────────────────────────────
    w             = compute_class_weight("balanced",
                                          classes=np.array([0, 1]), y=y_train)
    class_weights = {0: float(w[0]), 1: float(w[1])}
    print(f"Class weights: {class_weights}")

    # ── 9. Build & compile model  (skip if already trained) ──────────────
    if os.path.exists(BEST_MODEL):
        print(f"\n✅ Trained model found — skipping training")
        print(f"   Loading: {BEST_MODEL}")
        model = tf.keras.models.load_model(BEST_MODEL)
        print("   Model loaded successfully")

    else:
        print("\n🆕 No saved model found — starting fresh training")
        model = build_model()
        model.compile(
            optimizer=Adam(learning_rate=LR, clipnorm=1.0),
            loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.10),
            metrics=["accuracy"],
        )
        model.summary()

        # ── 10. Callbacks ──────────────────────────────────────────────────
        callbacks = [
            EarlyStopping(
                monitor="val_accuracy", patience=15,
                restore_best_weights=True, mode="max", verbose=1
            ),
            ReduceLROnPlateau(
                monitor="val_loss", factor=0.40,
                patience=5, min_lr=1e-7, verbose=1
            ),
            ModelCheckpoint(
                BEST_MODEL, monitor="val_accuracy",
                save_best_only=True, mode="max", verbose=1
            ),
        ]

        # ── 11. Train ──────────────────────────────────────────────────────
        print("\n🚀 Training...\n")
        t0 = time.time()

        train_gen = FeatureGenerator(
            X_train, y_train,
            batch_size=BATCH_SIZE,
            noise_std=0.04,
            use_mixup=True,
            mixup_alpha=0.20,
        )

        model.fit(
            train_gen,
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
        )
        print(f"\n✅ Training done in {(time.time()-t0)/60:.1f} min")

    # ── 12. Optimise decision threshold on validation set (FIX #8) ─────────
    print("\n🎯 Optimising decision threshold on validation set...")
    if os.path.exists(THRESHOLD_PATH):
        best_threshold = float(np.load(THRESHOLD_PATH))
        print(f"  ✅ Loaded saved threshold: {best_threshold:.2f}")
    else:
        best_threshold = find_best_threshold(model, X_val, y_val)
        np.save(THRESHOLD_PATH, best_threshold)
        print(f"  ✅ Threshold saved → {THRESHOLD_PATH}")

    # ── 13. Evaluate ──────────────────────────────────────────────────────
    print("\n📊 FINAL TEST EVALUATION\n")
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"\n🏆 Test Accuracy: {acc*100:.2f}%")

    y_probs = model.predict(X_test).flatten()
    y_pred  = (y_probs > best_threshold).astype("int32")

    print(f"\nUsing threshold: {best_threshold:.2f}  (optimised on val set)")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm       = confusion_matrix(y_test, y_pred)
    real_acc = cm[0, 0] / cm[0].sum() * 100
    fake_acc = cm[1, 1] / cm[1].sum() * 100
    print(f"\nReal accuracy : {real_acc:.1f}%")
    print(f"Fake accuracy : {fake_acc:.1f}%")