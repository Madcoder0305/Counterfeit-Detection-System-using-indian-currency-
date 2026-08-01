"""Classifier wrapper — loads CNN, SVC, Random Forest for INR counterfeit detection."""

import os, io, json, base64, numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAVE_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

FEATURES = ["intaglio_depth","security_thread_visible","watermark_clarity","color_shift_ink","microprint_score","uv_fluorescence","serial_number_font","denomination"]

_cnn_model = None
_svc_bundle = None
_rf_model = None
_mobilenet_model = None

def load_models():
    global _cnn_model, _svc_bundle, _rf_model, _mobilenet_model
    # CNN
    cnn_path = os.path.join(SAVE_DIR, "cnn_best.h5")
    if os.path.exists(cnn_path):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow import keras
        _cnn_model = keras.models.load_model(cnn_path)
        print("  [OK] CNN model loaded")
    else:
        print("  [WARN] CNN model not found")
    # SVC
    svc_path = os.path.join(SAVE_DIR, "svc_model.pkl")
    if os.path.exists(svc_path):
        import joblib
        _svc_bundle = joblib.load(svc_path)
        print("  [OK] SVC model loaded")
    else:
        print("  [WARN] SVC model not found")
    # RF
    rf_path = os.path.join(SAVE_DIR, "rf_model.pkl")
    if os.path.exists(rf_path):
        import joblib
        _rf_model = joblib.load(rf_path)
        print("  [OK] Random Forest model loaded")
    else:
        print("  [WARN] RF model not found")
    # MobileNetV2
    mobilenet_path = os.path.join(SAVE_DIR, "mobilenet_best.h5")
    if os.path.exists(mobilenet_path):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        from tensorflow import keras
        _mobilenet_model = keras.models.load_model(mobilenet_path)
        print("  [OK] MobileNetV2 model loaded")
    else:
        print("  [WARN] MobileNetV2 model not found")

def models_loaded():
    return any([_cnn_model is not None, _svc_bundle is not None, _rf_model is not None, _mobilenet_model is not None])


def run_mobilenet_inference(img_array):
    """Run MobileNetV2 inference on a single image.

    Args:
        img_array: numpy array of shape (1, 224, 224, 3) with pixel values in [0, 255].

    Returns:
        (probability, verdict) where probability is a float 0.0–1.0
        and verdict is "Genuine" or "Counterfeit".
    """
    if _mobilenet_model is None:
        return 0.0, "Unknown"
    import tensorflow as tf
    processed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array.copy())
    prob = float(_mobilenet_model.predict(processed, verbose=0).flatten()[0])
    verdict = "Counterfeit" if prob >= 0.5 else "Genuine"
    return prob, verdict


# ------------------------------------------------------------------
# Grad-CAM heatmap generation
# ------------------------------------------------------------------
IMG_SIZE = 128  # model's expected input size
CONFIDENCE_THRESHOLD = 0.70  # below this → "Uncertain"

# INR security-feature region map (quadrant -> label)
_REGION_MAP = {
    "top-left":      "Watermark zone",
    "top-right":     "Security thread",
    "bottom-left":   "Serial number (left)",
    "bottom-right":  "Serial number (right)",
    "centre":        "Central design / portrait",
}


def _find_last_conv_layer(model):
    """FIX 2 — walk layers using isinstance(Conv2D) instead of string match."""
    import tensorflow as tf
    last_name = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            last_name = layer.name
    return last_name


# Edge cropping removed — serial numbers and other edge features
# are critical for counterfeit detection and must not be cut off.


def _generate_gradcam(model, img_array, pred_class):
    """
    Generate a Grad-CAM heatmap for `pred_class` using tf.GradientTape.
    Handles Sequential models (including nested augmentation layers).
    Returns a 2-D numpy heatmap in [0, 1], or None on failure.
    """
    import tensorflow as tf

    last_conv_name = _find_last_conv_layer(model)
    if last_conv_name is None:
        return None

    img_tensor = tf.constant(img_array, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        x = img_tensor
        conv_output = None
        for layer in model.layers:
            x = layer(x)
            if layer.name == last_conv_name:
                conv_output = x
        predictions = x
        if conv_output is None:
            return None
        class_score = predictions[:, pred_class]

    grads = tape.gradient(class_score, conv_output)
    if grads is None:
        return None

    weights = tf.reduce_mean(grads, axis=(1, 2))
    cam = tf.reduce_sum(conv_output * weights[:, tf.newaxis, tf.newaxis, :], axis=-1)
    cam = cam[0].numpy()

    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam


def _label_for_position(cy_frac, cx_frac):
    """Map a normalised (y, x) position to the nearest INR security region."""
    if 0.35 <= cy_frac <= 0.65 and 0.35 <= cx_frac <= 0.65:
        return _REGION_MAP["centre"]
    if cy_frac < 0.5:
        return _REGION_MAP["top-left"] if cx_frac < 0.5 else _REGION_MAP["top-right"]
    return _REGION_MAP["bottom-left"] if cx_frac < 0.5 else _REGION_MAP["bottom-right"]


def _find_top_peaks(cam, n=2):
    """
    FIX 3 — find the top-n peak coordinates in the Grad-CAM array.
    Returns list of (row_frac, col_frac) normalised to [0,1].
    """
    h, w = cam.shape
    flat = cam.flatten()
    # suppress duplicates within a 3-cell radius by iterative masking
    peaks = []
    mask = cam.copy()
    for _ in range(n):
        idx = int(np.argmax(mask))
        r, c = divmod(idx, w)
        peaks.append((r / max(h - 1, 1), c / max(w - 1, 1)))
        # zero out a neighbourhood so the next peak is distinct
        r_lo, r_hi = max(r - 3, 0), min(r + 4, h)
        c_lo, c_hi = max(c - 3, 0), min(c + 4, w)
        mask[r_lo:r_hi, c_lo:c_hi] = 0
    return peaks


def _overlay_heatmap_b64(original_gray_img, cam, alpha=0.5):
    """
    Resize `cam` to match `original_gray_img`, apply a jet colormap,
    overlay on the original, and return as a clean base64 PNG.
    """
    w, h = original_gray_img.size  # PIL (W, H)

    # Resize heatmap to image size
    cam_uint8 = (cam * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8).resize((w, h), Image.BILINEAR)
    cam_resized = np.array(cam_pil, dtype=np.float64) / 255.0

    # Jet-like colormap  (blue → cyan → green → yellow → red)
    def jet_colormap(v):
        r = np.clip(1.5 - np.abs(4.0 * v - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * v - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * v - 1.0), 0, 1)
        return (r * 255).astype(np.uint8), (g * 255).astype(np.uint8), (b * 255).astype(np.uint8)

    r, g, b = jet_colormap(cam_resized)
    heatmap_rgb = np.stack([r, g, b], axis=-1)
    orig_rgb = np.stack([np.array(original_gray_img)] * 3, axis=-1).astype(np.float64)
    blended = ((1 - alpha) * orig_rgb + alpha * heatmap_rgb.astype(np.float64)).astype(np.uint8)

    # Encode as base64 PNG
    overlay_img = Image.fromarray(blended, "RGB")
    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ------------------------------------------------------------------
# Serial Number Anomaly Detection (no OCR required)
# ------------------------------------------------------------------
# INR notes have serial numbers at specific locations. Specimen/fake
# notes use repeated digits (e.g. "000000") which create a highly
# self-similar pattern detectable via cross-correlation.

# Serial number region coordinates (fraction of image: y1, y2, x1, x2)
# INR notes have serial numbers at:
#   - Top-left: "0AA 000000" — prefix ~5-15% x, digits ~15-35% x, y: 3-12%
#   - Bottom-center: "0AA 000000" — prefix ~25-35% x, digits ~35-55% x, y: 85-96%
# We scan a few slightly offset sub-regions for robustness.
_SERIAL_REGIONS = [
    # Top-left serial — digits only (skip "0AA" prefix)
    (0.02, 0.13, 0.14, 0.36),
    (0.02, 0.13, 0.10, 0.35),
    # Bottom serial — digits only (skip "0AA" prefix)
    (0.84, 0.97, 0.33, 0.56),
    (0.84, 0.97, 0.35, 0.58),
]

# Number of digit cells to split a serial region into
_NUM_DIGIT_CELLS = 4
# Threshold: genuine notes show min-pair corr < 0.1, specimen > 0.45
# Using 0.45 as decision boundary
_SERIAL_SIMILARITY_THRESHOLD = 0.45


def _check_serial_number_anomaly(img_gray_arr):
    """
    Detect repeated/specimen serial numbers (e.g. '000000') by measuring
    cross-correlation between ALL pairs of digit cells in the serial number
    regions. The key insight: on genuine notes, at least one pair of digits
    differs significantly (min correlation < 0.1), while on specimen notes
    with repeated digits, ALL pairs are similar (min correlation > 0.4).
    Returns (is_suspicious: bool, max_min_sim: float, detail: str).
    """
    h, w = img_gray_arr.shape
    best_min_sim = 0.0
    suspicious_region = None

    for (y1f, y2f, x1f, x2f) in _SERIAL_REGIONS:
        y1, y2 = int(y1f * h), int(y2f * h)
        x1, x2 = int(x1f * w), int(x2f * w)
        region = img_gray_arr[y1:y2, x1:x2]

        rh, rw = region.shape
        if rw < _NUM_DIGIT_CELLS * 3 or rh < 3:
            continue

        # Split region into digit-sized vertical strips
        cell_w = rw // _NUM_DIGIT_CELLS
        cells = []
        for i in range(_NUM_DIGIT_CELLS):
            cell = region[:, i * cell_w:(i + 1) * cell_w].flatten().astype(np.float64)
            # Normalise to zero mean, unit variance for correlation
            std = np.std(cell)
            if std > 1e-6:
                cell = (cell - np.mean(cell)) / std
            else:
                cell = cell - np.mean(cell)
            cells.append(cell)

        if len(cells) < 2:
            continue

        # Compute ALL pairwise correlations (not just adjacent)
        all_corrs = []
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                n = len(cells[i])
                corr = np.dot(cells[i], cells[j]) / max(n, 1)
                all_corrs.append(corr)

        # Use MINIMUM correlation — on repeated digits, even the minimum
        # stays high; on varied digits, at least one pair drops low
        min_corr = float(min(all_corrs))
        if min_corr > best_min_sim:
            best_min_sim = min_corr
            suspicious_region = f"y:{y1f:.0%}-{y2f:.0%} x:{x1f:.0%}-{x2f:.0%}"

    # Threshold: genuine < 0.1, specimen > 0.45 → use 0.40 as decision boundary
    is_suspicious = best_min_sim > _SERIAL_SIMILARITY_THRESHOLD
    detail = ""
    if is_suspicious:
        detail = (
            f"Serial number anomaly detected (min digit-pair similarity: {best_min_sim:.1%}). "
            f"Repeated/specimen digits found in region {suspicious_region}."
        )

    return is_suspicious, best_min_sim, detail


def predict_image(image_bytes):
    """
    Multi-stage INR note authentication:
      1. Serial number anomaly detection (image processing)
      2. CNN visual inference
      3. Combined result — serial anomaly overrides CNN if detected
    Returns (result, confidence, model_name, heatmap_b64, attention_regions).
    """
    if _cnn_model is None:
        return None, 0.0, "CNN", None, []

    # Load full-resolution image for serial number analysis
    img_full = Image.open(image_bytes).convert("L")
    full_arr = np.array(img_full, dtype=np.float32) / 255.0

    # Stage 1: Serial number anomaly check (on full-res image)
    serial_suspicious, serial_sim, serial_detail = _check_serial_number_anomaly(full_arr)
    if serial_suspicious:
        print(f"  [ALERT] {serial_detail}")

    # Stage 2: CNN inference (on resized image)
    img_resized = img_full.resize((IMG_SIZE, IMG_SIZE))
    img_arr = np.array(img_resized, dtype=np.float32) / 255.0
    img_arr = img_arr.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    probs = _cnn_model.predict(img_arr, verbose=0)[0]
    pred_class = int(np.argmax(probs))
    cnn_confidence = float(probs[pred_class])

    # Stage 3: Combine results
    if serial_suspicious:
        # Serial number anomaly overrides CNN — this is a specimen/fake
        result = "Counterfeit"
        # Boost confidence: blend CNN fake-class prob with serial similarity
        fake_prob = float(probs[1])
        confidence = max(fake_prob, serial_sim)
    elif cnn_confidence < CONFIDENCE_THRESHOLD:
        result = "Uncertain"
        confidence = cnn_confidence
    else:
        result = "Genuine" if pred_class == 0 else "Counterfeit"
        confidence = cnn_confidence

    # Grad-CAM on the full input
    heatmap_b64 = None
    attention_regions = []
    try:
        cam = _generate_gradcam(_cnn_model, img_arr, pred_class)
        if cam is not None:
            # Find top-2 peaks & assign INR region labels
            peaks = _find_top_peaks(cam, n=2)
            seen = set()
            for (ry, rx) in peaks:
                label = _label_for_position(ry, rx)
                if label not in seen:
                    seen.add(label)
                    attention_regions.append((label, ry, rx))

            # Overlay heatmap on the original image
            heatmap_b64 = _overlay_heatmap_b64(img_resized, cam)
    except Exception as e:
        print(f"  [WARN] Grad-CAM failed: {e}")

    # Add serial number region to attention if anomaly detected
    if serial_suspicious:
        if "Serial number (left)" not in [r[0] for r in attention_regions]:
            attention_regions.append(("Serial number (left)", 0.88, 0.25))
        if "Serial number (right)" not in [r[0] for r in attention_regions]:
            attention_regions.append(("Serial number (right)", 0.88, 0.75))

    return result, confidence, "CNN", heatmap_b64, [r[0] for r in attention_regions]

def predict_tabular(features_dict, model_name="SVC"):
    """Run SVC or RF inference on INR tabular features."""
    vals = np.array([[features_dict.get(f, 0.0) for f in FEATURES]])
    if model_name == "SVC" and _svc_bundle is not None:
        scaler = _svc_bundle["scaler"]
        model = _svc_bundle["model"]
        vals_scaled = scaler.transform(vals)
        pred = model.predict(vals_scaled)[0]
        proba = model.predict_proba(vals_scaled)[0]
        confidence = float(max(proba))
        if confidence < CONFIDENCE_THRESHOLD:
            result = "Uncertain"
        else:
            result = "Genuine" if pred == 0 else "Counterfeit"
        return result, confidence, "SVC"
    elif model_name == "Random Forest" and _rf_model is not None:
        pred = _rf_model.predict(vals)[0]
        proba = _rf_model.predict_proba(vals)[0]
        confidence = float(max(proba))
        if confidence < CONFIDENCE_THRESHOLD:
            result = "Uncertain"
        else:
            result = "Genuine" if pred == 0 else "Counterfeit"
        return result, confidence, "Random Forest"
    return None, 0.0, model_name
