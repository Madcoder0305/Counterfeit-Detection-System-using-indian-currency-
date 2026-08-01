# ============================================================
# Dataset Preparation -- Indian Rupee Counterfeit Detection
# Source: Kaggle  preetrank/indian-currency-real-vs-fake-notes-dataset
# Currency: INR
# ============================================================

import os, sys, shutil, random, csv
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
IMG_DIR      = os.path.join(DATA_DIR, "images")
GENUINE_DIR  = os.path.join(IMG_DIR, "genuine")
FAKE_DIR     = os.path.join(IMG_DIR, "fake")

IMG_SIZE = 224  # resize target for CNN (increased from 64 for better feature capture)

# Kaggle dataset cache path
KAGGLE_DS_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache", "kagglehub", "datasets",
    "preetrank", "indian-currency-real-vs-fake-notes-dataset",
    "versions", "2",
)

DENOMINATIONS = ["10", "20", "50", "100", "200", "500", "2000"]
INR_FEATURE_NAMES = [
    "intaglio_depth", "security_thread_visible", "watermark_clarity",
    "color_shift_ink", "microprint_score", "uv_fluorescence",
    "serial_number_font", "denomination",
]


def create_dirs():
    for d in [DATA_DIR, GENUINE_DIR, FAKE_DIR,
              os.path.join(PROJECT_ROOT, "models", "saved"),
              os.path.join(PROJECT_ROOT, "models", "results")]:
        os.makedirs(d, exist_ok=True)


# ------------------------------------------------------------------
# 1. Copy + resize images from Kaggle dataset into data/images/
# ------------------------------------------------------------------
def _copy_and_resize(src_dir, dst_dir, label_prefix):
    """Walk denomination subdirs, resize to 64x64 grayscale, save to dst."""
    count = 0
    for denom in DENOMINATIONS:
        denom_dir = os.path.join(src_dir, denom)
        if not os.path.isdir(denom_dir):
            continue
        for fname in os.listdir(denom_dir):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_path = os.path.join(denom_dir, fname)
            try:
                img = Image.open(src_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                out_name = f"{label_prefix}_{denom}_{count:05d}.png"
                img.save(os.path.join(dst_dir, out_name))
                count += 1
            except Exception as e:
                print(f"  skip {src_path}: {e}")
    return count


def _copy_features(features_base, dst_dir, start_count):
    """Copy Feature close-up images (security feature crops) as genuine data."""
    count = start_count
    for denom_folder in sorted(os.listdir(features_base)):
        denom_path = os.path.join(features_base, denom_folder)
        if not os.path.isdir(denom_path):
            continue
        # Extract denomination from folder name e.g. "500_Features" -> "500"
        denom = denom_folder.replace("_Features", "")
        if denom not in DENOMINATIONS:
            continue
        for fname in os.listdir(denom_path):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            src_path = os.path.join(denom_path, fname)
            try:
                img = Image.open(src_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
                out_name = f"genuine_feat_{denom}_{count:05d}.png"
                img.save(os.path.join(dst_dir, out_name))
                count += 1
            except Exception as e:
                print(f"  skip {src_path}: {e}")
    return count - start_count


def prepare_images():
    kaggle_data = os.path.join(KAGGLE_DS_PATH, "data", "data")
    real_src = os.path.join(kaggle_data, "real")
    fake_src = os.path.join(kaggle_data, "fake")
    features_src = os.path.join(KAGGLE_DS_PATH, "Features", "Features")

    if not os.path.isdir(real_src) or not os.path.isdir(fake_src):
        print("  ERROR: Kaggle dataset not found at", kaggle_data)
        print("  Run:  import kagglehub; kagglehub.dataset_download('preetrank/indian-currency-real-vs-fake-notes-dataset')")
        sys.exit(1)

    print("  Copying + resizing GENUINE note images ...")
    n_genuine = _copy_and_resize(real_src, GENUINE_DIR, "genuine")
    print(f"    -> {n_genuine} genuine note images")

    # Also include Feature close-ups as genuine data
    n_features = 0
    if os.path.isdir(features_src):
        print("  Copying + resizing FEATURE close-up images (genuine) ...")
        n_features = _copy_features(features_src, GENUINE_DIR, n_genuine)
        print(f"    -> {n_features} feature close-up images added to genuine set")
    else:
        print("  Features folder not found, skipping feature close-ups")

    print("  Copying + resizing FAKE note images ...")
    n_fake = _copy_and_resize(fake_src, FAKE_DIR, "fake")
    print(f"    -> {n_fake} fake images")

    total_genuine = n_genuine + n_features
    print(f"\n  TOTAL: {total_genuine} genuine ({n_genuine} notes + {n_features} features) | {n_fake} fake")

    return total_genuine, n_fake


# ------------------------------------------------------------------
# 2. Extract pixel-based tabular features from the images
# ------------------------------------------------------------------
def _extract_features_from_image(img_path, label, denomination):
    """
    Extract 7 texture / structural features from a 64x64 grayscale image
    that approximate INR banknote security characteristics.
    """
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img, dtype=np.float64) / 255.0
    except Exception:
        return None

    h, w = arr.shape

    # 1. Intaglio depth -- local variance in center (raised print = high var)
    center = arr[h//4:3*h//4, w//4:3*w//4]
    intaglio_depth = float(np.std(center))

    # 2. Security thread -- vertical band variance (col 28-36 in 64px img)
    thread_region = arr[:, w//2-4:w//2+4]
    security_thread_visible = float(np.std(thread_region))

    # 3. Watermark clarity -- contrast in right portrait region
    watermark_region = arr[h//4:3*h//4, 3*w//4:]
    watermark_clarity = float(np.std(watermark_region))

    # 4. Color shift ink -- gradient magnitude in top-left numeral region
    numeral_region = arr[:h//4, :w//4]
    gx = np.diff(numeral_region, axis=1)
    gy = np.diff(numeral_region, axis=0)
    color_shift_ink = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy)))

    # 5. Microprint score -- high frequency energy (Laplacian-like)
    lap = arr[1:-1, 1:-1] * 4 - arr[:-2, 1:-1] - arr[2:, 1:-1] - arr[1:-1, :-2] - arr[1:-1, 2:]
    microprint_score = float(np.std(lap))

    # 6. UV fluorescence -- brightness uniformity (genuine is more uniform)
    uv_fluorescence = 1.0 - float(np.std(arr))  # higher = more uniform

    # 7. Serial number font -- horizontal edge regularity in bottom strip
    bottom = arr[3*h//4:, :]
    h_edges = np.abs(np.diff(bottom, axis=1))
    serial_number_font = float(np.mean(h_edges))

    return {
        "intaglio_depth": round(intaglio_depth, 6),
        "security_thread_visible": round(security_thread_visible, 6),
        "watermark_clarity": round(watermark_clarity, 6),
        "color_shift_ink": round(color_shift_ink, 6),
        "microprint_score": round(microprint_score, 6),
        "uv_fluorescence": round(uv_fluorescence, 6),
        "serial_number_font": round(serial_number_font, 6),
        "denomination": int(denomination),
        "label": label,
    }


def extract_tabular_features():
    """Build tabular CSV from all images in data/images/{genuine,fake}."""
    rows = []

    for label, directory, prefix in [(0, GENUINE_DIR, "genuine"), (1, FAKE_DIR, "fake")]:
        if not os.path.isdir(directory):
            continue
        for fname in sorted(os.listdir(directory)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            # Parse denomination from filename  e.g. genuine_500_00042.png
            parts = fname.replace(".png", "").replace(".jpg", "").split("_")
            denom = 500  # default
            for p in parts:
                if p in ("10", "20", "50", "100", "200", "500", "2000"):
                    denom = int(p)
                    break

            feat = _extract_features_from_image(os.path.join(directory, fname), label, denom)
            if feat is not None:
                rows.append(feat)

    random.seed(42)
    random.shuffle(rows)
    return rows


def write_csv_splits(rows):
    """70 / 15 / 15 split, write to data/."""
    n = len(rows)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    train = rows[:n_train]
    val   = rows[n_train:n_train + n_val]
    test  = rows[n_train + n_val:]

    header = INR_FEATURE_NAMES + ["label"]
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(DATA_DIR, f"{split_name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(split_data)

    return len(train), len(val), len(test)


# ------------------------------------------------------------------
# 3. Validation report
# ------------------------------------------------------------------
def print_validation(n_genuine, n_fake, n_train, n_val, n_test, total_tab):
    print("=" * 60)
    print("  DATASET VALIDATION -- Indian Rupee (INR)")
    print("=" * 60)
    print(f"  Dataset Source     : Kaggle (preetrank/indian-currency-real-vs-fake-notes-dataset)")
    print(f"  Currency           : Indian Rupee (INR)")
    print(f"  Denominations      : {DENOMINATIONS}")
    print(f"  Features           : {INR_FEATURE_NAMES}")
    print(f"  Total samples      : {total_tab}")
    print(f"  Genuine            : {n_genuine}")
    print(f"  Fake               : {n_fake}")
    print(f"  Image count        : Genuine: {n_genuine} | Fake: {n_fake}")
    print(f"  Train / Val / Test : {n_train} / {n_val} / {n_test}")
    print("=" * 60)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    print("-" * 60)
    print("  CurrencyGuard -- Dataset Preparation (Kaggle INR)")
    print("-" * 60)

    create_dirs()

    # Step 1 -- copy + resize images
    n_genuine, n_fake = prepare_images()

    # Step 2 -- extract tabular features from images
    print("\n  Extracting tabular features from images ...")
    rows = extract_tabular_features()
    print(f"    -> {len(rows)} feature rows extracted")

    # Step 3 -- write CSV splits
    n_train, n_val, n_test = write_csv_splits(rows)

    print()
    print_validation(n_genuine, n_fake, n_train, n_val, n_test, len(rows))
    print(f"\n  Dataset ready at: {DATA_DIR}")
