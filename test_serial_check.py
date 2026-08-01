"""Final verification: all 3 images."""
import numpy as np
from PIL import Image
import sys, os, requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
import importlib, classifier
importlib.reload(classifier)
from classifier import _check_serial_number_anomaly

print("=== Serial Number Detector ===")
for fname in ["test_1.jpeg", "test_2.jpeg", "test_3.jpeg"]:
    path = os.path.expanduser(rf"~\Downloads\{fname}")
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    suspicious, sim, detail = _check_serial_number_anomaly(arr)
    expected = "FAKE" if fname != "test_2.jpeg" else "GENUINE"
    status = "PASS" if (suspicious and expected == "FAKE") or (not suspicious and expected == "GENUINE") else "FAIL"
    print(f"  {fname} [{expected}]: suspicious={suspicious} sim={sim:.4f} [{status}]")

# Wait for server reload
import time; time.sleep(3)

print("\n=== API Results ===")
for fname in ["test_1.jpeg", "test_2.jpeg", "test_3.jpeg"]:
    path = os.path.expanduser(rf"~\Downloads\{fname}")
    expected = "Counterfeit" if fname != "test_2.jpeg" else "NOT Counterfeit"
    try:
        with open(path, "rb") as f:
            resp = requests.post("http://localhost:8000/api/predict",
                               files={"file": (fname, f, "image/jpeg")})
        data = resp.json()
        result = data.get('result')
        conf = data.get('confidence')
        print(f"  {fname} [expect {expected}]: {result} ({conf:.1%})")
    except Exception as e:
        print(f"  {fname}: ERROR - {e}")
