# CurrencyGuard ₹ — Indian Rupee Counterfeit Detection System

> **Counterfeit Indian Currency Detection System using CNN with SVC & Random Forest Comparison**

## Overview

A full-stack counterfeit currency detection system for **Indian Rupee (INR)** banknotes using Convolutional Neural Networks (CNN), Support Vector Classifier (SVC), and Random Forest Classifier. The CNN model serves as the primary high-precision detector while SVC and Random Forest provide comparative baselines on tabular security features.

> ⚠️ **Dataset:** This system is trained on the **real Kaggle dataset** — [`preetrank/indian-currency-real-vs-fake-notes-dataset`](https://www.kaggle.com/datasets/preetrank/indian-currency-real-vs-fake-notes-dataset)
> Supported denominations: **₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000**

## Dataset Statistics

| Category | Count | Source |
|----------|-------|--------|
| Genuine note images | 4,937 | `data/data/real/` |
| Feature close-ups (genuine) | 1,318 | `Features/Features/` |
| Fake note images | 2,505 | `data/data/fake/` |
| **Total images** | **8,760** | All 7 denominations |
| Train / Val / Test | 6,132 / 1,314 / 1,314 | 70% / 15% / 15% split |

## Model Performance (Real Kaggle Data)

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| **CNN** 🏆 | **87.4%** | 74.0% | **87.5%** | **80.1%** |
| Random Forest | 86.4% | **78.3%** | 68.4% | 73.0% |
| SVC | 85.8% | 74.9% | 70.9% | 72.9% |

CNN achieves the highest accuracy and recall — critical for counterfeit detection where missing a fake note is costly.

## Features

- 🔍 **CNN-based image classification** of genuine vs counterfeit INR notes
- 📊 **SVC & Random Forest comparison** on INR-specific security features
- 🌐 **FastAPI REST backend** with SQLite scan history
- 💎 **Premium dark-mode frontend** — pure HTML/CSS/JS, no build step
- 📱 **Mobile-responsive** design with drag-and-drop upload
- 📈 **Real-time stats dashboard** with auto-refresh
- 🏦 **All 7 denominations**: ₹10, ₹20, ₹50, ₹100, ₹200, ₹500, ₹2000

## INR Security Features (Tabular Model Inputs)

| Feature | Description |
|---------|-------------|
| `intaglio_depth` | Tactile feel of intaglio printing (raised ink) |
| `security_thread_visible` | Windowed security thread clarity |
| `watermark_clarity` | Mahatma Gandhi watermark clarity score |
| `color_shift_ink` | Colour-shifting ink on numeral (green↔blue) |
| `microprint_score` | Micro-lettering sharpness ("RBI", "भारत") |
| `uv_fluorescence` | UV-reactive fiber visibility score |
| `serial_number_font` | Font regularity of the serial number |
| `denomination` | Note value (10, 20, 50, 100, 200, 500, 2000) |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
pip install kagglehub
```

### 2. Run Full Pipeline

**Windows:**
```cmd
run_training.bat
```

**Linux/Mac:**
```bash
bash run_training.sh
```

Or run each step manually:

```bash
python scripts/prepare_dataset.py    # Step 1: Download + prepare Kaggle dataset
python models/train_cnn.py           # Step 2: Train CNN (image classifier)
python models/train_svc.py           # Step 3: Train SVC (tabular features)
python models/train_rf.py            # Step 4: Train Random Forest (tabular features)
python models/compare_models.py      # Step 5: Generate comparison report + chart
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Access the Application

- **Frontend:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/predict` | Predict genuine/counterfeit (image or tabular features) |
| `GET` | `/api/history` | Last 50 INR scan results |
| `GET` | `/api/compare` | Model comparison report with chart |
| `GET` | `/api/stats` | Scan statistics |
| `GET` | `/health` | Health check |

### POST /api/predict

**Image upload (CNN):**
```bash
curl -X POST http://localhost:8000/api/predict -F "file=@note.jpg"
```

**Tabular features (SVC / Random Forest):**
```bash
curl -X POST http://localhost:8000/api/predict \
  -F "intaglio_depth=0.85" \
  -F "security_thread_visible=0.90" \
  -F "watermark_clarity=0.88" \
  -F "color_shift_ink=0.82" \
  -F "microprint_score=0.87" \
  -F "uv_fluorescence=0.91" \
  -F "serial_number_font=0.89" \
  -F "denomination=500" \
  -F "model_name=SVC"
```

**Response:**
```json
{
  "result": "Genuine",
  "confidence": 0.8226,
  "model_used": "SVC",
  "currency": "INR",
  "timestamp": "2026-05-01T19:07:17.265841+00:00",
  "scan_id": "eb00835e-ba24-4cba-a2dd-52dead43da17"
}
```

## Project Structure

```
EDI_SEM4PROJECT/
├── data/
│   ├── train.csv / val.csv / test.csv
│   └── images/
│       ├── genuine/     (6,255 images — notes + feature close-ups)
│       └── fake/        (2,505 images)
├── models/
│   ├── train_cnn.py           # CNN training (image classifier)
│   ├── train_svc.py           # SVC training (tabular features)
│   ├── train_rf.py            # Random Forest training
│   ├── compare_models.py      # Model comparison report + chart
│   ├── saved/
│   │   ├── cnn_best.h5        # Trained CNN model
│   │   ├── cnn_model.tflite   # TFLite export
│   │   ├── svc_model.pkl      # Trained SVC
│   │   └── rf_model.pkl       # Trained Random Forest
│   └── results/
│       ├── cnn_metrics.json
│       ├── svc_metrics.json
│       ├── rf_metrics.json
│       ├── rf_feature_importance.json
│       ├── comparison_report.json
│       └── comparison_chart.png
├── backend/
│   ├── main.py               # FastAPI application
│   ├── database.py           # SQLite helper
│   ├── classifier.py         # Model inference wrapper
│   └── currency_detector.db  # SQLite database (auto-created)
├── frontend/
│   └── index.html            # Single-file dark-mode frontend
├── scripts/
│   └── prepare_dataset.py    # Kaggle dataset download + preparation
├── requirements.txt
├── run_training.sh / .bat
└── README.md
```

## Tech Stack

- **ML:** TensorFlow/Keras (CNN), scikit-learn (SVC, Random Forest)
- **Backend:** FastAPI, SQLite, Uvicorn
- **Frontend:** Vanilla HTML/CSS/JS — no build step required
- **Dataset:** Kaggle `preetrank/indian-currency-real-vs-fake-notes-dataset`
- **Python:** 3.10+

## Research Abstract

> "The circulation of counterfeit currency presents a significant threat to the stability of financial systems, public trust, and overall economic integrity. This study introduces a counterfeit detection framework that employs Convolutional Neural Networks (CNNs) for high-precision identification of forged Indian Rupee banknotes. The performance of the CNN model is rigorously compared against traditional machine learning algorithms — Support Vector Classifier (SVC) and Random Forest Classifier — to ensure robust validation. A mobile-based, real-time detection system is developed, leveraging machine learning to enable users at various touchpoints to verify Indian currency effortlessly."

## License

Academic use — EDI Semester 4 Project.
