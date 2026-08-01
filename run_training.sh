#!/bin/bash
echo "============================================"
echo " CurrencyGuard ₹ — Indian Rupee Detector"
echo " Training Pipeline"
echo "============================================"

# Ensure kagglehub is installed
pip show kagglehub > /dev/null 2>&1 || pip install kagglehub

export PYTHONIOENCODING=utf-8

echo ""
echo "Step 1: Downloading + preparing Kaggle INR dataset..."
python scripts/prepare_dataset.py || { echo "ERROR in dataset prep"; exit 1; }

echo ""
echo "Step 2: Training CNN on INR images..."
python models/train_cnn.py || { echo "ERROR in CNN training"; exit 1; }

echo ""
echo "Step 3: Training SVC on INR features..."
python models/train_svc.py || { echo "ERROR in SVC training"; exit 1; }

echo ""
echo "Step 4: Training Random Forest on INR features..."
python models/train_rf.py || { echo "ERROR in RF training"; exit 1; }

echo ""
echo "Step 5: Generating INR model comparison report..."
python models/compare_models.py

echo ""
echo "============================================"
echo " All models trained on Indian Rupee data"
echo " Dataset: Kaggle (preetrank)"
echo " Denominations: 10, 20, 50, 100, 200, 500, 2000"
echo "============================================"
echo ""
echo "Starting backend server at http://localhost:8000 ..."
cd backend && uvicorn main:app --host 0.0.0.0 --port 8000 --reload
