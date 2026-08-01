#!/usr/bin/env bash
# ============================================================
#  MobileNetV2 Training — Indian Rupee Counterfeit Detection
#  This script fine-tunes MobileNetV2 on the INR dataset
# ============================================================

echo ""
echo "  =========================================="
echo "   Starting MobileNetV2 Training..."
echo "  =========================================="
echo ""

python models/train_mobilenet.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "  [ERROR] Training failed with exit code $EXIT_CODE"
else
    echo "  [DONE] Training complete. Check models/saved/ for outputs."
fi
echo ""
