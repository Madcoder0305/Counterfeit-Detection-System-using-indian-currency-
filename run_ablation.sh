#!/usr/bin/env bash
# ============================================================
#  Ablation Study — Fusion Framework Validation
#  Indian Rupee Counterfeit Detection (IEEE Paper)
# ============================================================

echo ""
echo "  =========================================="
echo "   Starting Ablation Study..."
echo "  =========================================="
echo ""

python models/ablation_study.py

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -ne 0 ]; then
    echo "  [ERROR] Ablation study failed with exit code $EXIT_CODE"
else
    echo "  [DONE] Results saved to models/results/ablation_results.json"
fi
echo ""
