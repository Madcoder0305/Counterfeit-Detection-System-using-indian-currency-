@echo off
REM ============================================================
REM  Ablation Study — Fusion Framework Validation
REM  Indian Rupee Counterfeit Detection (IEEE Paper)
REM ============================================================

echo.
echo  ==========================================
echo   Starting Ablation Study...
echo  ==========================================
echo.

python models\ablation_study.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Ablation study failed with exit code %ERRORLEVEL%
) else (
    echo  [DONE] Results saved to models\results\ablation_results.json
)
echo.
pause
