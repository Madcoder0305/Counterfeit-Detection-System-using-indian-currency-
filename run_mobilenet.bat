@echo off
REM ============================================================
REM  MobileNetV2 Training — Indian Rupee Counterfeit Detection
REM  This script fine-tunes MobileNetV2 on the INR dataset
REM ============================================================

echo.
echo  ==========================================
echo   Starting MobileNetV2 Training...
echo  ==========================================
echo.

python models\train_mobilenet.py

echo.
if %ERRORLEVEL% NEQ 0 (
    echo  [ERROR] Training failed with exit code %ERRORLEVEL%
) else (
    echo  [DONE] Training complete. Check models\saved\ for outputs.
)
echo.
pause
