@echo off
echo ============================================
echo  CurrencyGuard - Indian Rupee Detector
echo  Training Pipeline (Windows)
echo ============================================
echo.

REM Check kagglehub is installed
pip show kagglehub >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing kagglehub...
    pip install kagglehub
)

echo Step 1: Downloading + preparing Kaggle INR dataset...
set PYTHONIOENCODING=utf-8
python scripts\prepare_dataset.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in dataset prep & pause & exit /b 1 )

echo.
echo Step 2: Training CNN on INR images...
python models\train_cnn.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in CNN training & pause & exit /b 1 )

echo.
echo Step 3: Training SVC on INR features...
python models\train_svc.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in SVC training & pause & exit /b 1 )

echo.
echo Step 4: Training Random Forest on INR features...
python models\train_rf.py
if %ERRORLEVEL% NEQ 0 ( echo ERROR in RF training & pause & exit /b 1 )

echo.
echo Step 5: Generating INR model comparison report...
python models\compare_models.py

echo.
echo ============================================
echo  All models trained on Indian Rupee data
echo  Dataset: Kaggle (preetrank)
echo  Denominations: 10, 20, 50, 100, 200, 500, 2000
echo ============================================
echo.
echo Starting backend server at http://localhost:8000 ...
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
