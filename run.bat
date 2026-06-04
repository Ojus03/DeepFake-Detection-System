@echo off
title DeepFake Detection v3

set CUDA_VERSION=12.3
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v%CUDA_VERSION%
set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

:: ── Activate conda env ────────────────────────────────────────────────
call C:\Users\thopt\anaconda3\condabin\conda.bat activate tf_gpu
if errorlevel 1 (
    echo [ERROR] Could not activate conda env "tf_gpu"
    echo Run:  conda env list   to see available environments
    pause & exit /b 1
)

:: ── Add CUDA 12.3 DLLs to PATH ────────────────────────────────────────
set PATH=%CUDA_PATH%\bin;%PATH%
set PATH=%CUDA_PATH%\libnvvp;%PATH%
set PATH=%CUDA_PATH%\extras\CUPTI\lib64;%PATH%

:: ── Add Anaconda CUDA DLLs ────────────────────────────────────────────
set PATH=C:\Users\thopt\anaconda3\envs\tf_gpu\Library\bin;C:\Users\thopt\anaconda3\envs\tf_gpu\bin;%PATH%

:: ── Verify ptxas ──────────────────────────────────────────────────────
where ptxas.exe >nul 2>&1
if errorlevel 1 (
    echo [WARN] ptxas.exe not found — CUDA kernels may fall back to CPU
) else (
    echo [OK] ptxas.exe found
)

:: ── Install dependencies ──────────────────────────────────────────────
pip install gradio matplotlib scikit-learn mtcnn --quiet

:: ── Pin protobuf LAST ─────────────────────────────────────────────────
pip install protobuf==3.19.6 --no-deps --quiet
echo [OK] protobuf pinned to 3.19.6

:: ── Go to project folder ──────────────────────────────────────────────
cd /d "%~dp0"
echo [OK] Working directory: %CD%

:: ── Run main.py ───────────────────────────────────────────────────────
echo.
echo Starting DeepFake Detection v3...
echo.
python main.py

:: ── Launch UI ─────────────────────────────────────────────────────────
echo.
echo Launching DeepFake Detection UI...
echo Open browser at: http://localhost:7860
echo.
python app.py
pause