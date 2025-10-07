@echo off
echo ==========================================
echo Setting up Python environment (GPU version - NVIDIA CUDA)
echo ==========================================

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install base dependencies
pip install llama-cpp-python langchain-community langchain-huggingface transformers pillow pymupdf accelerate

REM Install FAISS GPU version
pip install faiss-gpu

REM Install CUDA-enabled PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo ==========================================
echo ✅ GPU (NVIDIA CUDA) Environment setup complete!
echo Environment is now active.
echo ==========================================
cmd /k ".venv\Scripts\activate"
