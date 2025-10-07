@echo off
echo ==========================================
echo Setting up Python environment (CPU / Intel GPU version)
echo ==========================================

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

echo Installing core dependencies...
pip install pillow pymupdf accelerate

echo Installing LangChain components...
pip install langchain-community langchain-huggingface faiss-cpu

echo Installing Transformers and Torch (CPU)
pip install transformers torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

echo Installing Intel GPU-compatible llama-cpp-python build...
pip install --force-reinstall llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/oneapi

echo ==========================================
echo ✅ Environment setup complete!
echo -> Virtual environment: .venv
echo -> Llama.cpp supports Intel iGPU (oneAPI / OpenCL)
echo ==========================================

REM Open a new shell with environment activated
cmd /k ".venv\Scripts\activate"
