@echo off
echo ==========================================
echo Setting up Python environment (CPU version)
echo ==========================================

REM Create virtual environment
python -m venv .venv

REM Activate virtual environment
call .venv\Scripts\activate

REM Upgrade pip
python -m pip install --upgrade pip

REM Install dependencies
pip install llama-cpp-python langchain-community langchain-huggingface faiss-cpu transformers pillow pymupdf torch accelerate

echo ==========================================
echo CPU Environment setup complete!
echo Environment is now active.
echo ==========================================
cmd /k ".venv\Scripts\activate"
