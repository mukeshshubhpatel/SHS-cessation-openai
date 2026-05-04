# SHS Chatbot (Pinecone) - Run Guide

This folder contains setup and run instructions for the SHS chatbot module under:
- smoking cessation 26/SHS_chatbot_pinecone-main

## 1) Current Project Status

The files in this folder are currently stub implementations:
- chatbot_orchestrator.py
- conversation_manager.py
- prompt_engine.py
- ollama_test.py

That means they return placeholder values and do not call live services yet.

## 2) What This Module Is Used For

This module is imported by the merged Streamlit app at:
- smoking cessation 26/merged_smoking_rag_app.py

The merged app adds this folder to Python path and imports:
- conversation_manager
- chatbot_orchestrator
- prompt_engine
- ask_ollama

## 3) Prerequisites

Install the following before running the integrated app:

- Python 3.10 or newer
- pip
- PowerShell (Windows)
- Optional for full LLM flow: Ollama installed and running

If you later replace stubs with full Pinecone + embeddings logic, also prepare:
- Pinecone API key and index configuration
- Hugging Face model download capability

## 4) Environment Setup (Windows)

From workspace root (Website):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install core dependencies used by merged app:

```powershell
pip install streamlit requests certifi numpy pinecone transformers torch
```

Notes:
- statistics is a Python standard library module, so no install is needed.
- If torch install fails, use the official PyTorch install selector for your system.

## 5) Quick Module Smoke Test

Run from:
- smoking cessation 26/SHS_chatbot_pinecone-main

```powershell
python -c "from chatbot_orchestrator import chatbot_orchestrator; print(chatbot_orchestrator.run('test'))"
python -c "from prompt_engine import prompt_engine; print(prompt_engine.build('hello', []))"
python -c "from ollama_test import ask_ollama; print(ask_ollama('hello', [], []))"
```

Expected with current stubs:
- Empty string outputs or simple passthrough behavior

## 6) Run the Integrated Streamlit App

From workspace root (Website):

```powershell
.venv\Scripts\Activate.ps1
streamlit run "smoking cessation 26\merged_smoking_rag_app.py"
```

If streamlit command is not found:

```powershell
.venv\Scripts\python.exe -m streamlit run "smoking cessation 26\merged_smoking_rag_app.py"
```

## 7) Ollama Setup (Optional, for full generation)

If you use local LLM responses:

```powershell
ollama serve
ollama pull qwen2.5:3b
```

The merged app is configured to use model:
- qwen2.5:3b

## 8) Troubleshooting

1. Import errors for SHS module files
- Make sure you run from the Website root or from smoking cessation 26.
- Confirm folder name is exactly SHS_chatbot_pinecone-main.

2. Streamlit not found
- Activate .venv first.
- Or run with python -m streamlit.

3. SSL certificate issues
- certifi package is used by merged app to patch stale SSL_CERT_FILE values.

4. Pinecone/transformers/torch import errors
- Reinstall the packages.
- Verify Python environment is the same one where packages were installed.

5. No chatbot output
- Current SHS files are stubs and can legitimately return empty output.
- Replace stubs with real implementation to enable full behavior.

## 9) Recommended Next Step

If you want this folder to be runnable as a standalone chatbot (not only imported), add:
- a requirements.txt dedicated to SHS_chatbot_pinecone-main
- a small entrypoint script (for example run_chatbot.py)
- environment variable docs (.env.example)
