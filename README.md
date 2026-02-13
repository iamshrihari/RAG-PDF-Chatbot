# RAG PDF Chatbot

A simple Retrieval-Augmented Generation (RAG) PDF chatbot using LangChain, FAISS, and local or remote LLMs.

## What this project does
- Loads a PDF, splits it into document chunks
- Creates embeddings and stores them in a FAISS vector store
- Retrieves relevant passages for a user question
- (Optional) Uses a local LLM (via Ollama) or OpenAI to generate answers from the retrieved context

## Quick Start
1. Create and activate a Python virtual environment (Windows PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. (Optional) If using Ollama local inference, install Ollama and pull a model (example: `mistral`) in a new terminal:

```powershell
# Install Ollama (follow installer from https://ollama.ai)
ollama pull mistral
```

4. Run the Streamlit app:

```powershell
C:/Users/spada/Desktop/rag_pdf_chatbot/venv/Scripts/python.exe -m streamlit run streamlit_app.py
```

Open the shown URL (usually http://localhost:8501).

## Configuration
- `config.py` contains embedding model configuration:
  - `EMBEDDING_MODEL` — HuggingFace model name used for embeddings (default set to a lightweight model).
  - You can add `OLLAMA_MODEL` or switch to OpenAI by adding your API key and restoring `ChatOpenAI` usage.

## Files
- `streamlit_app.py` — Streamlit UI and main RAG flow.
- `utils.py` — PDF loading and document splitting helpers.
- `config.py` — Small config (embedding model, keys if you add them).
- `requirements.txt` — Python dependencies for the project.

## Using Ollama vs OpenAI
- Ollama (local): lower latency and no API cost, requires installing Ollama and pulling a model (e.g., `mistral`). The app currently attempts to use Ollama if available.
- OpenAI (remote): add your API key in `config.py` and change the LLM code to use `ChatOpenAI`.

## Troubleshooting
- `ImportError: langchain.document_loaders` or similar: LangChain v1.x reorganized modules. The project uses `langchain_community` and `langchain_openai` packages for compatibility.
- `ImportError: pypdf package not found`: install `pypdf` (`pip install pypdf`).
- `InstructorEmbedding` dependency errors: install `InstructorEmbedding` and `sentence-transformers` if you use instructor-style embeddings.
- If Streamlit appears to load stale code, stop the server (Ctrl+C) and restart. Remove Python caches if needed:

```powershell
Get-ChildItem -Recurse -Directory -Name __pycache__ | ForEach-Object { Remove-Item -Path $_ -Recurse -Force }
Get-ChildItem -Path . -Recurse -Include "*.pyc" | Remove-Item -Force
```



