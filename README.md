# SHS Cessation OpenAI

The repository is now separated by responsibility so frontend, backend, and older experiments are no longer mixed together.

- `backend/` - FastAPI API, Streamlit app, Python core modules, environment file, and runtime logs
- `frontend/` - React/Vite chat client
- `legacy/` - archived Pinecone-based chatbot code and preserved duplicate root module copies
- `scripts/` - local developer launch scripts
- `.venv/` - project virtualenv

## Active apps

- FastAPI API: `backend/api_server.py`
- Streamlit app: `backend/merged_smoking_rag_app.py`
- React app: `frontend/`

## Run

### Full dev stack

```bash
./scripts/run_chat_stack.sh
```

This starts:

- API at `http://localhost:8000/health`
- React app at `http://localhost:5173`

Logs are written to:

- `backend/logs/api_server.out.log`
- `backend/logs/react_app.out.log`

### Backend API only

```bash
cd backend
../.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

### Streamlit app only

```bash
cd backend
../.venv/bin/python -m streamlit run merged_smoking_rag_app.py
```

### Frontend only

```bash
cd frontend
npm run dev
```

### Frontend production build

```bash
cd frontend
npm run build
```

## Verified on this layout

The current repo structure and commands were validated on May 2, 2026 with:

- `./.venv/bin/python -m py_compile backend/api_server.py backend/merged_smoking_rag_app.py backend/core/*.py`
- `./.venv/bin/python -m py_compile legacy/root_module_copies/*.py legacy/shs_chatbot_pinecone/*.py`
- `npm run build` from `frontend/`
- FastAPI startup plus `GET /health`
- Streamlit startup on localhost
- Vite dev server startup with HTTP `200` on `/`

## Notes

- `backend/.env` is used for backend runtime configuration, including `OPENAI_API_KEY`.
- `legacy/` is kept for reference. It was syntax-checked, but it is not part of the main FE/BE runtime path.
