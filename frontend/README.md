# SHS React Chat App

## Run

1. Start API server from repo root:
```bash
cd backend
../.venv/bin/python -m uvicorn api_server:app --host 0.0.0.0 --port 8000
```

2. Start React app:
```bash
cd frontend
npm install
npm run dev
```

3. Open:
`http://localhost:5173`

The app streams responses from:
`GET http://localhost:8000/chat/stream`
