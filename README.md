# AI Drug Repurposing Agent

This repository contains a full-stack biomedical reasoning system for drug repurposing.

## Architecture

- `backend`: FastAPI service with in-memory graph-style reasoning over CSV datasets.
- `frontend`: Next.js UI for querying diseases and visualizing candidate drugs.

Reasoning path:

- Disease -> Protein from `diseaseToProtein.csv`
- Protein -> Drug from `proteinToDrug.csv`
- Optional protein expansion with STRING interaction partners

## Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```

Backend endpoints:

- `GET /health`
- `POST /query`

Example payload:

```json
{
  "disease": "Alzheimer disease",
  "species": 9606,
  "expand_with_string": true,
  "include_explanation": true
}
```

## Frontend Setup

```bash
cd frontend
npm install
cp .env.example .env.local
npm run dev
```

Open `http://localhost:3000`.

## Notes

- Data is loaded once at startup into in-memory dictionaries for fast lookups.
- No database is required for the current dataset size.
- LangGraph is included for agent orchestration and can be extended with richer LLM calls.
- STRING API calls are rate-limited with a one-second delay between calls.
