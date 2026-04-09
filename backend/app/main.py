from __future__ import annotations

import os
import json
import queue
import threading
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv

from app.models import DiseaseSuggestionsResponse, QueryRequest, QueryResponse
from app.services.agent import DrugRepurposingAgent
from app.services.data_store import StoreBuilder
from app.services.reasoner import DrugRepurposingReasoner
from app.services.string_client import StringClient, StringConfig

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DISEASE_FILE = DATA_DIR / "diseaseToprotein.csv"
PROTEIN_DRUG_FILE = DATA_DIR / "proteinToDrug.csv"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR / ".env.example")

STRING_BASE_URL = os.getenv("STRING_BASE_URL", "https://version-12-0.string-db.org")
STRING_CALLER_IDENTITY = os.getenv("STRING_CALLER_IDENTITY", "drug-repurposing-agent")

app = FastAPI(title="Drug Repurposing Reasoning API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

builder = StoreBuilder(DISEASE_FILE, PROTEIN_DRUG_FILE)
store = builder.build()
string_client = StringClient(
    StringConfig(
        base_url=STRING_BASE_URL,
        caller_identity=STRING_CALLER_IDENTITY,
    )
)
reasoner = DrugRepurposingReasoner(store=store, string_client=string_client)
agent = DrugRepurposingAgent(reasoner=reasoner)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/disease-suggestions", response_model=DiseaseSuggestionsResponse)
def disease_suggestions(
    q: str = Query(default="", max_length=120),
    limit: int = Query(default=8, ge=1, le=20),
) -> DiseaseSuggestionsResponse:
    suggestions = reasoner.suggest_diseases(query=q, limit=limit)
    return DiseaseSuggestionsResponse(query=q, suggestions=suggestions)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    disease = request.disease.strip()
    if not disease:
        raise HTTPException(status_code=400, detail="Disease cannot be empty")

    agent_state = agent.run(
        disease=disease,
        species=request.species,
        expand_with_string=request.expand_with_string,
    )
    result = agent_state.get("result")
    explanation = agent_state.get("explanation") if request.include_explanation else None

    if result is None:
        raise HTTPException(status_code=500, detail="Reasoning agent failed to produce a result")

    return _build_query_response(
        disease=disease,
        explanation=explanation,
        result=result,
    )


def _build_query_response(disease: str, result, explanation: str | None) -> QueryResponse:
    mechanism_groups: dict[str, int] = {}
    tier_counts: dict[str, int] = {
        "Tier 1 — Strong": 0,
        "Tier 2 — Hypothesis": 0,
        "Tier 3 — Exploratory": 0,
    }
    tier_label = {1: "Tier 1 — Strong", 2: "Tier 2 — Hypothesis", 3: "Tier 3 — Exploratory"}
    for candidate in result.candidates:
        label = candidate.mechanism_group or "General protein-target interaction"
        mechanism_groups[label] = mechanism_groups.get(label, 0) + 1
        tier_counts[tier_label.get(candidate.tier, "Tier 3 — Exploratory")] += 1

    return QueryResponse(
        disease=disease,
        normalized_disease=result.normalized_disease,
        direct_proteins=result.direct_proteins,
        expanded_proteins=result.expanded_proteins,
        candidates=result.candidates,
        mechanism_groups=mechanism_groups,
        strategy=result.strategy,
        explanation=explanation,
        summary=result.summary,
        reasoning_trace=result.reasoning_trace,
        hypotheses=[asdict(item) for item in result.hypotheses],
        tier_summary=tier_counts,
    )


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    disease = request.disease.strip()
    if not disease:
        raise HTTPException(status_code=400, detail="Disease cannot be empty")

    event_queue: queue.Queue[dict[str, object]] = queue.Queue()
    done = threading.Event()

    def on_progress(step: str) -> None:
        event_queue.put({"type": "trace", "message": step})

    def worker() -> None:
        try:
            agent_state = agent.run(
                disease=disease,
                species=request.species,
                expand_with_string=request.expand_with_string,
                on_progress=on_progress,
            )
            result = agent_state.get("result")
            explanation = agent_state.get("explanation") if request.include_explanation else None
            if result is None:
                event_queue.put({"type": "error", "message": "Reasoning agent failed to produce a result"})
                return

            response_payload = _build_query_response(
                disease=disease,
                result=result,
                explanation=explanation,
            )
            event_queue.put({"type": "final", "data": response_payload.model_dump()})
        except Exception as exc:  # noqa: BLE001
            event_queue.put({"type": "error", "message": str(exc)})
        finally:
            done.set()

    threading.Thread(target=worker, daemon=True).start()

    def stream() -> object:
        while not done.is_set() or not event_queue.empty():
            try:
                event = event_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            yield json.dumps(event) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
