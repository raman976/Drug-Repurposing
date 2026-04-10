"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from "react";

type CandidateDrug = {
  drug: string;
  matched_proteins: string[];
  source: string;
  support_count: number;
  score: number;
  confidence?: string;
  primary_target?: string | null;
  mechanism_group?: string | null;
  mechanism_alignment?: number | null;
  lifecycle_stage?: string | null;
  lifecycle_alignment?: number | null;
  explanation?: string | null;
  evidence_level?: string;
  tier?: number;
  caveats?: string[];
};

type QueryResponse = {
  disease: string;
  normalized_disease: string;
  direct_proteins: string[];
  expanded_proteins: string[];
  candidates: CandidateDrug[];
  mechanism_groups?: Record<string, number>;
  strategy?: string | null;
  explanation?: string | null;
  summary?: string | null;
  reasoning_trace?: string[];
  hypotheses?: {
    pathway: string;
    supporting_proteins: string[];
    candidate_drugs: string[];
    top_score: number;
    confidence: string;
    status: string;
    summary: string;
  }[];
  tier_summary?: Record<string, number>;
};

type StreamEvent =
  | { type: "trace"; message: string }
  | { type: "final"; data: QueryResponse }
  | { type: "error"; message: string };

const DEPLOYED_API_BASE = "http://drugapp.nstsdc.org";
const ENV_API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL;
const FALLBACK_API_BASE =
  process.env.NEXT_PUBLIC_API_FALLBACK_URL ?? "http://127.0.0.1:8000";
const API_BASES = Array.from(
  new Set([
    DEPLOYED_API_BASE,
    ENV_API_BASE,
    FALLBACK_API_BASE,
  ].filter((value): value is string => Boolean(value))),
);

async function fetchWithFallback(path: string, init?: RequestInit): Promise<Response> {
  let lastError: Error | null = null;

  for (const base of API_BASES) {
    try {
      const response = await fetch(`${base}${path}`, init);
      if (response.ok) {
        return response;
      }
      lastError = new Error(`Request failed (${response.status}) via ${base}`);
    } catch (error) {
      lastError = error instanceof Error ? error : new Error("Unknown network error");
    }
  }

  throw lastError ?? new Error("No API endpoint available");
}

export default function Home() {
  const [disease, setDisease] = useState("Alzheimer disease");
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [expandWithString, setExpandWithString] = useState(true);
  const [includeExplanation, setIncludeExplanation] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<QueryResponse | null>(null);
  const [liveTrace, setLiveTrace] = useState<string[]>([]);

  const candidateCount = useMemo(() => data?.candidates.length ?? 0, [data]);

  useEffect(() => {
    const trimmed = disease.trim();
    if (trimmed.length < 2) {
      setSuggestions([]);
      return;
    }

    const controller = new AbortController();
    const timeout = setTimeout(async () => {
      try {
        const response = await fetchWithFallback(
          `/disease-suggestions?q=${encodeURIComponent(trimmed)}&limit=8`,
          { signal: controller.signal },
        );
        const payload = (await response.json()) as { suggestions?: string[] };
        setSuggestions(payload.suggestions ?? []);
      } catch {
        setSuggestions([]);
      }
    }, 180);

    return () => {
      clearTimeout(timeout);
      controller.abort();
    };
  }, [disease]);

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setData(null);
    setLiveTrace(["Initializing biomedical reasoning agent..."]);

    try {
      const response = await fetchWithFallback("/query/stream", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          disease,
          species: 9606,
          expand_with_string: expandWithString,
          include_explanation: includeExplanation,
        }),
      });

      if (!response.ok) {
        const payload = (await response.json()) as { detail?: string };
        throw new Error(payload.detail ?? "Query failed");
      }

      if (!response.body) {
        throw new Error("Streaming response body is not available.");
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) {
          break;
        }
        buffer += decoder.decode(value, { stream: true });
        const chunks = buffer.split("\n");
        buffer = chunks.pop() ?? "";

        for (const line of chunks) {
          const payloadLine = line.trim();
          if (!payloadLine) {
            continue;
          }
          const eventPayload = JSON.parse(payloadLine) as StreamEvent;
          if (eventPayload.type === "trace") {
            setLiveTrace((prev) => [...prev, eventPayload.message].slice(-2));
          } else if (eventPayload.type === "final") {
            setData(eventPayload.data);
            const traceTail = (eventPayload.data.reasoning_trace ?? []).slice(-2);
            if (traceTail.length > 0) {
              setLiveTrace(traceTail);
            }
          } else if (eventPayload.type === "error") {
            throw new Error(eventPayload.message || "Streaming query failed");
          }
        }
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unknown error";
      setError(message);
      setData(null);
      setLiveTrace([]);
    } finally {
      setLoading(false);
    }
  }

  const topCandidates = useMemo(() => (data?.candidates ?? []).slice(0, 6), [data]);
  const otherCandidates = useMemo(() => (data?.candidates ?? []).slice(6, 18), [data]);

  return (
    <main className="page">
      <div className="background-grid" aria-hidden="true" />
      <div className="ambient-shape ambient-left" aria-hidden="true" />
      <div className="ambient-shape ambient-right" aria-hidden="true" />

      <div className="layout-shell">
        <div className="main-column">
          <section className="hero">
            <p className="eyebrow">Biomedical Reasoning Agent</p>
            <h1>Drug Repurposing Through Multi-Hop Biology</h1>
            <p className="subtitle">
              Explore disease to protein to drug links with optional STRING expansion and
              language-model explanations.
            </p>
          </section>

          <section className="panel">
            <form className="query-form" onSubmit={onSubmit}>
              <label htmlFor="disease">Disease</label>
              <input
                id="disease"
                value={disease}
                onChange={(event: ChangeEvent<HTMLInputElement>) =>
                  setDisease(event.target.value)
                }
                placeholder="e.g. Parkinson disease"
                list="disease-suggestions"
                autoComplete="off"
              />
              <datalist id="disease-suggestions">
                {suggestions.map((item) => (
                  <option value={item} key={item} />
                ))}
              </datalist>

              <div className="toggles">
                <label>
                  <input
                    type="checkbox"
                    checked={expandWithString}
                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                      setExpandWithString(event.target.checked)
                    }
                  />
                  Expand proteins with STRING
                </label>
                <label>
                  <input
                    type="checkbox"
                    checked={includeExplanation}
                    onChange={(event: ChangeEvent<HTMLInputElement>) =>
                      setIncludeExplanation(event.target.checked)
                    }
                  />
                  Include AI explanation
                </label>
              </div>

              <button type="submit" disabled={loading}>
                {loading ? "Reasoning..." : "Run Agent"}
              </button>
            </form>

            {error ? <p className="error">{error}</p> : null}

            <div className="live-trace" aria-live="polite">
              <p className="live-title">Live Process</p>
              {(loading || liveTrace.length > 0) ? (
                <>
                  <p className="trace-line">{liveTrace[liveTrace.length - 2] ?? "Waiting for next step..."}</p>
                  <p className="trace-line active">{liveTrace[liveTrace.length - 1] ?? "Waiting for next step..."}</p>
                </>
              ) : (
                <p className="trace-line">Run the agent to see real-time reasoning steps.</p>
              )}
            </div>
          </section>

          <section className="results">
            <div className="stat-card">
              <p>Candidate Drugs</p>
              <strong>{candidateCount}</strong>
            </div>
            <div className="stat-card">
              <p>Direct Proteins</p>
              <strong>{data?.direct_proteins.length ?? 0}</strong>
            </div>
            <div className="stat-card">
              <p>Expanded Proteins</p>
              <strong>{data?.expanded_proteins.length ?? 0}</strong>
            </div>
          </section>

          {data ? (
            <section className="panel">
              <p className="detail-line">
                Normalized disease: <strong>{data.normalized_disease}</strong>
              </p>
              {data.summary ? <p className="detail-line">{data.summary}</p> : null}
              {data.strategy ? <p className="detail-line subtle">Search strategy: {data.strategy}</p> : null}
            </section>
          ) : null}

          {data ? (
            <section className="panel output">
              <h2>Top Drug Candidates</h2>
              <div className="meta-row">
                {data.mechanism_groups ? (
                  <p>
                    Mechanism spread: {Object.entries(data.mechanism_groups)
                      .map(([group, count]) => `${group} (${count})`)
                      .join(" | ")}
                  </p>
                ) : null}
                {data.tier_summary ? (
                  <p>
                    Evidence tiers: {Object.entries(data.tier_summary)
                      .map(([tier, count]) => `${tier} (${count})`)
                      .join(" | ")}
                  </p>
                ) : null}
              </div>
              <ul className="candidate-grid">
                {topCandidates.map((candidate) => {
                  const visibleProteins = candidate.matched_proteins.slice(0, 5);
                  const hiddenCount = Math.max(0, candidate.matched_proteins.length - visibleProteins.length);
                  return (
                    <li key={candidate.drug} className="candidate-card">
                      <div className="candidate-head">
                        <h3>{candidate.drug}</h3>
                        <span className="score-pill">{candidate.score.toFixed(3)}</span>
                      </div>

                      <div className="pill-row">
                        <span className="pill">{candidate.confidence ?? "Low"}</span>
                        <span className="pill">{candidate.evidence_level ?? "Exploratory"}</span>
                        <span className="pill">Support {candidate.support_count}</span>
                      </div>

                      <p className="candidate-meta">
                        Target: <strong>{candidate.primary_target ?? "n/a"}</strong>
                      </p>
                      <p className="candidate-meta">
                        {candidate.mechanism_group ?? "General protein-target interaction"}
                      </p>
                      <p className="candidate-meta">
                        Stage: {candidate.lifecycle_stage ?? "n/a"} · Mechanism {typeof candidate.mechanism_alignment === "number"
                          ? candidate.mechanism_alignment.toFixed(2)
                          : "n/a"} · Lifecycle {typeof candidate.lifecycle_alignment === "number"
                          ? candidate.lifecycle_alignment.toFixed(2)
                          : "n/a"}
                      </p>

                      <div className="chip-row">
                        {visibleProteins.map((protein) => (
                          <span key={protein} className="chip">{protein}</span>
                        ))}
                        {hiddenCount > 0 ? <span className="chip">+{hiddenCount} more</span> : null}
                      </div>

                      {candidate.explanation ? <p className="candidate-note">{candidate.explanation}</p> : null}
                    </li>
                  );
                })}
              </ul>

              {data.explanation ? (
                <article className="explanation">
                  <h3>AI Explanation</h3>
                  <p>{data.explanation}</p>
                </article>
              ) : null}

              {data.hypotheses?.length ? (
                <article className="explanation">
                  <h3>Pathway Hypotheses</h3>
                  {data.hypotheses.map((item) => (
                    <div className="hypothesis-card" key={item.pathway}>
                      <h4>
                        {item.pathway} <span>({item.confidence})</span>
                      </h4>
                      <p>{item.summary}</p>
                      <p>Supporting proteins: {item.supporting_proteins.join(", ") || "None"}</p>
                      <p>Candidate drugs: {item.candidate_drugs.join(", ") || "None"}</p>
                    </div>
                  ))}
                </article>
              ) : null}

              {otherCandidates.length > 0 ? (
                <article className="explanation">
                  <h3>Other Candidates</h3>
                  <div className="other-table">
                    {otherCandidates.map((candidate, index) => (
                      <div className="other-row" key={`${candidate.drug}-${index}`}>
                        <p className="other-drug">{candidate.drug}</p>
                        <p className="other-meta">
                          {candidate.mechanism_group ?? "General"} · score {candidate.score.toFixed(3)} · support {candidate.support_count}
                        </p>
                      </div>
                    ))}
                  </div>
                </article>
              ) : null}
            </section>
          ) : null}
        </div>

        <aside className="side-column">
          <section className="side-card">
            <p className="side-label">Process Pulse</p>
            <p className="side-heading">{loading ? "Reasoning Live" : "Session Idle"}</p>
            <p className="side-line">{liveTrace[liveTrace.length - 1] ?? "Waiting for a query to begin."}</p>
            <p className="side-line muted">{liveTrace[liveTrace.length - 2] ?? ""}</p>
          </section>

          {data?.mechanism_groups ? (
            <section className="side-card">
              <p className="side-label">Mechanism Mix</p>
              <div className="mini-list">
                {Object.entries(data.mechanism_groups)
                  .sort((a, b) => b[1] - a[1])
                  .slice(0, 6)
                  .map(([group, count]) => (
                    <div key={group} className="mini-row">
                      <span>{group}</span>
                      <strong>{count}</strong>
                    </div>
                  ))}
              </div>
            </section>
          ) : null}

          {data?.direct_proteins?.length ? (
            <section className="side-card">
              <p className="side-label">Direct Protein Set</p>
              <div className="protein-cloud">
                {data.direct_proteins.slice(0, 14).map((protein) => (
                  <span key={protein} className="cloud-chip">{protein}</span>
                ))}
              </div>
            </section>
          ) : null}
        </aside>
      </div>
    </main>
  );
}
