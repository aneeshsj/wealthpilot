
# WealthPilot Architecture

## Scope and outcome

WealthPilot is an SME loan-underwriting assistant. It is a decision-support
system for synthetic applications, not a production lending approval engine.
The system produces a structured recommendation, policy citations, financial
metrics, guardrail flags, and an audit trail. Borderline applications and loan
requests above the configured human-sign-off threshold pause for a credit
officer decision.

## Components and data flow

```text
Client
  |
  v
FastAPI app.py  ---- /health
  |              \-- /policy/query --> hybrid RAG --> Qdrant + BM25 + reranker
  |
  +--> input scan + Pydantic LoanApplication validation
  |
  v
LangGraph StateGraph
  intake
    -> policy RAG + MCP policy resource
    -> deterministic risk tools
    -> committee analyst (structured LLM output)
    -> risk/compliance critic (structured LLM output + hard checks)
        |-- revision loop, capped by MAX_GRAPH_REVISIONS
        |-- human interrupt for borderline/high-risk/high-value cases
        '-- finalization after approval/rejection
  |
  v
SQLite checkpoint + past-decision audit record
```

The graph uses a typed `UnderwritingState`. Audit entries append; control
fields such as the current verdict, reviewer feedback, and final decision are
overwritten. The checkpointer is keyed by LangGraph `thread_id`, allowing a
paused approval to resume after the process is restarted.

## External and mock systems

- Mock credit-bureau, bank-statement, loan-application, authority, and policy
  tables live under `data/wealthpilot/mock_api/`.
- Policy Markdown and PDF files live under `data/wealthpilot/corpus/`.
- Qdrant is opened in local-disk mode at `QDRANT_PATH`.
- FastMCP exposes deterministic underwriting tools and read-only policy
  resources in `src/mcp_server.py`.
- `src/mcp_client.py` consumes those resources and tools through a real MCP
  client/server protocol session. Local runs use the SDK in-process transport
  by default; `MCP_TRANSPORT=stdio` enables a separate MCP process. Transport
  failures return structured errors and never silently call server functions
  directly. `src/a2a.py` records the typed analyst-to-critic handoff.
- SQLite stores workflow checkpoints and final decision audit records.

## Guardrail coverage

1. User intake is scanned for direct prompt injection before schema validation.
2. Retrieved policy text is scanned before it is placed in an LLM context.
3. Demographic proxies are removed before financial scoring and committee work,
   including proxy references embedded in free-text narratives.
4. Tool calculations and policy thresholds are deterministic Python checks.
5. DSCR is calculated only from an explicit application debt-service amount or
   application interest rate. If neither basis is present, DSCR remains unavailable
   and the case is routed for review; no indicative basis is fabricated.
6. Verdict rationale is checked for protected/proxy language and rebuilt from
   sanitized financial facts plus valid policy citations so matched-pair
   applications receive identical rationale content.
7. Borderline or above-threshold decisions require human sign-off through
   `interrupt()`; finalization happens only after the interrupt resumes.
8. API guardrail failures return HTTP 422 with explicit `detail.flags` values.

The guardrails are defense-in-depth. Prompt instructions alone are not treated
as enforcement.

## Failure modes and recovery

| Failure | System behavior | Remaining limitation |
|---|---|---|
| Malformed intake | HTTP 422 with `malformed_application_payload` | Caller must correct the payload |
| Direct or indirect injection | Block or omit unsafe input/context and record flags | Pattern scanning cannot prove semantic safety |
| Missing mock record | Structured tool error | Application may require human review |
| Transient bureau/API failure | Retry, circuit breaker, and flagged fallback where configured | Fallback data is incomplete and must not be treated as a clean bureau result |
| LLM/schema failure | Bounded retry or deterministic fallback with `used_fallback=true` | Fallback recommendations require human scrutiny |
| Invalid citation, DSCR, or bias language | Critic rejects and sends work through the bounded revision path | Revision exhaustion escalates rather than silently approving |
| Process termination during approval | SQLite checkpoint preserves the thread | The same `thread_id` must be supplied on resume |
| Missing Qdrant collection | RAG query returns an explicit service error; underwriting retains MCP/core fallback behavior | The index must be built before policy QA |
| Missing or inconsistent DSCR inputs | DSCR remains unavailable and the case is routed for review | A valid debt-service basis must be present |
| LLM 429/503/provider outage | Classify the provider error, retry 503 once, stop 429 retries, and route to human review without exhausting graph revisions | A valid provider key/quota is still required for full LLM evaluation |

## Evaluation summary

The committed golden set contains 20 WealthPilot cases, including matched-pair
bias probes, guardrail probes, multi-hop cases, factual questions, and an
unanswerable case. `evals/evaluate.py` reports deterministic intake,
retrieval, routing, and guardrail metrics. `evals/baseline.json` and
`evals/improved.json` record the comparison for the guardrail change.

The optional LLM judge is not fabricated: it runs only when candidate outputs,
a judge model, and credentials are present. A result with
`"status": "not_run"` is an explicit missing-evidence state, not a passing
score.

## Known gaps and limitations

- MCP transport calls are time-bounded and return structured errors when the
  configured runtime is unavailable; the application does not silently replace
  an MCP call with a direct local function call.
- Memory is integrated at intake, policy review, and finalization; memory
  failures remain fail-safe and are surfaced as guardrail flags.
- Local trace export and deterministic fault-injection artifacts are included;
  remote Langfuse export remains optional and requires deployment credentials.
- The committee LLM defaults to Gemini Flash Lite Latest. Full LLM evaluation requires provider availability and quota; 429/503 failures are shown to the user and routed for review.
- Local Qdrant and SQLite paths are suitable for a demo, not horizontally
  scaled production deployment.
- The generated policy corpus is synthetic and must be replaced and reviewed
  before any real lending use.

## Deployment notes

For a local run, install `requirements.txt`, copy `.env.example` to `.env`,
and start Uvicorn with `app:app`. Set `DATA_DIR`, `SQLITE_DB_PATH`, and
`QDRANT_PATH` to writable mounted directories in a container or hosted
service. Local Qdrant is single-process and must not be opened concurrently by
multiple processes. Store provider keys in the deployment secret manager; never place
them in source, committed `.env` files, evaluation outputs, or logs.

Before deployment, run the test suite, build the Qdrant index, execute the
improved evaluation, verify `/health`, and exercise both approve and reject
human-resume paths using synthetic data.
