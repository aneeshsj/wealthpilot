# WealthPilot Capstone

WealthPilot is a local-first SME loan underwriting system developed as part of the Applied AI Professional Certification capstone. It turns a messy loan application into a typed intake object, retrieves financial-doc and credit-policy evidence, computes deterministic risk metrics, routes the case through a bounded LangGraph committee workflow, and pauses for credit-officer approval when the WealthPilot human-approval trigger is reached.

The implementation uses synthetic data only: generated intake records, a committed policy corpus, mock credit-bureau and banking systems, and a committed golden set.

## WealthPilot scope

The domain-specific expectations are:

- Agent roster: Application Intake, Financial-Doc RAG, Risk Scoring, and a Committee made up of Analyst, Risk Reviewer, and Compliance Reviewer with a loop-back path.
- Human approval: borderline cases and cases above the configured loan-size threshold pause before finalization.
- MCP system: mock credit-bureau, bank-statement, sector-policy, and policy reference services.
- Policy guardrail: decisions must not reference protected attributes or their proxies; flagged cases are refused or routed to human review.
- Golden-set requirement: matched-pair bias probes are evaluated by pair agreement, alongside retrieval, routing, guardrail, intake, and final-answer checks.

## Architecture

```
FastAPI / Streamlit
        |
        v
Input validation and injection scan
        |
        v
Typed LoanApplication intake
        |
        v
LangGraph underwriting workflow
  intake -> policy RAG -> risk scoring -> analyst <-> critic
                                      |                 |
                                      +--> human approval -> finalize
        |
        +--> SQLite checkpoint and four-type memory
        +--> MCP client -> mock bureau, bank, sector and policy systems
        +--> Langfuse root trace, node spans and LLM generations
```

The graph keeps deterministic decisions in nodes and conditional edges. LLM calls are used for bounded structured analysis; schema validation, policy thresholds, guardrails, routing limits, checkpointing, and human approval are implemented in code.

## Reproduce from a clean checkout

Python 3.11 or newer is required. The dependencies are pinned in requirements.txt file.

```

python -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
.venv/bin/python -c "from src.rag import build_index; print(build_index())"
```

Populate `.env` with the provider key used for live LLM calls. The default configuration is Gemini with an optional fallback of Groq. Local runs use the MCP in-process protocol transport by default; set `MCP_TRANSPORT=stdio` when a separate MCP process is required. MCP requests use `MCP_TIMEOUT_SECONDS` (10 seconds by default).

Langfuse is optional for local execution, but the committed integration is enabled automatically when `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.

## Run the service

Start the FastAPI service:

```
.venv/bin/uvicorn app:app --host 127.0.0.1 --port 8000
```

Check service health:

```
curl http://127.0.0.1:8000/health
```

The service exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Service and active-model health check |
| `POST /underwrite` | Validate and run an underwriting request |
| `POST /underwrite/resume` | Resume a paused approval thread with approve or reject |
| `POST /policy/query` | Query the policy corpus through the RAG path |

Malformed or prompt-injection input returns HTTP 422 with structured guardrail flags. A valid high-risk or borderline request returns `paused_for_human` and a `thread_id`; the thread is resumed through `/underwrite/resume`.

Run the interactive interface:

```
.venv/bin/streamlit run streamlit_app.py
```

Run the MCP server directly for a stdio integration check:

```
.venv/bin/python -m src.mcp_server
```

The MCP server writes diagnostics to stderr so the JSON-RPC stream remains valid. The application-side MCP client is implemented in `src/mcp_client.py`.

## Evaluation commands

Run the complete automated test suite and hygiene checks:

```
.venv/bin/pytest -q
.venv/bin/python scripts/check_hygiene.py
```

Run the deterministic baseline and improved evaluations on the same committed golden set:

```
.venv/bin/python evals/evaluate.py --variant baseline --output evals/baseline.json
.venv/bin/python evals/evaluate.py --variant improved --output evals/improved.json
```

The improved evaluation uses the committed candidate artifact:

```
.venv/bin/python evals/evaluate.py \
  --variant improved \
  --llm-judge \
  --output evals/improved.json
```

The evaluator loads `.env`, uses the configured judge model, retries transient provider failures, and records `not_run` instead of fabricating a score when a required artifact or credential is absent. Candidate generation is resumable:

```
.venv/bin/python evals/generate_candidate_outputs.py
```

Run the seeded production-hardening harness:

```
.venv/bin/python scripts/failure_injection.py
```

## 

## Current committed evaluation evidence

The committed result files are:

- `data/wealthpilot/eval/golden_set.json`: 20 end-to-end cases, including adversarial and matched-pair bias probes.
- `evals/baseline.json`: deterministic baseline component results.
- `evals/improved.json`: improved component results and LLM-judge results.
- `evals/candidate_outputs.jsonl`: one model-generated candidate per golden-set case.
- `evals/failure_injection.json`: seeded reliability results before and after hardening.

### M1 Provider-agnostic LLM client and structured intake

- `src/llm_client.py` is the provider-agnostic LiteLLM adapter. The default Gemini model is `gemini-flash-lite-latest`. 429 quota/rate-limit and 5xx provider failures are explicitly classified for the UI and routed to review without consuming the full revision budget.
- `src/schemas.py` defines the typed `LoanApplication` and related decision models. Structured responses are validated with Pydantic and sent through a bounded repair loop in `generate_structured`.
- `src/graph.py` consumes validated fields and structured outputs rather than raw model text.
- Relevant checks: `tests/test_config.py`, `tests/test_llm_client.py`, `tests/test_schemas.py`, and `tests/test_graph.py`.

### M2 Tool-enabled single-agent capability

- `src/tools.py` contains typed deterministic underwriting tools for DSCR, financial ratios, bureau data, bank statements, policy limits, FX conversion, and decision support.
- DSCR is calculated only from an explicit application debt-service amount or application interest rate. If neither basis is present, DSCR remains unavailable rather than being fabricated, and the case is routed for review.
- `src/mcp_server.py` exposes domain tools with typed parameters, docstrings, structured return values, validation, structured errors, timeouts, retries, and circuit-breaker protection.
- Iteration and graph recursion are bounded through named configuration values.
- Relevant checks: `tests/test_tools.py`, `tests/test_mcp_server.py`, `tests/test_graph.py`, and `evals/failure_injection.json`.

### M3 Persistent memory and semantic index

- `src/memory.py` implements working, episodic, semantic, and procedural memory in SQLite with user/entity namespacing and restart persistence.
- `src/graph.py` integrates memory with underwriting so retrieved facts and procedures are available to the workflow rather than stored as disconnected records.
- `src/rag.py` builds the Qdrant semantic index over the committed policy corpus, with metadata payloads. Local Qdrant is intentionally used in single-process mode; no Qdrant server is required.
- Relevant checks: `tests/test_memory.py`, `tests/test_memory_restart.py`, and `tests/test_rag.py`.

### M4 Production RAG and evaluation baseline

- `src/rag.py` supports Markdown/PDF ingestion, structure-aware chunks, source and section metadata, hybrid dense plus BM25 retrieval, fusion, reranking, grounded prompts, citation handling, and abstention for unsupported policy questions.
- `data/wealthpilot/corpus/` contains the synthetic policy corpus and `data/wealthpilot/eval/golden_set.json` contains the retrieval cases.
- `evals/baseline.json` and `evals/improved.json` score the same golden set; the improved run includes the deterministic citation metric and LLM judge.

### M5 Orchestrated LangGraph workflow and checkpointing

- `src/graph.py` defines the typed `UnderwritingState`, deterministic routers, bounded analyst-critic loop, explicit recursion limit, SQLite checkpointing, and terminal paths to `END`.
- The human-approval node uses an interrupt/resume path. Borderline cases and cases above `HUMAN_SIGNOFF_THRESHOLD_INR` pause before finalization; approve and reject decisions are handled through `/underwrite/resume`.
- Audit entries capture the approval decision, reviewer, timestamp, and reason.
- Relevant checks: `tests/test_graph.py`, `tests/test_hitl_restart.py`, and `tests/test_memory_restart.py`.

### M6 Multi-agent team and MCP integration

- The LangGraph topology is an intake, policy-RAG, risk-scoring, analyst, critic/reviewer, human-approval, and finalization workflow. Analyst and critic handoff data is typed in `src/a2a.py`.
- Agent context and writes are scoped in `src/graph.py`; revision counts and legal routes are bounded and validated.
- `src/mcp_server.py` exposes typed mock credit-bureau, bank-statement, sector policy, and policy-reference capabilities. `src/mcp_client.py` consumes the tools through a real MCP client/server protocol session and reads policy resources by URI. No direct server-function fallback is used.
- The server exposes MCP prompt templates for policy-grounded underwriting and uses stderr-only diagnostics. MCP version `1.26.0` is pinned.
- Relevant checks: `tests/test_mcp_server.py`, `tests/test_graph.py`, and the MCP command in the run section above.

### M7 Observability and reliability hardening

- `src/observability.py` provides local PII-safe structured traces and mirrors them to Langfuse when credentials are configured. Each underwriting request creates a root trace with nested graph-node spans, resume spans, and LLM generation spans carrying session and user identifiers.
- Langfuse setup follows the workbook pattern: `get_client()`, `start_as_current_observation(...)`, explicit close, and `flush()`. A readiness check is:\
  .venv/bin/python -c "from langfuse import get\_client; print(get\_client().auth\_check())"
- `src/observability.py` also provides structured logging, redaction, timeout handling, transient-only retries with exponential backoff, circuit-breaker states, and fallback accounting.
- `scripts/failure_injection.py` records seeded before/after hardening results; `TRACE_OUTPUT_PATH` can persist masked local traces as JSONL.
- Langfuse credentials and personal data are never included in trace inputs or repository files.

### M8 End-to-end evaluation, guardrails, and deployment package

- `data/wealthpilot/eval/golden_set.json` contains 20 domain cases, including prompt-injection, policy-abstention, protected-proxy, routing, and matched-pair probes.
- `src/guardrails.py` scans user input and retrieved content, removes protected proxies from fields and free text, validates output policy, and routes blocked cases to refusal or human review. Thresholds are centralized in `src/config.py`.
- Final committee rationales are canonicalized from sanitized financial facts and valid policy sources, so matched applicants receive identical rationale content when only demographic proxies differ.
- `app.py` provides FastAPI `/health`, `/underwrite`, `/underwrite/resume`, and `/policy/query` endpoints with explicit 4xx guardrail responses.
- `ARCHITECTURE.md` documents components, data flow, failure modes, evaluation, guardrail coverage, known gaps, and deployment notes.
- Pinned dependencies, `.env.example`, test data, evaluation artifacts, `pytest.ini`, and reproducible commands are committed for clean-clone evaluation.

## Repository layout

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI service and domain endpoints |
| `streamlit_app.py` | Interactive underwriting interface |
| `src/config.py` | Environment settings and named policy thresholds |
| `src/schemas.py` | Typed intake, state, tool and decision schemas |
| `src/llm_client.py` | Provider-agnostic structured LLM adapter |
| `src/graph.py` | LangGraph workflow, routing, memory and HITL |
| `src/rag.py` | Corpus ingestion, indexing, retrieval and grounded answers |
| `src/tools.py` | Deterministic financial and domain tools |
| `src/mcp_server.py` | MCP tools, resources and prompts |
| `src/mcp_client.py` | MCP client transport and application consumption |
| `src/a2a.py` | Analyst-to-critic typed handoff |
| `src/memory.py` | Persistent four-type memory |
| `src/observability.py` | Local tracing, Langfuse export, redaction and resilience |
| `data/wealthpilot/` | Synthetic corpus, mock systems, intake data and golden set |
| `evals/` | Baseline, improved, candidate and failure-injection outputs |
| `tests/` | Unit, integration, guardrail, memory, MCP and graph tests |
| `ARCHITECTURE.md` | Architecture and deployment documentation |

## Known limitations

The evaluator does not fabricate scores when candidate evidence or credentials are missing. Citation completeness is enforced at answer time by appending retrieved sources omitted by the model. Langfuse generation spans are wired and exported; detailed token and cost dashboards depend on provider usage fields and the configured Langfuse project.
