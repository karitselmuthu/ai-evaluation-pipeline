# AI Evaluation Pipeline

An end-to-end, automated evaluation pipeline to assess LLM outputs for quality, accuracy, safety, and performance as outlined in the system architecture.

---

## ⚡ System Architecture

The pipeline consists of modular components designed for high performance, parallel evaluation, logging, and visual analytics:

```
[Input Dataset] ---> [LLM Application Client] ---> [Evaluation Engine]
(JSON/JSONL/CSV)     (Vault Proxy, rotating)       (Pinned Judge, parallel)
                                                             |
                                                             v
[Feedback Loop] <--- [Streamlit Dashboard]    <--- [Score Aggregator]
(Prompt Iteration)   (Plotly Graphs & Traces)      (Pandas & Reports)
```

1. **Input Dataset Loader** (`src/core/dataset.py`): Parses evaluation datasets and validates schemas (questions, contexts, ground truth).
2. **LLM Application Client** (`src/core/generator.py`): Queries the target model through the local vault proxy (OpenAI-compatible), measures token counts, and tracks execution latency. Generation **may rotate providers** on rate-limit (`429`), so a capped provider does not stall the run.
3. **Evaluation Engine** (`src/core/evaluator.py`): Asynchronously runs parallel metric evaluations using LLM-as-a-judge rubrics. The judge is **pinned to a single provider** (no fallback) so scores stay comparable within a run; a call served by a different provider is treated as an error rather than silently mixed in.
4. **Score Aggregator** (`src/core/aggregator.py`): Computes summaries, averages, and standard deviations via Pandas/NumPy.
5. **Streamlit Frontend** (`src/dashboard/app.py`): Premium dark-mode UI with radar charts, detail viewers, and an interactive prompt sandbox.
6. **FastAPI Backend** (`src/api/app.py`): REST API providing endpoints for automated triggers and report management.

### 🔑 Secrets & Provider Routing

API keys are resolved at startup through a pluggable secret seam (`src/core/secret_provider.py`), selected by the `SECRET_BACKEND` env var:

- **`vault`** (default in `.env`): keys are fetched from a local OS-keychain-backed **LLM Key Vault** daemon (`http://127.0.0.1:9876`) using the token at `~/.llm_vault_token`. Nothing sensitive lives in the repo. **The daemon must be running** before you start the app, or startup fails loudly.
- **`env`**: keys are read from `.env` / the environment (`GEMINI_API_KEY`, `GOOGLE_API_KEY`, etc.). Use this if you are not running the vault.

Generation and judging are routed through the vault's OpenAI-compatible proxy. Attribution (which provider actually served each call) is summarized into every run report under `providers`, and mixed-provider runs raise a warning.

---

## 🛠️ Getting Started

### 1. Prerequisites
- Python 3.10+
- The **LLM Key Vault daemon running** at `http://127.0.0.1:9876` with a token at `~/.llm_vault_token` (default `SECRET_BACKEND=vault`). If you are not using the vault, set `SECRET_BACKEND=env` and provide a valid `GEMINI_API_KEY` in `.env` instead.

### 2. Setup
Create a virtual environment and install the required dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install langchain-google-genai google-antigravity
```
> `langchain-google-genai` / `google-antigravity` are only needed for the RAG retriever/reranker (`src/core/retriever.py`), which still calls Gemini directly. The core generate/judge path uses the proxy and needs only `requests` + `nest_asyncio` (already in `requirements.txt`).

Configure your environment variables in `.env`. The defaults use the vault + the OpenAI-compatible proxy:
```env
# Secret backend: fetch keys from the local vault daemon (no keys in the repo).
SECRET_BACKEND=vault
VAULT_URL=http://127.0.0.1:9876

# Generator MAY rotate providers on 429; "default" lets the proxy choose.
EVAL_GENERATOR_PROVIDER=groq
EVAL_GENERATOR_MODEL=default

# Judge is PINNED (no fallback) so scores stay comparable across a run.
EVAL_JUDGE_PROVIDER=groq
EVAL_JUDGE_MODEL=llama-3.3-70b-versatile

# Max concurrent proxy calls across all cases/metrics (keeps under rate limits).
EVAL_MAX_CONCURRENCY=6
```
To run without the vault instead, set `SECRET_BACKEND=env` and provide `GEMINI_API_KEY` / `GOOGLE_API_KEY` directly.

### 3. Running the Pipeline via CLI
Run the pipeline headlessly on a dataset:
```bash
PYTHONPATH=. ./.venv/bin/python main.py --dataset tests/data/sample_dataset.json
```
This saves reports to the `output/` directory in JSON and Markdown formats.

### 4. Running the Streamlit Dashboard
Launch the visual web dashboard:
```bash
PYTHONPATH=. ./.venv/bin/streamlit run src/dashboard/app.py
```

### 5. Running the FastAPI Server
Launch the backend REST API:
```bash
PYTHONPATH=. ./.venv/bin/uvicorn src.api.app:app --reload --port 8000
```
Visit `http://127.0.0.1:8000/docs` to view the interactive Swagger API documentation.

### 6. Running Unit Tests
Execute the test suites:
```bash
PYTHONPATH=. ./.venv/bin/pytest tests/
```

---

## 🚀 Future Enhancements Roadmap

The following enhancements are proposed to scale and advance this evaluation pipeline:

### 1. CI/CD Integration (Gatekeeper Evals)
- **Goal**: Integrate the evaluation pipeline into Github Actions or GitLab CI/CD.
- **Details**: Automatically trigger `main.py` on pull requests. Check if the computed quality index drops below a defined threshold (e.g., `0.80`) or if safety violations occur, and fail the build if regression is detected.

### 2. Relational Database Backend
- **Goal**: Move away from local flat JSON report files to a relational datastore (SQLite/PostgreSQL).
- **Details**: Expose a backend database schema tracking datasets, prompts, models, runs, and case-level scores. Enables cross-run comparison queries, longitudinal trend analysis, and historical dashboard charts.

### 3. Automated User Simulation & Synthetic Datasets
- **Goal**: Generate high-coverage evaluation datasets without manual writing.
- **Details**: Use an LLM-backed simulator to play user personas, interacting with the application to synthesize multi-turn dialogue trees and generate trace cases for the evaluation engine.

### 4. Advanced RAG Metrics
- **Goal**: Provide targeted diagnostics for Retrieval-Augmented Generation (RAG) systems.
- **Details**: Implement context recall, context precision, and semantic similarity metrics to isolate whether failures occur in the retrieval search vs. the generation step.

### 5. Programmatic Prompt Optimizer (GEPA)
- **Goal**: Automatically fix prompts that fail evaluation.
- **Details**: Hook the sandbox feedback loop to an optimization algorithm. The engine will iterate and mutate prompt instructions automatically, running evals on each generation until all rubrics pass.
