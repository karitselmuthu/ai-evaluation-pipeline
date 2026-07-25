# Future Enhancements Document: AI Evaluation Pipeline

This document outlines detailed architectural and functional proposals for future enhancements to the AI Evaluation Pipeline. The goal is to evolve the current prototype into a production-grade enterprise evaluation framework.

---

## 📋 Enhancements Mapping to Pipeline Architecture

```
[Input Dataset] -------> [LLM Application] -------> [Evaluation Engine]
       |                        |                         |
       v                        v                         v
Synthetic Scenario     Multi-Model Support        RAG-Specific Metrics
Generator              & A/B Testing              & Guardrail Filters
                                                          |
                                                          v
[Feedback Loop] <------- [Analysis / UI]   <------- [Score Aggregator]
       |                        |                         |
       v                        v                         v
GEPA Prompt            Time-Series Dashboard      PostgreSQL & MLflow
Auto-Optimizer         & Alerting Integrations    Longitudinal Database
```

---

## 1. Input Dataset Enhancements

### 1.1. Synthetic Scenario Generator (User-Simulation)
- **Concept**: Automate the creation of test datasets.
- **Implementation**:
  - Implement a generator script that reads system instructions and tool definitions to synthesize standard user personas (e.g., impatient user, malicious attacker, beginner).
  - An LLM-backed simulator plays the role of these personas in multi-turn chat trials against the LLM under test.
  - The traces are saved directly as new evaluation cases, expanding coverage from 10s to 1000s of cases without human data curation.

---

## 2. LLM / Application Client Enhancements

### 2.1. Multi-Provider & Model A/B Testing
> **Status: partially implemented.** Generation and judging already route through the local vault proxy (OpenAI-compatible), so multiple backends (groq, mistral, openrouter, …) are reachable today via `EVAL_GENERATOR_PROVIDER` / `EVAL_JUDGE_PROVIDER`, and the generator rotates providers on rate-limit. What remains is first-class **A/B orchestration** — running the same dataset across several providers/models in one command and diffing the results.
- **Concept**: Compare performance across different LLM backends.
- **Implementation**:
  - Extend `generator.py` to support Anthropic Claude (via `langchain-anthropic`), local models (via Ollama/llama.cpp), and custom hosted REST API endpoints.
  - Enable concurrent A/B runs: inputting the same dataset across `gemini-2.5-flash`, `gpt-4o-mini`, and `claude-3-5-sonnet` to compare quality, token counts, costs, and latencies side-by-side.

---

## 3. Evaluation Engine Enhancements

### 3.1. RAG-Specific Diagnostic Metrics
- **Concept**: Drill down on retrieval performance vs. synthesis quality.
- **Implementation**:
  - **Context Precision**: Rates how relevant the retrieved chunk contexts are to the user's question.
  - **Context Recall**: Measures if all key information needed to answer the question was successfully retrieved.
  - **Semantic Answer Similarity**: Uses embedding models to score semantic closeness rather than purely relying on LLM-as-a-judge rubrics.

### 3.2. Pre-Evaluation Guardrail Filters
- **Concept**: Isolate safety checks before running heavy LLM evaluations.
- **Implementation**:
  - Hook deterministic safety/toxicity checkers (e.g. Llama Guard, Perspective API, or regex patterns for PII) as pre-evaluation filters.
  - Instantly flag cases that trigger safety violations to bypass the rest of the evaluation loop, saving API costs and execution time.

---

## 4. Score Aggregator & Datastore Enhancements

### 4.1. Relational Database Backend
- **Concept**: Transition from flat JSON files to a relational datastore.
- **Implementation**:
  - Integrate a PostgreSQL or SQLite database using SQLAlchemy.
  - **Schema Layout**:
    - `datasets`: schema info, filepath, version hash.
    - `prompts`: instruction templates and revisions.
    - `runs`: target model, judge model, timestamp, aggregated quality score.
    - `eval_cases`: individual question, context, generated answer, and case-level scores for correctness, relevance, faithfulness, coherence, and safety.
  - Enables query-based filtering (e.g., *"Select all cases where faithfulness < 0.7 on Prompt version 3"*).

---

## 5. Analysis & Visualization Enhancements

### 5.1. Time-Series Tracking Dashboard
- **Concept**: Visualize quality improvements over prompt/model iterations.
- **Implementation**:
  - Create line charts showing how the overall quality index, latency, and cost change across successive runs.
  - Provide a side-by-side run difference viewer in the Streamlit UI to highlight which exact test cases regressed or improved after a prompt change.

### 5.2. Automated Alerting (Slack/Teams)
- **Concept**: Broadcast evaluation results.
- **Implementation**:
  - Integrate Webhook clients to post run summaries directly to engineering Slack channels.
  - Notify the team when scheduled cron evaluations succeed, or trigger warnings if average latencies exceed thresholds.

---

## 6. Feedback Loop & Automation Enhancements

### 6.1. Programmatic Prompt Optimizer (GEPA)
- **Concept**: Automated self-healing prompts.
- **Implementation**:
  - If a run fails to meet the target quality score (e.g. correctness < 0.85), feed the failed cases and judge explanations back into a prompt optimizer.
  - The optimizer generates 3 candidate prompt variations, runs evaluations on a validation subset, selects the highest-scoring candidate, and writes the improved template back to the source code automatically.
