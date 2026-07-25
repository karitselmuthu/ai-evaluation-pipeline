# Cost Estimation Report: RAG Configurations Evaluation

This document provides a detailed breakdown of the token consumption and API cost estimation for running the 3-way RAG configuration evaluation on the full `sample_dataset.json` (53 test cases).

> **⚠️ Note (routing update):** The figures below model an all-Gemini run. The current default runtime routes **Answer Generation** and **Judge Evaluation** through the local vault proxy (default `groq/llama-3.3-70b-versatile`) — so those two phases no longer incur Gemini `gemini-3.5-flash` cost; the applicable price is that of the serving provider. **Context Reranking (`src/core/retriever.py`) still calls Gemini directly**, so Phase 2 remains the one path that hits the Gemini spend cap. Treat this report as an upper-bound estimate for the all-Gemini configuration.

---

## 1. Parameters & Assumptions

- **Dataset**: `sample_dataset.json` (53 test cases)
- **RAG Configurations**: 
  - Config A (Vector-Only)
  - Config B (Hybrid)
  - Config C (Hybrid + Reranker)
- **Model**: `gemini-3.5-flash` (Default model for Generation, Reranking, and LLM-as-a-judge evaluation)
- **Gemini 3.5 Flash Pricing (Standard Tier)**:
  - **Input Tokens**: $1.50 per 1M tokens
  - **Output Tokens**: $9.00 per 1M tokens
- **Text Embedding 004 Pricing**: Free tier (negligible cost)

---

## 2. Phase-by-Phase Token Consumption

### Phase 1: Answer Generation (All 3 Configs)
For each case, we format the question and 3 retrieved context passages (averaging ~125 tokens per passage, 375 tokens total) into the prompt template.
- **Average Prompt Size**: 475 input tokens
- **Average Completion Size**: 100 output tokens
- **Total per case**: 475 input / 100 output tokens
- **Per Run (53 cases)**: 25,175 input / 5,300 output tokens
- **Total for 3 Runs**: **75,525 input / 15,900 output tokens**

---

### Phase 2: Context Reranking (Config C Only)
Config C utilizes Gemini to rerank 10 candidate context chunks down to the top 3.
- **Average Prompt Size** (10 chunks + question + ranking instructions): 1,500 input tokens
- **Average Completion Size** (JSON array of indices + explanations): 150 output tokens
- **Total for Config C (53 cases)**: **79,500 input / 7,950 output tokens**

---

### Phase 3: LLM-as-a-Judge Evaluation (All 3 Configs)
Each case is evaluated across **7 dimensions** (Correctness, Relevance, Faithfulness, Coherence, Safety, Context Recall, Context Precision). Each metric runs a separate parallel LLM prompt.
- **Average Prompt Size per Metric**: 600 input tokens
- **Average Completion Size per Metric**: 150 output tokens (JSON structure)
- **Per Case (7 metrics)**: 4,200 input / 1,050 output tokens
- **Per Run (53 cases)**: 222,600 input / 55,650 output tokens
- **Total for 3 Runs**: **667,800 input / 166,950 output tokens**

---

## 3. Total Aggregated Usage & Cost Projection

| Phase | Input Tokens | Output Tokens | Input Cost (at $1.50/M) | Output Cost (at $9.00/M) | Subtotal |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Answer Generation** | 75,525 | 15,900 | $0.113 | $0.143 | $0.256 |
| **Context Reranking** | 79,500 | 7,950 | $0.119 | $0.072 | $0.191 |
| **Judge Evaluation** | 667,800 | 166,950 | $1.002 | $1.503 | $2.505 |
| **TOTAL** | **822,825** | **190,800** | **$1.234** | **$1.718** | **$2.952** |

### Projected Total Cost: **$2.95 USD**

---

## 4. Key Recommendations for Cost Optimization

1. **Enable Context Caching**: Since the same evaluation prompts (e.g., scoring instructions, rubrics, and same dataset context) are repeated across cases, caching prompt structures can reduce input token costs by up to 50%.
2. **Selective Evaluation**: Run full 7-dimension evaluations during release gates, but use a subset of core metrics (e.g., Correctness & Faithfulness) during development cycles.
3. **Utilize Flash-Lite**: For simple evaluation tasks (such as Coherence or Safety checks), transition the judge model from `gemini-3.5-flash` to `gemini-3.1-flash-lite` to lower costs further.
