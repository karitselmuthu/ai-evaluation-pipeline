import os
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.core.dataset import DatasetLoader
from src.core.generator import LLMGenerator
from src.core.evaluator import EvaluationEngine
from src.core.aggregator import ScoreAggregator
from src.core.tracker import ExperimentTracker
from src.core.retriever import RAGRetriever
from src.core.secret_bootstrap import bootstrap_secrets
from src.core.llm_client import summarize_providers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
# Silence the Antigravity/Gemini harness 429 spend-cap storm (set
# VERBOSE_HARNESS_LOGS=1 to see it). Must run after basicConfig.
from src.core.log_filters import install_noise_filter
install_noise_filter()

logger = logging.getLogger("AI-Evaluation-Pipeline")

def run_pipeline(
    dataset_path: str,
    model_name: str = None,
    judge_model: str = None,
    prompt_template_str: str = None,
    output_dir: str = "output",
    track: bool = False,
    retrieval_config: str = None,
    docs_dir: str = None
) -> str:
    """
    Runs the full end-to-end evaluation pipeline with optional dynamic retrieval.
    """
    # Resolve LLM keys via the SecretProvider seam (vault or env) and export them
    # to the environment before any generator / judge / retriever call runs.
    bootstrap_secrets()

    # Resolve model names against the proxy defaults (a None/omitted model must
    # not fall back to a capped provider). model "default" enables rotation.
    model_name = model_name or os.getenv("EVAL_GENERATOR_MODEL", "default")
    judge_model = judge_model or os.getenv("EVAL_JUDGE_MODEL", "llama-3.3-70b-versatile")

    logger.info(f"Starting evaluation pipeline...")
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Model under test: {model_name}")
    logger.info(f"Judge model: {judge_model}")
    if docs_dir:
        logger.info(f"Dynamic Retrieval Enabled: Config={retrieval_config}, Docs={docs_dir}")

    # 1. Load Dataset
    cases = DatasetLoader.load(dataset_path)
    logger.info(f"Successfully loaded {len(cases)} evaluation cases.")

    # 1b. Optional Retrieval Step
    if docs_dir:
        logger.info("Initializing Retriever and performing context search...")
        retriever = RAGRetriever(docs_dir=docs_dir)
        cfg = retrieval_config or "hybrid"
        for case in cases:
            retrieved_contexts = retriever.retrieve(case["question"], config_type=cfg, top_n=3)
            case["contexts"] = retrieved_contexts
        logger.info("Retrieval complete for all cases.")

    # 2. Generate Answers
    logger.info("Generating responses from application model...")
    generator = LLMGenerator(model_name=model_name)
    generated_cases = generator.generate_batch(cases, prompt_template=prompt_template_str)
    logger.info("Generation complete.")

    # 3. Evaluate Answers
    logger.info("Evaluating responses across dimensions (Correctness, Relevance, Faithfulness, Coherence, Safety)...")
    evaluator = EvaluationEngine(judge_model=judge_model)
    evaluated_cases = evaluator.evaluate_batch(generated_cases)
    logger.info("Evaluation complete.")

    # 4. Aggregate Scores
    logger.info("Aggregating results...")
    result = ScoreAggregator.aggregate(evaluated_cases)
    summary = result["summary"]

    # 4b. Provider attribution — surface mixed-provider runs instead of hiding
    # them inside a single average. Covers both generation and judging.
    attributions = []
    for c in evaluated_cases:
        attributions.append((c.get("gen_serving_provider", "unknown"), c.get("gen_fell_back", False)))
        attributions.append((c.get("judge_serving_provider", "unknown"), c.get("judge_fell_back", False)))
    result["providers"] = summarize_providers(attributions)
    if result["providers"].get("mixed_provider_run"):
        logger.warning("MIXED-PROVIDER RUN: %s", result["providers"]["by_serving_provider"])

    # 5. Generate Reports
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    report_name = f"eval_report_{timestamp}"
    md_report = ScoreAggregator.generate_markdown_report(result, title=f"AI Evaluation Report - {model_name} ({timestamp})")
    
    # Save files
    json_file = out_path / f"{report_name}.json"
    md_file = out_path / f"{report_name}.md"
    
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    logger.info(f"Aggregated JSON report saved to: {json_file}")
    logger.info(f"Markdown report saved to: {md_file}")

    # 6. Log to Tracker
    if track:
        logger.info("Logging run details to Experiment Tracker...")
        tracker = ExperimentTracker(use_mlflow=True, use_wandb=False)
        params = {
            "model_under_test": model_name,
            "judge_model": judge_model,
            "prompt_template": prompt_template_str or "DEFAULT_PROMPT_TEMPLATE",
            "dataset_path": dataset_path,
            "timestamp": timestamp,
            "retrieval_config": retrieval_config or "N/A",
            "docs_dir": docs_dir or "N/A"
        }
        
        # Extract flat metrics for logging
        metrics = {
            "overall_quality_score": summary.get("overall_quality_score", 0.0),
            "correctness": summary.get("correctness_score", 0.0),
            "relevance": summary.get("relevance_score", 0.0),
            "faithfulness": summary.get("faithfulness_score", 0.0),
            "coherence": summary.get("coherence_score", 0.0),
            "safety": summary.get("safety_score", 0.0),
            "avg_latency": summary.get("avg_latency", 0.0),
            "total_cost": summary.get("total_cost", 0.0)
        }
        if "context_recall_score" in summary:
            metrics["context_recall"] = summary.get("context_recall_score", 0.0)
        if "context_precision_score" in summary:
            metrics["context_precision"] = summary.get("context_precision_score", 0.0)
        
        tracker.log_run(
            params=params,
            metrics=metrics,
            report_data=result,
            run_name=f"eval-{model_name}-{timestamp}"
        )

    print("\n" + "="*50)
    print(f"EVALUATION COMPLETE - OVERALL QUALITY: {summary.get('overall_quality_score', 0.0):.2f}/1.00")
    print(f"Total cases: {summary.get('total_cases', 0)} | Total Cost: ${summary.get('total_cost', 0.0):.4f} | Avg Latency: {summary.get('avg_latency', 0.0):.2f}s")
    print(f"Served by: {result['providers'].get('by_serving_provider', {})}" + (" | ⚠ MIXED-PROVIDER RUN" if result['providers'].get('mixed_provider_run') else ""))
    print("="*50)
    
    return str(json_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Evaluation Pipeline CLI Runner")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset (JSON, JSONL, CSV)")
    parser.add_argument("--model", type=str, default=None, help="Proxy model to evaluate ('default' rotates; e.g. llama-3.3-70b-versatile). Defaults to EVAL_GENERATOR_MODEL.")
    parser.add_argument("--judge-model", type=str, default=None, help="Pinned LLM-as-a-judge model (must be explicit, e.g. llama-3.3-70b-versatile). Defaults to EVAL_JUDGE_MODEL.")
    parser.add_argument("--prompt-template", type=str, default=None, help="Prompt template text or path to prompt file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save report outputs")
    parser.add_argument("--track", action="store_true", help="Enable MLflow tracking logs")
    parser.add_argument("--retrieval-config", type=str, default=None, choices=["vector", "hybrid", "hybrid_rerank"], help="RAG retrieval strategy to use")
    parser.add_argument("--docs-dir", type=str, default=None, help="Directory containing documents for RAG context retrieval")
    
    args = parser.parse_args()

    # Load prompt template from file if path is passed
    prompt_str = args.prompt_template
    if prompt_str and os.path.exists(prompt_str):
        with open(prompt_str, "r", encoding="utf-8") as f:
            prompt_str = f.read()

    run_pipeline(
        dataset_path=args.dataset,
        model_name=args.model,
        judge_model=args.judge_model,
        prompt_template_str=prompt_str,
        output_dir=args.output_dir,
        track=args.track,
        retrieval_config=args.retrieval_config,
        docs_dir=args.docs_dir
    )
