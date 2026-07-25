import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
from main import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("RAG-Config-Comparer")

def run_comparison(
    dataset_path: str,
    docs_dir: str,
    model_name: str = None,
    judge_model: str = None,
    output_dir: str = "output"
):
    """
    Runs the evaluation pipeline across all three RAG configurations and compares results.
    """
    configs = {
        "Config A: Vector-Only": "vector",
        "Config B: Hybrid (BM25 + Vector)": "hybrid",
        "Config C: Hybrid + Reranker": "hybrid_rerank"
    }

    results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(output_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    for config_name, config_key in configs.items():
        logger.info("\n" + "="*80)
        logger.info(f"RUNNING EVALUATION FOR: {config_name}")
        logger.info("="*80)

        # Run pipeline
        try:
            report_file = run_pipeline(
                dataset_path=dataset_path,
                model_name=model_name,
                judge_model=judge_model,
                output_dir=output_dir,
                retrieval_config=config_key,
                docs_dir=docs_dir,
                track=False
            )
            
            # Load run summary
            with open(report_file, "r") as f:
                run_data = json.load(f)
                results[config_name] = run_data["summary"]
                # Save retrieval config type in results for dashboard reference
                results[config_name]["retrieval_config"] = config_key
                results[config_name]["report_file"] = report_file
        except Exception as e:
            logger.error(f"Failed to run evaluation for {config_name}: {e}", exc_info=True)

    if not results:
        logger.error("No configurations successfully ran.")
        return

    # Create comparison table
    df = pd.DataFrame(results).T
    
    # Selected columns to show
    show_cols = [
        "overall_quality_score",
        "correctness_score",
        "relevance_score",
        "faithfulness_score",
        "context_recall_score",
        "context_precision_score",
        "avg_latency",
        "total_cost",
        "total_cases"
    ]
    
    # Filter available columns
    available_cols = [col for col in show_cols if col in df.columns]
    comparison_df = df[available_cols]

    # Generate Markdown comparison
    md_report = f"""# RAG Retrieval Configurations Comparison Report
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*

This report compares the performance of three different Retrieval-Augmented Generation (RAG) configurations evaluated against the same test dataset.

## Core Comparison Table
{comparison_df.to_markdown()}

## Key Takeaways
1. **Retrieval Precision**: Notice how Reranking (**Config C**) affects *Context Precision* and *Faithfulness* compared to the other configs. Rerankers filter out distracting chunks, increasing the signal-to-noise ratio.
2. **Context Recall**: Compare how Hybrid retrieval (**Config B**) compares to Vector-only (**Config A**) on *Context Recall*. Adding keyword matching (BM25) helps catch specific entities and IDs.
3. **Latency Tradeoffs**: Adding a reranker (**Config C**) typically adds a latency penalty since it requires an extra model call. Review the **Average Latency** to evaluate if the quality improvements are worth the performance cost.

## Detailed Metrics Breakdown

"""
    for config_name, summary in results.items():
        md_report += f"""
### {config_name}
- **Overall Quality Score**: {summary.get('overall_quality_score', 0.0):.4f}
- **Correctness**: {summary.get('correctness_score', 0.0):.4f}
- **Relevance**: {summary.get('relevance_score', 0.0):.4f}
- **Faithfulness**: {summary.get('faithfulness_score', 0.0):.4f}
- **Context Recall**: {summary.get('context_recall_score', 0.0):.4f}
- **Context Precision**: {summary.get('context_precision_score', 0.0):.4f}
- **Average Latency**: {summary.get('avg_latency', 0.0):.2f}s
- **Total Cost**: ${summary.get('total_cost', 0.0):.4f}
- **Reference Report**: [{Path(summary.get('report_file')).name}]({summary.get('report_file')})

---
"""

    # Save reports
    comparison_md_file = out_dir / f"rag_comparison_{timestamp}.md"
    comparison_json_file = out_dir / f"rag_comparison_{timestamp}.json"
    latest_md_file = out_dir / "rag_comparison_latest.md"
    latest_json_file = out_dir / "rag_comparison_latest.json"

    with open(comparison_md_file, "w", encoding="utf-8") as f:
        f.write(md_report)
    with open(latest_md_file, "w", encoding="utf-8") as f:
        f.write(md_report)

    # Save structured json comparison
    comparison_structure = {
        "timestamp": timestamp,
        "model_under_test": model_name,
        "judge_model": judge_model,
        "dataset_path": dataset_path,
        "docs_dir": docs_dir,
        "comparison_table": comparison_df.to_dict(orient="index"),
        "raw_results": results
    }
    
    with open(comparison_json_file, "w", encoding="utf-8") as f:
        json.dump(comparison_structure, f, indent=4)
    with open(latest_json_file, "w", encoding="utf-8") as f:
        json.dump(comparison_structure, f, indent=4)

    logger.info("\n" + "="*80)
    logger.info("RAG COMPARISON COMPLETE")
    logger.info(f"Comparison Markdown Report saved to: {comparison_md_file}")
    logger.info(f"Comparison JSON data saved to: {comparison_json_file}")
    logger.info("="*80)
    
    print("\n" + "="*80)
    print("RAG CONFIGURATIONS COMPARISON SUMMARY")
    print("="*80)
    print(comparison_df.to_string())
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare Config A, B, and C RAG retrieval strategies in the Evaluation Pipeline.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the evaluation dataset (JSON, JSONL, CSV)")
    parser.add_argument("--docs-dir", type=str, required=True, help="Directory containing documents for RAG contexts")
    parser.add_argument("--model", type=str, default=None, help="Proxy model to evaluate (defaults to EVAL_GENERATOR_MODEL)")
    parser.add_argument("--judge-model", type=str, default=None, help="Pinned judge model (defaults to EVAL_JUDGE_MODEL)")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save report outputs")
    
    args = parser.parse_args()

    run_comparison(
        dataset_path=args.dataset,
        docs_dir=args.docs_dir,
        model_name=args.model,
        judge_model=args.judge_model,
        output_dir=args.output_dir
    )
