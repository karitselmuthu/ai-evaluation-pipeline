import pandas as pd
import numpy as np
from typing import List, Dict, Any

class ScoreAggregator:
    """
    Normalizes, aggregates, and summarizes evaluation results using Pandas/NumPy.
    Generates structured summaries and markdown reports.
    """

    @staticmethod
    def aggregate(cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates results from a list of evaluated cases.
        """
        if not cases:
            return {
                "summary": {},
                "cases": []
            }

        df = pd.DataFrame(cases)

        # Core metric columns
        score_cols = [
            "correctness_score", 
            "relevance_score", 
            "faithfulness_score", 
            "coherence_score", 
            "safety_score"
        ]
        # Include context-specific metrics if evaluated
        for extra_col in ["context_recall_score", "context_precision_score"]:
            if extra_col in df.columns:
                score_cols.append(extra_col)

        # Calculate averages for score columns
        summary = {}
        for col in score_cols:
            if col in df.columns:
                summary[col] = float(df[col].mean())
                summary[f"{col}_std"] = float(df[col].std()) if len(df) > 1 else 0.0
            else:
                summary[col] = 0.0
                summary[f"{col}_std"] = 0.0

        # Performance metrics
        if "latency" in df.columns:
            summary["avg_latency"] = float(df["latency"].mean())
            summary["p95_latency"] = float(df["latency"].quantile(0.95))
            summary["max_latency"] = float(df["latency"].max())
        else:
            summary["avg_latency"] = 0.0
            summary["p95_latency"] = 0.0
            summary["max_latency"] = 0.0

        if "cost" in df.columns:
            summary["total_cost"] = float(df["cost"].sum())
            summary["avg_cost"] = float(df["cost"].mean())
        else:
            summary["total_cost"] = 0.0
            summary["avg_cost"] = 0.0

        if "input_tokens" in df.columns:
            summary["total_input_tokens"] = int(df["input_tokens"].sum())
        if "output_tokens" in df.columns:
            summary["total_output_tokens"] = int(df["output_tokens"].sum())

        summary["total_cases"] = len(df)

        # Overall quality index (simple average of all dimensions)
        dimension_scores = [summary[col] for col in score_cols]
        summary["overall_quality_score"] = float(np.mean(dimension_scores))

        return {
            "summary": summary,
            "cases": cases
        }

    @staticmethod
    def generate_markdown_report(result: Dict[str, Any], title: str = "AI Evaluation Report") -> str:
        """
        Generates a beautiful markdown text report summarizing the evaluation run.
        """
        summary = result["summary"]
        cases = result["cases"]

        report = f"""# {title}

## Executive Summary
| Metric | Value |
| :--- | :--- |
| **Overall Quality Score** | {summary.get('overall_quality_score', 0.0):.2f} / 1.00 |
| **Total Test Cases** | {summary.get('total_cases', 0)} |
| **Total API Cost** | ${summary.get('total_cost', 0.0):.4f} |
| **Average Latency** | {summary.get('avg_latency', 0.0):.2f}s |
| **95th Percentile Latency** | {summary.get('p95_latency', 0.0):.2f}s |

## Evaluation Dimensions Breakdown
| Dimension | Average Score | Std Dev | Status |
| :--- | :---: | :---: | :---: |
| **Correctness** | {summary.get('correctness_score', 0.0):.2f} | {summary.get('correctness_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('correctness_score', 0.0) >= 0.8 else '❌ Fail'} |
| **Relevance** | {summary.get('relevance_score', 0.0):.2f} | {summary.get('relevance_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('relevance_score', 0.0) >= 0.8 else '❌ Fail'} |
| **Faithfulness** | {summary.get('faithfulness_score', 0.0):.2f} | {summary.get('faithfulness_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('faithfulness_score', 0.0) >= 0.8 else '❌ Fail'} |
| **Coherence** | {summary.get('coherence_score', 0.0):.2f} | {summary.get('coherence_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('coherence_score', 0.0) >= 0.8 else '❌ Fail'} |
| **Safety** | {summary.get('safety_score', 0.0):.2f} | {summary.get('safety_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('safety_score', 0.0) >= 0.95 else '❌ Fail'} |
"""

        if "context_recall_score" in summary:
            report += f"| **Context Recall** | {summary.get('context_recall_score', 0.0):.2f} | {summary.get('context_recall_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('context_recall_score', 0.0) >= 0.80 else '❌ Fail'} |\n"
        if "context_precision_score" in summary:
            report += f"| **Context Precision** | {summary.get('context_precision_score', 0.0):.2f} | {summary.get('context_precision_score_std', 0.0):.2f} | {'✅ Pass' if summary.get('context_precision_score', 0.0) >= 0.80 else '❌ Fail'} |\n"

        report += """
## Individual Case Details
"""
        for case in cases:
            report += f"""
### Case: {case.get('eval_case_id', 'Unknown')}
* **Question**: {case.get('question')}
* **Answer**: {case.get('answer')}
* **Metrics**:
  * Correctness: {case.get('correctness_score', 0.0):.2f} (Reason: {case.get('correctness_explanation', 'N/A')})
  * Relevance: {case.get('relevance_score', 0.0):.2f} (Reason: {case.get('relevance_explanation', 'N/A')})
  * Faithfulness: {case.get('faithfulness_score', 0.0):.2f} (Reason: {case.get('faithfulness_explanation', 'N/A')})
  * Coherence: {case.get('coherence_score', 0.0):.2f} (Reason: {case.get('coherence_explanation', 'N/A')})
  * Safety: {case.get('safety_score', 0.0):.2f} (Reason: {case.get('safety_explanation', 'N/A')})
"""
            if "context_recall_score" in case:
                report += f"  * Context Recall: {case.get('context_recall_score', 0.0):.2f} (Reason: {case.get('context_recall_explanation', 'N/A')})\n"
            if "context_precision_score" in case:
                report += f"  * Context Precision: {case.get('context_precision_score', 0.0):.2f} (Reason: {case.get('context_precision_explanation', 'N/A')})\n"

            report += f"""* **Performance**: Latency: {case.get('latency', 0.0):.2f}s | Cost: ${case.get('cost', 0.0):.4f}

---
"""
        return report
