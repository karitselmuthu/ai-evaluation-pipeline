import pytest
from src.core.aggregator import ScoreAggregator

def test_score_aggregator_empty():
    res = ScoreAggregator.aggregate([])
    assert res["summary"] == {}
    assert res["cases"] == []

def test_score_aggregator_metrics():
    mock_cases = [
        {
            "eval_case_id": "case_1",
            "question": "Q1",
            "answer": "A1",
            "correctness_score": 1.0,
            "relevance_score": 1.0,
            "faithfulness_score": 0.8,
            "coherence_score": 0.9,
            "safety_score": 1.0,
            "latency": 2.0,
            "cost": 0.005,
            "input_tokens": 100,
            "output_tokens": 50
        },
        {
            "eval_case_id": "case_2",
            "question": "Q2",
            "answer": "A2",
            "correctness_score": 0.6,
            "relevance_score": 0.8,
            "faithfulness_score": 0.6,
            "coherence_score": 0.7,
            "safety_score": 1.0,
            "latency": 3.0,
            "cost": 0.007,
            "input_tokens": 120,
            "output_tokens": 60
        }
    ]
    
    result = ScoreAggregator.aggregate(mock_cases)
    summary = result["summary"]
    
    assert summary["total_cases"] == 2
    assert summary["correctness_score"] == 0.8  # (1.0 + 0.6) / 2
    assert summary["relevance_score"] == 0.9    # (1.0 + 0.8) / 2
    assert summary["faithfulness_score"] == 0.7  # (0.8 + 0.6) / 2
    assert summary["avg_latency"] == 2.5        # (2.0 + 3.0) / 2
    assert summary["total_cost"] == 0.012       # 0.005 + 0.007
    assert summary["total_input_tokens"] == 220
    assert summary["total_output_tokens"] == 110
    assert summary["overall_quality_score"] == pytest.approx(0.84)  # mean of [0.8, 0.9, 0.7, 0.8, 1.0]

def test_markdown_report_generation():
    mock_result = {
        "summary": {
            "overall_quality_score": 0.85,
            "total_cases": 1,
            "total_cost": 0.004,
            "avg_latency": 1.5,
            "p95_latency": 1.5,
            "correctness_score": 0.9,
            "correctness_score_std": 0.0,
            "relevance_score": 0.9,
            "relevance_score_std": 0.0,
            "faithfulness_score": 0.8,
            "faithfulness_score_std": 0.0,
            "coherence_score": 0.8,
            "coherence_score_std": 0.0,
            "safety_score": 1.0,
            "safety_score_std": 0.0
        },
        "cases": [
            {
                "eval_case_id": "test_case",
                "question": "test question",
                "answer": "test answer",
                "correctness_score": 0.9,
                "correctness_explanation": "Good correctness",
                "relevance_score": 0.9,
                "relevance_explanation": "Good relevance",
                "faithfulness_score": 0.8,
                "faithfulness_explanation": "Grounded",
                "coherence_score": 0.8,
                "coherence_explanation": "Clear",
                "safety_score": 1.0,
                "safety_explanation": "Safe",
                "latency": 1.5,
                "cost": 0.004
            }
        ]
    }
    
    report = ScoreAggregator.generate_markdown_report(mock_result)
    assert "# AI Evaluation Report" in report
    assert "Overall Quality Score" in report
    assert "Executive Summary" in report
    assert "test question" in report
