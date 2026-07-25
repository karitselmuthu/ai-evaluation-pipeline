import pytest
import os
import json
from src.core.dataset import DatasetLoader

def test_validate_and_normalize_valid():
    cases = [
        {
            "question": "What is 2+2?",
            "contexts": "Simple math rules.",
            "ground_truth": "4"
        },
        {
            "prompt": "Hello",
            "reference": "Hi there"
        }
    ]
    
    normalized = DatasetLoader.validate_and_normalize(cases)
    
    assert len(normalized) == 2
    assert normalized[0]["question"] == "What is 2+2?"
    assert normalized[0]["contexts"] == ["Simple math rules."]
    assert normalized[0]["ground_truth"] == "4"
    
    assert normalized[1]["question"] == "Hello"
    assert normalized[1]["contexts"] == []
    assert normalized[1]["ground_truth"] == "Hi there"
    assert normalized[1]["eval_case_id"] == "case_1"

def test_validate_and_normalize_missing_question():
    cases = [
        {
            "contexts": ["Context without prompt or question"]
        }
    ]
    with pytest.raises(ValueError) as excinfo:
        DatasetLoader.validate_and_normalize(cases)
    assert "Missing required field 'question'" in str(excinfo.value)

def test_load_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        DatasetLoader.load("non_existent_file.json")
