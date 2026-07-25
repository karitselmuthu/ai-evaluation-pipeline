import json
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

class DatasetLoader:
    """
    Handles loading, parsing, and validating evaluation datasets.
    Supported formats: JSON, JSONL, CSV.
    """

    REQUIRED_FIELDS = ["question"]

    @staticmethod
    def load(file_path: str) -> List[Dict[str, Any]]:
        """
        Loads dataset from file and validates its structure.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".json":
            return DatasetLoader._load_json(path)
        elif ext == ".jsonl":
            return DatasetLoader._load_jsonl(path)
        elif ext in [".csv", ".xlsx"]:
            return DatasetLoader._load_tabular(path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Use JSON, JSONL, or CSV.")

    @staticmethod
    def _load_json(path: Path) -> List[Dict[str, Any]]:
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON file: {e}")

        # Support both wrapped object with 'eval_cases' key or a direct array
        if isinstance(data, dict):
            if "eval_cases" in data:
                cases = data["eval_cases"]
            elif "cases" in data:
                cases = data["cases"]
            else:
                raise ValueError("JSON object must contain 'eval_cases' or 'cases' key representing the list of evaluation cases.")
        elif isinstance(data, list):
            cases = data
        else:
            raise ValueError("JSON file must contain either an array of cases or an object with 'eval_cases' key.")

        return DatasetLoader.validate_and_normalize(cases)

    @staticmethod
    def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
        cases = []
        with open(path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {idx} of JSONL: {e}")
        return DatasetLoader.validate_and_normalize(cases)

    @staticmethod
    def _load_tabular(path: Path) -> List[Dict[str, Any]]:
        try:
            df = pd.read_csv(path) if path.suffix.lower() == ".csv" else pd.read_excel(path)
        except Exception as e:
            raise ValueError(f"Error reading spreadsheet file: {e}")
        
        cases = df.to_dict(orient="records")
        return DatasetLoader.validate_and_normalize(cases)

    @staticmethod
    def validate_and_normalize(cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validates structure and normalizes schemas.
        Each case is normalized to contain:
          - question: str
          - contexts: List[str]
          - ground_truth: str
        """
        normalized_cases = []
        for idx, case in enumerate(cases):
            # Normalize prompt -> question mapping if present
            if "prompt" in case and "question" not in case:
                # Handle prompt as dict or string
                p = case["prompt"]
                if isinstance(p, dict) and "parts" in p:
                    # Ragas schema pattern
                    parts = p["parts"]
                    if isinstance(parts, list) and len(parts) > 0 and "text" in parts[0]:
                        case["question"] = parts[0]["text"]
                elif isinstance(p, str):
                    case["question"] = p

            # Check required fields
            for field in DatasetLoader.REQUIRED_FIELDS:
                if field not in case:
                    raise ValueError(f"Missing required field '{field}' in case index {idx}: {case}")

            # Normalize fields
            question = str(case["question"]).strip()
            
            # Contexts normalization
            contexts = case.get("contexts", [])
            if isinstance(contexts, str):
                # Try to parse stringified list
                if contexts.startswith("[") and contexts.endswith("]"):
                    try:
                        contexts = json.loads(contexts.replace("'", '"'))
                    except Exception:
                        contexts = [contexts]
                else:
                    contexts = [contexts]
            elif contexts is None:
                contexts = []
            elif not isinstance(contexts, list):
                contexts = [str(contexts)]
            
            contexts = [str(c).strip() for c in contexts]

            # Ground truth normalization
            ground_truth = case.get("ground_truth", case.get("reference", ""))
            if ground_truth is None:
                ground_truth = ""
            ground_truth = str(ground_truth).strip()

            # Pass along any extra fields (e.g. metadata)
            normalized_case = {
                "question": question,
                "contexts": contexts,
                "ground_truth": ground_truth
            }
            # Add metadata or identifiers
            if "eval_case_id" in case:
                normalized_case["eval_case_id"] = case["eval_case_id"]
            else:
                normalized_case["eval_case_id"] = f"case_{idx}"
                
            # Keep additional fields
            for k, v in case.items():
                if k not in normalized_case and k not in ["prompt", "reference"]:
                    normalized_case[k] = v

            normalized_cases.append(normalized_case)

        return normalized_cases
