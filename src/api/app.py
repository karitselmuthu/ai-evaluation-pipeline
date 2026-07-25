import os
import json
import logging
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from datetime import datetime

app = FastAPI(
    title="AI Evaluation Pipeline Backend",
    description="REST API for triggering evaluation runs, serving reports, and prompt iteration.",
    version="1.0.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
OUTPUT_DIR = Path("output")
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
UPLOAD_DIR.mkdir(exist_ok=True, parents=True)

logger = logging.getLogger("api")

class EvalRunRequest(BaseModel):
    dataset_path: str
    model_name: Optional[str] = None  # None -> EVAL_GENERATOR_MODEL (proxy 'default' rotates)
    judge_model: Optional[str] = None  # None -> EVAL_JUDGE_MODEL (pinned)
    prompt_template: Optional[str] = None
    track: Optional[bool] = False

# Import pipeline runner
from main import run_pipeline

def run_evaluation_task(
    dataset_path: str,
    model_name: str,
    judge_model: str,
    prompt_template: Optional[str],
    track: bool
):
    try:
        run_pipeline(
            dataset_path=dataset_path,
            model_name=model_name,
            judge_model=judge_model,
            prompt_template_str=prompt_template,
            output_dir=str(OUTPUT_DIR),
            track=track
        )
    except Exception as e:
        logger.error(f"Async evaluation task failed: {e}")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Evaluation Pipeline API. Visit /docs for OpenAPI specs."}

@app.post("/eval/run/sync")
def run_evaluation_sync(request: EvalRunRequest):
    """
    Triggers an evaluation run synchronously and returns the output JSON filename.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    
    try:
        report_path = run_pipeline(
            dataset_path=request.dataset_path,
            model_name=request.model_name,
            judge_model=request.judge_model,
            prompt_template_str=request.prompt_template,
            output_dir=str(OUTPUT_DIR),
            track=request.track
        )
        return {"status": "success", "report_file": report_path}
    except Exception as e:
        logger.error(f"Sync evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/eval/run/async")
def run_evaluation_async(request: EvalRunRequest, background_tasks: BackgroundTasks):
    """
    Triggers an evaluation run asynchronously in the background.
    """
    if not os.path.exists(request.dataset_path):
        raise HTTPException(status_code=404, detail="Dataset file not found.")
    
    background_tasks.add_task(
        run_evaluation_task,
        dataset_path=request.dataset_path,
        model_name=request.model_name,
        judge_model=request.judge_model,
        prompt_template=request.prompt_template,
        track=request.track
    )
    return {"status": "started", "message": "Evaluation job dispatched to background."}

@app.post("/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Saves an uploaded evaluation dataset file (JSON or CSV) to the server.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = UPLOAD_DIR / f"{timestamp}_{file.filename}"
    
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        return {"status": "success", "file_path": str(file_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.get("/reports")
def list_reports():
    """
    Lists all evaluation run reports in the output directory.
    """
    reports = []
    for file in OUTPUT_DIR.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
                summary = data.get("summary", {})
                reports.append({
                    "filename": file.name,
                    "filepath": str(file),
                    "created_at": datetime.fromtimestamp(file.stat().st_mtime).isoformat(),
                    "overall_quality_score": summary.get("overall_quality_score", 0.0),
                    "total_cases": summary.get("total_cases", 0),
                    "total_cost": summary.get("total_cost", 0.0),
                    "avg_latency": summary.get("avg_latency", 0.0)
                })
        except Exception as e:
            logger.warning(f"Could not parse report file {file}: {e}")
            
    # Sort by created time descending
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return reports

@app.get("/reports/{filename}")
def get_report(filename: str):
    """
    Retrieves full metrics details for a specific evaluation run report.
    """
    filepath = OUTPUT_DIR / filename
    if not filepath.exists() or filepath.suffix != ".json":
        raise HTTPException(status_code=404, detail="Report file not found.")
        
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
