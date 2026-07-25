import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    Handles logging runs, metrics, parameters, and artifacts to experiment tracking frameworks
    like MLflow or Weights & Biases (W&B).
    """

    def __init__(self, use_mlflow: bool = True, use_wandb: bool = False):
        self.use_mlflow = use_mlflow
        self.use_wandb = use_wandb
        self._mlflow_initialized = False
        self._wandb_initialized = False

        if self.use_mlflow:
            self._init_mlflow()
        if self.use_wandb:
            self._init_wandb()

    def _init_mlflow(self):
        try:
            import mlflow
            # Check for env variables or default tracking URI
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
            mlflow.set_tracking_uri(tracking_uri)
            
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "AI-Evaluation-Pipeline")
            mlflow.set_experiment(experiment_name)
            
            self._mlflow_initialized = True
            logger.info(f"MLflow tracking initialized at {tracking_uri} for experiment '{experiment_name}'")
        except ImportError:
            logger.warning("MLflow library not installed. Skipping MLflow tracking initialization.")
        except Exception as e:
            logger.warning(f"Failed to initialize MLflow tracking: {e}. Runs will not be logged to MLflow.")

    def _init_wandb(self):
        try:
            import wandb
            # Check for API key
            if os.getenv("WANDB_API_KEY"):
                wandb.login(key=os.getenv("WANDB_API_KEY"))
                self._wandb_initialized = True
                logger.info("Weights & Biases logging initialized.")
            else:
                logger.warning("WANDB_API_KEY not found. Skipping W&B tracking initialization.")
        except ImportError:
            logger.warning("wandb library not installed. Skipping W&B tracking initialization.")
        except Exception as e:
            logger.warning(f"Failed to initialize W&B: {e}")

    def log_run(
        self, 
        params: Dict[str, Any], 
        metrics: Dict[str, Any], 
        report_data: Dict[str, Any],
        run_name: Optional[str] = None
    ):
        """
        Logs a single evaluation run (parameters, metrics, and output report JSON artifact).
        """
        # Log to MLflow
        if self._mlflow_initialized:
            try:
                import mlflow
                with mlflow.start_run(run_name=run_name) as run:
                    # Log parameters
                    mlflow.log_params(params)
                    
                    # Log metrics (flattened dict)
                    mlflow.log_metrics(metrics)
                    
                    # Log final results as a JSON artifact
                    temp_file = f"eval_run_{run.info.run_id}.json"
                    with open(temp_file, "w", encoding="utf-8") as f:
                        json.dump(report_data, f, indent=4)
                    
                    mlflow.log_artifact(temp_file, "reports")
                    os.remove(temp_file)
                    
                    logger.info(f"Run logged successfully to MLflow: Run ID {run.info.run_id}")
            except Exception as e:
                logger.error(f"Error logging to MLflow: {e}")

        # Log to W&B
        if self._wandb_initialized:
            try:
                import wandb
                # Initialize run
                run = wandb.init(
                    project=os.getenv("WANDB_PROJECT", "ai-evaluation-pipeline"),
                    name=run_name,
                    config=params,
                    reinit=True
                )
                
                # Log metrics
                wandb.log(metrics)
                
                # Log report data artifact
                artifact = wandb.Artifact(
                    name=run_name or "eval-report",
                    type="evaluation"
                )
                
                temp_file = "eval_results.json"
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(report_data, f, indent=4)
                
                artifact.add_file(temp_file)
                run.log_artifact(artifact)
                run.finish()
                
                os.remove(temp_file)
                logger.info("Run logged successfully to W&B.")
            except Exception as e:
                logger.error(f"Error logging to W&B: {e}")
