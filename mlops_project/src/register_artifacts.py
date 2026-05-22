import logging
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import dagshub # Make sure to import this if you use dagshub.init!

logger = logging.getLogger("src.register_artifacts")

# 1. CONNECT TO DAGSHUB BEFORE CREATING THE CLIENT
# (If you don't use dagshub.init, uncomment the set_tracking_uri line below instead)
dagshub.init(repo_owner='RodrigoSchneiderbr', repo_name='MLOPS', mlflow=True)
# mlflow.set_tracking_uri("https://dagshub.com/RodrigoSchneiderbr/MLOPS.mlflow")

# 2. NOW create the client so it connects to the remote server
client = MlflowClient()

def get_best_run(experiment_id: str, parent_run_id: str) -> pd.Series:
    """Get the best child run based on test accuracy for a given parent run."""
    child_runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
        order_by=["metrics.test_accuracy DESC"],
        max_results=1000
    )       
    return child_runs[0]

def register_model() -> None:
    """Register the model that was logged during training."""
    logger.info("Registering model from latest MLflow run")

    # 3. SAFER WAY TO GET EXPERIMENT
    experiment_name = "ml_classification"
    experiment = client.get_experiment_by_name(experiment_name)
    
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' not found on DagsHub! Did you run train_model.py first?")
        
    experiment_id = experiment.experiment_id

    # Get the latest run from the experiment
    latest_run = client.search_runs(
        experiment_ids=[experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )[0]
    
    # Check if the latest run has a parent run
    run_id = latest_run.info.run_id
    parent_run_id = latest_run.data.tags.get('mlflow.parentRunId')
    
    if parent_run_id:
        logger.info(f"Latest run has parent run ID: {parent_run_id}")
        best_run = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=f"tags.mlflow.parentRunId = '{parent_run_id}'",
            order_by=["metrics.test_accuracy DESC"],
            max_results=1
        )[0]
        run_id = best_run.info.run_id
        logger.info(f"Using best run {run_id} with test_accuracy: {best_run.data.metrics['test_accuracy']}")

    # Register the model from the run
    logger.info("Registering model")
    try:
        client.create_registered_model("model")
    except mlflow.exceptions.MlflowException:
        logger.debug("Model already exists")

    model_uri = f"runs:/{run_id}/model"
    client.create_model_version(
        name="model",
        source=model_uri,
        run_id=run_id
    )
    logger.info("Registered model successfully")

def main() -> None:
    """Main function to orchestrate the model registration process."""
    register_model()
    logger.info("Model registration completed")

if __name__ == "__main__":
    main()