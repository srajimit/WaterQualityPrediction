import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import time

def training_Model(X: pd.DataFrame, y: pd.Series):
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient(tracking_uri=mlflow_tracking_uri)

    # Set experiment
    mlflow.set_experiment("Water_Quality_Prediction")

    with mlflow.start_run() as run:
        params = {
            "n_estimators": 100,
            "max_depth": 6,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "bootstrap": True,
            "oob_score": False,
            "random_state": 888,
        }

        model = RandomForestClassifier(**params)
        model.fit(xtrain, ytrain)

        # Evaluate
        ypred = model.predict(xtest)
        accuracy = accuracy_score(ytest, ypred)
        print(f"Test Accuracy = {accuracy:.4f}")
        print(classification_report(ytest, ypred))

        # Log params, metrics, and model
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(model, "WQ_RandomForestModel")

        # Register model
        registered_model = mlflow.register_model(
            f"runs:/{run.info.run_id}/WQ_RandomForestModel",
            "WaterQualityModel"
        )

        # Wait for model registration
        elapsed = 0
        while True:
            try:
                latest_version = client.get_latest_versions("WaterQualityModel")[0].version
                print(f"Registered model version: {latest_version}")
                break
            except Exception as e:
                print(f"Waiting for model registration... {e}")
                time.sleep(2)
                elapsed += 2
                if elapsed > 60:
                    raise Exception("Model registration failed!")

        # Promote to Production
        client.transition_model_version_stage(
            name="WaterQualityModel",
            version=latest_version,
            stage="Production"
        )
        print(f"Model promoted to Production. Version: {latest_version}")

    print("Train completed successfully!")
    return model
