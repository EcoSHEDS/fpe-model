# test mlflow connection

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("test-mlflow-connection")

with mlflow.start_run():
    mlflow.log_metric("foo", 1)
    mlflow.log_metric("bar", 2)
