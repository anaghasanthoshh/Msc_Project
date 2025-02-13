#!/bin/bash

# Start MLflow tracking server
mlflow server \
#specify where data is stored
    --backend-store-uri sqlite:///mlflow_logs/mlflow.db \
#specifies where artifacts are stored
    --default-artifact-root ./mlflow_logs \
#access only to the local host
    --host 127.0.0.1 \
#mlflow uses this port
    --port 5000

#Note:
#./name to run
#ctrl+c to stop