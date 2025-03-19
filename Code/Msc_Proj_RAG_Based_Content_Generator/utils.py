from contextlib import contextmanager
import mlflow
import time

@contextmanager
def log_timing(label):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    mlflow.log_metric(f"{label}_time_seconds", elapsed_time)