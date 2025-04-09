from contextlib import contextmanager
import mlflow
import time

@contextmanager
def log_timing(label):
    start_time = time.time()
    yield
    elapsed_time = time.time() - start_time
    mlflow.log_metric(f"{label}_time_seconds", elapsed_time)

def print_banner(*lines):
    print("\n" + "*" * 40)
    for line in lines:
        print(line)
    print("*" * 40 + "\n")