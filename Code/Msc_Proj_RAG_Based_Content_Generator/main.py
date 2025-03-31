from retrieval.config import PRODUCT_PATH, PROD_PROCESSED,GROUND_TRUTH
from generation.recommendation import Recommendation
from retrieval.data_retrieval import Retrieval
from retrieval.retrieval_eval import evaluate_retrieval
from utils import log_timing
import mlflow
import pandas as pd

# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retrieval_and_evaluation")
with mlflow.start_run(run_name="baseline_run") as run:
    ret = Retrieval()

## Block for executing single query ###
    query_text = input("Enter your query :")
    with log_timing("Query Embedding"):
        query_embed = ret.query_to_vector(query_text)
    with log_timing("Data Retrieval"):
        _, product_results = ret.data_retrieval(query_text, query_embed, "l2")
    print(f"Product results new:{product_results}")

    evaluation_results = evaluate_retrieval(product_results, k=5, run_name="retrieval_l2")
    print(f"Retrieval Evaluation results:\n{evaluation_results}\n")

    # rec = Recommendation(query_text, generation_results)
    # content = rec.generate_content()
    # print(f"Product Recommendation:\n\n{content}\n")
    # print(f"Explainability:{rec.explainability()}")
 ### End of Block-1 ###



