from retrieval.config import PRODUCT_PATH,PROD_PROCESSED
from generation.recommendation import Recommendation
from retrieval.data_retrieval import Retrieval
import retrieval.retrieval_eval as reval
from utils import log_timing
import mlflow


# Set MLflow tracking server URI
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("retrieval_and_evaluation")
with mlflow.start_run(run_name="baseline_run") as run:

    ret=Retrieval()
    query_text=input("Enter your query :")
    with log_timing("Query Embedding"):
        query_embed=ret.query_to_vector(query_text)
    with log_timing("Data Retrieval"):
        generation_results,product_results=ret.data_retrieval(query_text,query_embed)

    evaluation_results = reval.evaluate_retrieval(product_results, k=5, run_name="baseline_retrieval")
    print(f"Retrieval Evaluation results:\n{evaluation_results}\n")

    rec=Recommendation(query_text,generation_results)
    content=rec.generate_content()
    print(f"Product Recommendation:\n\n{content}\n")
    print(f"Explainability:{rec.explainability()}")




