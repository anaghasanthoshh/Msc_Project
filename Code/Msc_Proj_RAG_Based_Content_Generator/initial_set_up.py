from retrieval.config import PRODUCT_PATH,PROD_PROCESSED
import data_loader as dl
import retrieval.embedding as e
import pandas as pd
from utils.utils import log_timing
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("DataLoad_Embed")
with mlflow.start_run(run_name="run2") as run:
# One time runs
    data_loader=dl.DataLoader(PRODUCT_PATH)
    df = pd.read_csv(PROD_PROCESSED,sep='^')
    embed=e.EmbedData()
    with log_timing("Prod data Embedding"):
        embed.product_embedding(df)