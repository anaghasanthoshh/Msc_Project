import mlflow
import pandas as pd
import os
import shutil
from src.data_loader import load_data





# ====================Tracking Data Ingestion===================================

# data path where raw data exists for loading
data_path = "/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/"
processed_data_path = "/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/processed_data/"

# iterating through directory to get all file names present
file_path = [os.path.join(data_path, x) for x in os.listdir(data_path)]
print(file_path)
# ========================================================================


for index,path in enumerate(file_path):
    print(f"path inside for loop{path}")
    # calling the load_data to get values of metadata and filepaths
    metadata = load_data(path)
    # Log dataset parameters
    mlflow.log_param(f"raw_data_path_{index}", path)
    mlflow.log_param("processed_data_path", processed_data_path)
    mlflow.log_param(f"metadata{index}",metadata)

    print("Data Ingestion Tracked")


