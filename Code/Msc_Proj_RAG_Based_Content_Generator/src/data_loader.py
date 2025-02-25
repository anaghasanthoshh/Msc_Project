import pandas as pd
import os
from io import StringIO
PRODUCT_PATH='/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/product_sample.json'
REVIEW_PATH='/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/raw_data/review_sample.json'


#load the data  and return metadata,dataframes for tracking details in MLflow_tracking.py
def load_data(p_path,r_path):
    product_df=pd.read_json(p_path)
    p_metadata={
        "dat_path":p_path,
        "row_count":product_df.shape[0],
        "column_names":product_df.columns
        }
    review_df=pd.read_json(r_path)
    r_metadata={
        "dat_path":r_path,
        "row_count":review_df.shape[0],
        "column_names":review_df.columns
        }


    #return metadata
    return product_df,review_df



