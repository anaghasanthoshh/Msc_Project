import pandas as pd
import os
from io import StringIO

#dataframe=[]

#load the data  and return metadata,dataframes for tracking details in MLflow_tracking.py
def load_data(path):
    df=pd.read_json(path)
    metadata={
        "dat_path":path,
        "row_count":df.shape[0],
        "column_names":df.columns
        }
        #dataframe.append(df)

    #return metadata
    return df



