import pandas as pd
import os
import mlflow
from bs4 import BeautifulSoup
from retrieval.config import PRODUCT_PATH, PROD_PROCESSED
from dotenv import load_dotenv


class DataLoader:
    def __init__(self, prod_path):
        self.prod_path = prod_path
        self.product_count = 0
        self.product_df = pd.DataFrame()

    # TODO: load the data  and return metadata,dataframes for tracking details in MLflow_tracking.py
    def load_data(self):

        # wrapping JSON string in StringIO before passing it to `read_json`
        self.product_df = pd.read_json(self.prod_path)
        self.product_count = self.product_df.shape[0]

        p_metadata = {
            "dat_path": self.prod_path,
            "row_count": self.product_count,
            "column_names": self.product_df.columns
        }
        mlflow.log_param("dat_path", self.prod_path)
        mlflow.log_param("prod_data_count", self.product_count)
        mlflow.log_param("column_names", self.product_df.columns)
        return self.product_df

    @staticmethod
    def remove_html_soup(text):
        return BeautifulSoup(str(text), "html.parser").get_text()

    # TODO: Need check for language,removal of html tags from text
    def preprocessing_product_data(self):

        prod_df = self.product_df.drop_duplicates().copy()

        # to fill na,lowercase
        column_list = prod_df.columns
        for column in column_list:
            prod_df[column] = prod_df[column].fillna(lambda x: 0.0 if isinstance(x, (int, float)) else '')
            prod_df[column] = (prod_df[column].apply
                               (lambda x: x.lower() if isinstance(x, str) else x))
            prod_df[column] = (prod_df[column].apply
                               (lambda x: DataLoader.remove_html_soup(x) if isinstance(x, str) else x))

        if os.path.exists(PROD_PROCESSED):
            os.remove(PROD_PROCESSED)
        prod_df.to_csv(PROD_PROCESSED, sep='^')
        return 'File saved as csv'


if __name__ == "__main__":
    # calling the functions performing the data_cleaning/data loading
    data_loader = DataLoader(PRODUCT_PATH)
    print(data_loader.load_data().head())
    print(data_loader.preprocessing_product_data())
