# products_features.parquet
# This file contains the CLIP embedding features for all product_id's in the dataset
# queries_features.parquet
# This file contains the CLIP text embedding features for all querie_id's in the dataset
import pandas as pd

df = pd.read_parquet("hf://datasets/crossingminds/shopping-queries-image-dataset/data/product_features.parquet")

df.head()