
# ====================================================================================##
# One-time-run file as it generates embedding for the data.
# ====================================================================================##
# importing  required libraries
import pandas as pd
import math
from sentence_transformers import SentenceTransformer
import chromadb
from retrieval.config  import PRODUCT_PATH,PROD_PROCESSED
from tqdm import tqdm
import uuid

# ====================================================================================##
## TODO:Check why we get 'Add of existing embedding ID' during product embedding
# ====================================================================================##
# Embedding and storing the data in chromadb
# ====================================================================================##

class EmbedData:
    def __init__(self,model="all-MiniLM-L6-v2",dbpath="../../Data/chroma_db"):
        #load embedding model
        self.model = SentenceTransformer(model)
        self.chroma_client=chromadb.PersistentClient(path=dbpath)# .PersistentClient () creates persistent storage of chromadb collections
        self.product_collection=self.chroma_client.get_or_create_collection("product_embeddings")
        print("initialised")

# noticed an issue of duplicates being added into collection. Hence, checking for
# existing ids
#     def fetch_existing_ids(self):
#         # Fetch all existing embeddings from the collection
#         existing_embeddings = self.product_collection.get(include=["ids"])
#         existing_ids = set(existing_embeddings['ids'])  # Convert to set for faster lookup
#         return existing_ids

# to embed product related text data into chroma db
    def product_embedding(self,df_product):
        #existing_ids = self.fetch_existing_ids()
        global batch_start, batch_end
        df_product["combined_text"]=df_product[["category","metadata"]] \
                .astype(str)\
                .fillna('')\
                .agg(" ".join,axis=1) \
                .str.strip()
        print('Products combined')
        df_product["embeddings"]=df_product["combined_text"].apply(self.model.encode)
        print(f"Number of null embeds:{df_product["embeddings"].isnull().sum()}")
        # Add to ChromaDB
        max_count=len(df_product)
        batch_size = 5461
        num_batches=math.ceil(max_count/batch_size)


        for i in tqdm(range(num_batches)):
            batch_start=i*batch_size
            batch_end=min(i*batch_size+batch_size,max_count)
            data=df_product[batch_start:batch_end]
            self.product_collection.add(
                ids=data["item_id"].tolist(),
                embeddings=data["embeddings"].tolist(),
                metadatas=data[["category","metadata"]].to_dict(orient="records")
            )
            print(f"Product_embedding:Batch {i} completed.")

if __name__=="__main__":
    df = pd.read_csv(PROD_PROCESSED,sep='^')
    print('df loaded')
    embed=EmbedData()
    embed.product_embedding(df)



