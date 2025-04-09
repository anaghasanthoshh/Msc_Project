
# ====================================================================================##
# One-time-run file as it generates embedding for the data.
# ====================================================================================##
# importing  required libraries
import pandas as pd
import math
from sentence_transformers import SentenceTransformer
import chromadb
from retrieval.config  import PRODUCT_PATH,PROD_PROCESSED,CHROMA_DB
from tqdm import tqdm
from enum import Enum
import uuid

# ====================================================================================##
## TODO:Check why we get 'Add of existing embedding ID' during product embedding
# ====================================================================================##
# Embedding and storing the data in chromadb
# ====================================================================================##

#model="all-MiniLM-L6-v2"
class Collection(Enum):
    MINI_CO = "product_cosine_mini"
    MINI_IP = "product_ip_mini"
    MINI_L2 = "product_l2_mini"
    QA_CO="product_cosine"
    QA_IP="product_ip"
    QA_L2="product_l2"

class EmbedData:
    def __init__(self,model,dbpath=CHROMA_DB):
        #load embedding model
        self.model = model
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB)# .PersistentClient () creates persistent storage of chromadb collections
        self.product_coll_l2 = self.chroma_client.get_or_create_collection(
                                            name=Collection.QA_L2.value,
                                            metadata={
                                                    "hnsw:space": "l2"
                                                      }
                                                    )
        self.product_coll_cosine = self.chroma_client.get_or_create_collection(
                                             name= Collection.QA_CO.value,
                                            metadata={
                                                    "hnsw:space": "cosine"
                                                     }
                                                )
        self.product_coll_ip = self.chroma_client.get_or_create_collection(
            name=Collection.QA_IP.value,
            metadata={
                "hnsw:space": "ip"
            }
        )

        #print("initialised")

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
            self.product_coll_l2.add(
                ids=data["item_id"].tolist(),
                embeddings=data["embeddings"].tolist(),
                metadatas=data[["category","metadata"]].to_dict(orient="records")
            )

            print(f"Product_embedding l2:Batch {i} completed.")
            self.product_coll_cosine.add(
                ids=data["item_id"].tolist(),
                embeddings=data["embeddings"].tolist(),
                metadatas=data[["category", "metadata"]].to_dict(orient="records")
            )
            print(f"Product_embedding ip:Batch {i} completed.")
            self.product_coll_ip.add(
                ids=data["item_id"].tolist(),
                embeddings=data["embeddings"].tolist(),
                metadatas=data[["category", "metadata"]].to_dict(orient="records")
            )
            #self.chroma_client.persist()
            print(f"Product_embedding cosine:Batch {i} completed.")



if __name__=="__main__":
    df = pd.read_csv(PROD_PROCESSED,sep='^')
    print('df loaded')
    model=SentenceTransformer("all-MiniLM-L6-v2")
    embed=EmbedData(model)
    embed.product_embedding(df)



