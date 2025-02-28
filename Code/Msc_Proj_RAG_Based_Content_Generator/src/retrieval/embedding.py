'''
Models to be considered:
1.all-MiniLM-L6-v2
2.multi-qa-MiniLM-L6-cos-v1
3.paraphrase-mpnet-base-v2
'''
# ====================================================================================##
# One-time-run file as it generates embedding for the data.
# ====================================================================================##
# importing  required libraries
import pandas as pd
from io import StringIO
from functools import lru_cache
#from data_loader import load_data, PRODUCT_PATH, REVIEW_PATH
from sentence_transformers import SentenceTransformer
import chromadb
import uuid

# ====================================================================================##

## TODO:Convert .add() to batchwise insertion of 5461(max value of chromadb)

#load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# ====================================================================================##
'''considering that our data is amazon reviews/ product descriptions and are relatively
shorter sentences, we will do basic cleaning like handling of duplicates.
Since sentence transformers automatically handle tokenisation, its not performed'''

# ====================================================================================##
# Embedding and storing the data in chromadb
# ====================================================================================##

def chroma_clients_collection():
    chroma_client = chromadb.PersistentClient(path="../../Data/chroma_db")
    # .PersistentClient () creates persistent storage of chromadb collections
    #chroma_client.delete_collection(name="product_embeddings")
    product_collection = chroma_client.get_or_create_collection("product_embeddings")
    #chroma_client.delete_collection(name="review_embeddings")
    review_collection = chroma_client.get_or_create_collection("review_embeddings")
    print("inside chroma_clients function")
    return product_collection, review_collection


# noticed an issue of duplicates being added into collection. Hence, checking for
# existing ids
def fetch_existing_ids(collection):
    # Fetch all existing embeddings from the collection
    existing_embeddings = collection.get()
    existing_ids = set(existing_embeddings['ids'])  # Convert to set for faster lookup
    return existing_ids

# to embed product related text data into chroma db
def product_embedding(df_product, product_collection):
    #existing_ids = fetch_existing_ids(product_collection)

    df_product["combined_text"]=df_product[["description", "title", "features", "details"]] \
                .astype(str)\
                .fillna('')\
                .agg(" ".join,axis=1) \
                .str.strip()

    df_product["embeddings"]=df_product["combined_text"].apply(model.encode)

        # Add to ChromaDB
    product_collection.add(
                ids=df_product["parent_asin"].tolist(),
                embeddings=df_product["embeddings"].tolist(),
                metadatas=df_product[["title", "price", "combined_text"]].to_dict(orient="records")
        )


#  to embed review text data and adding them to chromadb
def review_embedding(df_review, review_collection):
    df_review["combined_text"]=df_review[["title","text"]]\
        .astype(str)\
        .fillna('')\
        .agg(" ".join,axis=1) \
        .str.strip()
    df_review["embedding"]=df_review["combined_text"].apply(model.encode)
    df_review["unique_id"]=df_review.apply(lambda _:str(uuid.uuid4()),axis=1)
   # Add to ChromaDB
    review_collection.add(
            ids=df_review["unique_id"].tolist(),
            embeddings=df_review["embedding"].tolist(),
            metadatas=df_review[["title","text"]].to_dict(orient="records")
            )





#df_product, df_review = load_data(PRODUCT_PATH, REVIEW_PATH)
#dfr = preprocessing_review_data(df_review)
#dfp = preprocessing_product_data(df_product)
p_collection, r_collection = chroma_clients_collection()
dfp=pd.read_csv('../../Data/processed_data/processed_prod_data.json',sep='^')
dfr=pd.read_csv('../../Data/processed_data/processed_review_data.json',sep='^')

product_embedding(dfp, p_collection)
review_embedding(dfr, r_collection)
