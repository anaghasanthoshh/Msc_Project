# ====================================================================================##
# data indexing module for building search indices and vector stores
# ====================================================================================##
#importing required libraries
import shutil
from whoosh.fields import Schema, TEXT, ID
from whoosh.index import create_in
from whoosh.analysis import StemmingAnalyzer
import os
from retrieval.config import PROD_PROCESSED,WHOOSH_INDEX
import pandas as pd

df=pd.read_csv(PROD_PROCESSED,sep="^")
# ====================================================================================##
# create_search_index: builds a whoosh index from processed metadata
# ====================================================================================##
def whoosh_index(df, index_dir=WHOOSH_INDEX):
    schema = Schema(
        item_id=ID(stored=True),
        category=TEXT(stored=True, analyzer=StemmingAnalyzer()),
        metadata=TEXT(stored=True, analyzer=StemmingAnalyzer())
    )

    if not os.path.exists(index_dir):
        os.mkdir(index_dir)
    else:
        # Clear and re-create index
        shutil.rmtree(WHOOSH_INDEX)

    ix = create_in(index_dir, schema)
    writer = ix.writer()

    for _, row in df.iterrows():
        writer.add_document(
            item_id=row["item_id"],
            category=row.get("category", ""),
            metadata=row.get("metadata", "")
        )
    writer.commit()
# ====================================================================================##
# command-line interface for running the indexing tasks
# ====================================================================================##
if __name__=="__main__":
    whoosh_index(df, index_dir=WHOOSH_INDEX)