import os


DATA='/Users/anu_nambiar/PycharmProjects/Msc_Project/Code/Msc_Proj_RAG_Based_Content_Generator/Data/'

PRODUCT_PATH=os.path.join(DATA,'Raw_Data/final_meta_sample.json')
PROD_PROCESSED=os.path.join(DATA,'Processed_Data/final_processed_metadata.csv')
GROUND_TRUTH=os.path.join(DATA,'Raw_Data/ground_truth.csv')
CHROMA_DB=os.path.join(DATA,'chroma_db/')
WHOOSH_INDEX=os.path.join(DATA,'whoosh/')