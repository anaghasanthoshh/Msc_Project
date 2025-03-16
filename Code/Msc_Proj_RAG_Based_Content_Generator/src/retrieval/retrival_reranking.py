import pandas as pd

from retrieval.data_retrieval import query_to_vector,data_retrieval,k
from transformers import pipeline


query_text="Find long-lasting gel nail polish."
query_embed=query_to_vector(query_text)
product_result,review_result,prod_id_reviews=data_retrieval(query_embed)
#for i in range(k):product_result['ids']}
#print(review_result)#[0][i]['combined_text']
##TODO:The 'Add of existing embedding ID' is carried forward.Check.
'''
Features to be considered for reranking:
1.review_rating
2.sentiment_score
3.distance
'''

def sentiment_analysis(prod_id_reviews) :
    analysis = pipeline('sentiment-analysis')
    #count=len(prod_id_reviews)
    for i in range(len(prod_id_reviews['ids'])):
        data=prod_id_reviews['metadatas'][i]['text']
        result=analysis(data)
        prod_id_reviews['metadatas'][i]['label']=result[0]['label']
        prod_id_reviews['metadatas'][i]['senti_conf_score']=result[0]['score']
        return prod_id_reviews

#def combining_prod_and_reviews(prod_result,prod_id_reviews):
prod_data=pd.DataFrame(product_result)
review_data=pd.DataFrame(prod_id_reviews)
whole_data=pd.DataFrame.merge(prod_data,review_data,"left",on='parent_asin')
print(whole_data)









#fetch_prod_related_reviews(product_result)



