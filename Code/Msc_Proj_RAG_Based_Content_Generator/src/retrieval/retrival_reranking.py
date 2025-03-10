from retrieval.data_retrieval import query_to_vector,data_retrieval,k

query_text="Find long-lasting gel nail polish."
query_embed=query_to_vector(query_text)
product_result,review_result,prod_id_reviews=data_retrieval(query_embed)
#for i in range(k):product_result['ids']}
print(review_result)#[0][i]['combined_text']
##TODO:The 'Add of existing embedding ID' is carried forward.Check.

def sentiment_analysis(prod_id_reviews) :




#fetch_prod_related_reviews(product_result)



