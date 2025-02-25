from retrieval.data_retrieval import query_to_vector,data_retrieval



query_embedding=query_to_vector(query_text)
presult,rresult=data_retrieval(query_embedding)
print(presult,rresult)