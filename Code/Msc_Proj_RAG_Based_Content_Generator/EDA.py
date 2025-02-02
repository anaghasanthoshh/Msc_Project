import pandas as pd
import matplotlib.pyplot


review_data=pd.read_json('Data/Raw_Data/All_Beauty.jsonl',lines=True)
review_data=pd.DataFrame(review_data)

print(review_data.head())
