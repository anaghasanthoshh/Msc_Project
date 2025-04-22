import pandas as pd
import numpy as np
import ast
from retrieval.config import GROUND_TRUTH

ground_truth_path = GROUND_TRUTH
gt_df = pd.read_csv(ground_truth_path)
gt_df=gt_df[gt_df["split"]=="train"]

ground_truth_length=[]
for idx,row in gt_df.iterrows():
    ground_truths=ast.literal_eval(row["item_id"])
    No_ground_truths=len(ground_truths)
    ground_truth_length.append(No_ground_truths)
print(np.mean(sorted(ground_truth_length,reverse=True)))
print(np.median(sorted(ground_truth_length,reverse=True)))


