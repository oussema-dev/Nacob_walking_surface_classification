"""This module drops the non wanted classes from the dataset
and merges the specified classes together"""

import pandas as pd
import os

data_file = os.path.join("data", "concatenated_data.csv")
df = pd.read_csv(data_file)

# For example we will remove surfaces 2 and 3
surfaces_to_remove = [2, 3]
df = df[~df["class"].isin(surfaces_to_remove)]

# And we merge surfaces 1 and 4 together
df.loc[df["class"] == 1, "class"] = 0
df.loc[df["class"] == 4, "class"] = 0
df.loc[df["class"] == 5, "class"] = 1


df.to_csv(os.path.join("data", "concatenated_data.csv"), index=False)

print("Classes fused successfully")
