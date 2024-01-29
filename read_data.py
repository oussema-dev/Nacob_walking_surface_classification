"""This module reads the different types of provided data"""

import numpy as np
import pandas as pd
import os

# Read dataframe (.csv)
df = pd.read_csv(os.path.join("data", "concatenated_data.csv"))

# Print the data types of each column
print(df.dtypes)

# Print first row of the dataframe
print("dataframe", df.head(1))
