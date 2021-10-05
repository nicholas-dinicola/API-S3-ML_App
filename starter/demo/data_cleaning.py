import pandas as pd
import os
import starter
# read the csv file

root_dir = os.path.basename(os.path.abspath(starter.__file__))
df = pd.read_csv(os.path.join(root_dir, "data", "census.csv"))

# remove spaces in column names
df.columns = df.columns.str.replace(' ', '')

print(df.columns)
print(df.info())

# save the new dataset
df.to_csv("data/census_no_spaces.csv", index=False)