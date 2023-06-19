import pandas as pd

# Read the CSV files as dataframes
df1 = pd.read_csv('data_x.csv')
df2 = pd.read_csv('data_y.csv')
df3 = pd.read_csv('data_eq.csv')

# Concatenate the dataframes vertically
concatenated_df = pd.concat([df1, df2, df3], ignore_index=True)

# Save the concatenated dataframe as a new CSV file
concatenated_df.to_csv('concatenated.csv', index=False)