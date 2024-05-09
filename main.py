import pandas as pd

df = pd.read_csv('smilestyle_dataset.tsv', sep = '\t')
print(df['formal'])