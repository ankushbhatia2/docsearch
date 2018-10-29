from docsearch import DocSearch
import pandas as pd

docsearch = DocSearch()

path = "path/to/dataset.csv"
df = pd.read_csv(path)

docsearch.fit(df['text'])

print docsearch.get_most_similar_documents([str(df.at[100, 'text'])])
