# docsearch
It is a python library for Searching similar documents in a large corpus of documents based on my recent project.
It uses a 2-layer Earth Mover's Distance (my research) or a Jenson Shannon Distance over latent topic distribution of documents and word embeddings. 

I'll update the further methodology once my paper is published.
## Usage
```from docsearch import DocSearch
import pandas as pd

docsearch = DocSearch()

path = "path/to/dataset.csv"
df = pd.read_csv(path)

docsearch.fit(df['text'])

print docsearch.get_most_similar_documents([str(df.at[100, 'text'])])```
