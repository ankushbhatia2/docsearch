# docsearch
It is a python library for Searching similar documents in a large corpus of documents based on my recent project.
It uses a 2-layer Earth Mover's Distance (my research) or a Jenson Shannon Distance over latent topic distribution of documents and word embeddings. 

I'll update the further methodology once my paper is published.


## Classes
DocSearch() :

(i)__init__() takes 5 optional arguments.
"""
        :param n_topics: number of topics (default 100)
        :param wv_size: word embedding dimension (default 100)
        :param stop_words: stop words list (default list) 
        :param min_word_freq: minimum word frequency (default 15000)
        :param sim_metric: allowed values :['jenson-shannon', 'emd']
"""

(ii) __fit__() takes one single argument which is the list of documents.

(iii) __get_most_similar_documents__() takes 2 arguments _viz._ query_document and number of similar documents to be shown(k).
## Usage
```from docsearch import DocSearch
import pandas as pd

docsearch = DocSearch()

path = "path/to/dataset.csv"
df = pd.read_csv(path)

docsearch.fit(df['text'])

print docsearch.get_most_similar_documents([str(df.at[100, 'text'])])```
