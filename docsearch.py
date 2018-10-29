# coding: utf-8
from gensim.models import LdaModel, Word2Vec
from gensim import models, corpora, similarities
from nltk.stem.porter import PorterStemmer
from nltk import FreqDist
from scipy.stats import entropy

from pyemd import emd
from scipy.spatial.distance import cosine

import pandas as pd
import numpy as np
import nltk
import re
import stop_words

#STOP WORDS
stopWords = stop_words.get_stop_words('english')

#STEMMER
stemmer = PorterStemmer()


class _LDA(object):
    def __init__(self):
        """
        This is the LDA class constructor for initialzing class variables
        """
        self.dictionary = None
        self.corpus = None
        self.lda = None
        self.doc_topic_distribution = None

    def train_lda(self, data):
        """
        This function trains the lda model
        We setup parameters like number of topics, the chunksize to use in Hoffman method
        We also do 2 passes of the data since this is a small dataset, so we want the distributions to stabilize
        """
        num_topics = 100
        chunksize = 300
        print("Preparing Dictionary")
        self.dictionary = corpora.Dictionary(data['tokenized'])
        self.corpus = [self.dictionary.doc2bow(doc) for doc in data['tokenized']]
        # low alpha means each document is only represented by a small number of topics, and vice versa
        # low eta means each topic is only represented by a small number of words, and vice versa
        print("Training Topic Model")
        self.lda = LdaModel(corpus=self.corpus, num_topics=num_topics, id2word=self.dictionary,
                       alpha=1e-2, eta=0.5e-2, chunksize=chunksize, minimum_probability=0.0, passes=2)
        self.doc_topic_distribution = np.array([[tup[1] for tup in lst] for lst in self.lda[self.corpus]])
        self.doc_topic_distribution = self.doc_topic_distribution.astype(np.float64)

class DocSearch(object):
    def __init__(self, n_topics=100, wv_size=100, stop_words=stopWords, min_word_freq=15000, sim_metric='emd'):
        """

        :param n_topics: number of topics
        :param wv_size: word embedding dimension
        :param stop_words: stop words list
        :param min_word_freq: minimum word frequency
        :param sim_metric: allowed values :['jenson-shannon', 'emd']
        """
        #INITIALIZING VARIABLES
        self.n_topics = n_topics
        self.wv_size = wv_size
        self.stop_words = stop_words[:]
        self.__df = pd.DataFrame({})
        self.wv = None
        self.min_word_freq = min_word_freq
        self.lda = _LDA()
        self.top_k_words = None
        self.sim_metric = sim_metric
        if self.sim_metric not in ['jenson-shannon', 'emd']:
            raise ValueError("Allowed values for sim_metric are [jenson-shannon, emd]")

        #BOTTOM LAYER
        self.topic_matrix = None
        self.topics = None
        self.word_sim_matrix = None

        #TOP LAYER MATRIX
        self.doc_metric_space = np.zeros((self.n_topics, self.n_topics), dtype=np.float64)


    def __initial_clean(self, text):
        """
        Cleaning text
        :param text: document
        :return: cleaned document
        """
        text = re.sub("((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)", " ", text)
        text = re.sub("[^a-zA-Z ]", "", text)
        text = text.lower()  # lower case the text
        text = nltk.word_tokenize(text)
        return text

    def __remove_stop_words(self, text):
        """
        Removing stop words
        :param text: document
        :return: cleaned document
        """
        return [word for word in text if word not in self.stop_words]

    def __stem_words(self, text):
        """
        Stemming
        :param text: document
        :return: stemmed document
        """
        try:
            text = [stemmer.stem(word) for word in text]
            text = [word for word in text if len(word) > 1]
        except IndexError:
            pass
        return text

    def __apply_all(self, text):
        """
        Final cleaning function
        :param text: document from a series
        :return: cleaned document
        """
        return self.__stem_words(self.__remove_stop_words(self.__initial_clean(text)))

    def __jensen_shannon(self, query, matrix):
        """
        Utility function for comparing emd with jenson-shannon
        :param query: query document
        :param matrix: document-distance matrix
        :return: similarity with each document
        """
        p = query[None, :].T
        q = matrix.T
        m = 0.5 * (p + q)
        return np.sqrt(0.5 * (entropy(p, m) + entropy(q, m)))

    def __emd_query(self, query, matrix):
        """
        Same as jenson shannon but uses Earth-Mover's Distane to get similar documents
        :param query: query document
        :param matrix: document-distance matrix
        :return: similarity with each document
        """
        return np.array([emd(query, lst, self.doc_metric_space) for lst in matrix])

    def __get_most_similar_documents(self, query, k=10):
        """
        Function for calling the similarity measure
        :param query: query document
        :param k: number of similar documents
        :return: top k similar documents
        """
        if self.sim_metric == 'jenson-shannon':
            sims = self.__jensen_shannon(query, self.lda.doc_topic_distribution)  # list of jensen shannon distances
        else:
            sims = self.__emd_query(query, self.lda.doc_topic_distribution)
        return sims.argsort()[:k], sorted(sims)[:k]

    def __get_doc_distribution(self, doc):
        """
        Get topic distribution for a document
        :param doc: document
        :return: np.ndarray
        """
        new_bow = self.lda.dictionary.doc2bow(doc)
        new_doc_distribution = np.array([tup[1] for tup in self.lda.lda.get_document_topics(bow=new_bow)])
        new_doc_distribution = new_doc_distribution.astype(np.float64)
        return new_doc_distribution

    def fit(self, docs):
        """
        Main function to train over a large corpus of documents
        :param docs: List of documents
        :return: None
        """
        # INITIAL DATAFRAME
        print("Initializing dataframe")
        self.__df['text'] = docs
        self.__df = self.__df[self.__df['text'].map(type) == str]
        self.__df = self.__df.sample(frac=1.0)
        self.__df.reset_index(drop=True, inplace=True)

        # CLEANING
        print("Tokenizing Data")
        self.__df['tokenized'] = self.__df['text'].apply(self.__apply_all)
        all_words = [word for item in self.__df['tokenized'].tolist() for word in item]
        fdist = FreqDist(all_words)
        self.top_k_words, _ = zip(*fdist.most_common(self.min_word_freq))
        self.top_k_words = set(self.top_k_words)
        self.__df['tokenized'] = self.__df['tokenized'].apply(lambda text: [word for word in text if word in self.top_k_words])


        #TOPIC TRAINING
        print("Topic modelling")
        self.lda.train_lda(self.__df)

        """ LAYER-2 MATRIX"""
        print("Bottom Layer")
        #WORD2VEC
        print("     Getting Word Embeddings")
        self.wv = Word2Vec(sentences=self.__df.tokenized, workers=4, size=100)

        self.topics = self.lda.lda.print_topics(num_topics=100, num_words=10)
        M = []
        unique_words = set()
        def __get_word_prob(j):
            prob, word = (float(j.split('*')[0]), j.split('*')[1].strip().strip('"'))
            unique_words.add(word)
            return prob, word
        print("     Initializing Topic Distance Matrix")
        for i in self.topics:
            tops = i[1].split('+')
            M.append([__get_word_prob(j) for j in tops])
        unique_words = list(unique_words)

        self.topic_matrix = np.zeros((self.n_topics, len(unique_words)), dtype=np.float64)
        for ind, i in enumerate(M):
            for prob, word in i:
                self.topic_matrix[ind, unique_words.index(word)] = prob

        self.word_sim_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=np.float64)
        for i in xrange(len(unique_words)):
            for j in xrange(i, len(unique_words)):
                self.word_sim_matrix[i, j] = cosine(self.wv[unique_words[i]], self.wv[unique_words[j]])
                self.word_sim_matrix[j, i] = cosine(self.wv[unique_words[j]], self.wv[unique_words[i]])


        """LAYER-1 MATRIX"""
        print("Top Layer")
        for i in xrange(self.n_topics):
            for j in xrange(self.n_topics):
                self.doc_metric_space[i, j] = emd(self.topic_matrix[i], self.topic_matrix[j], self.word_sim_matrix)
        print("Complete")


    def get_most_similar_documents(self, query_doc, k=10):
        """
        Utility function for getting the most similar documents
        :param query_doc: query document
        :param k: number of similar documents
        :return: returns a list of similar documents and their similarity score
        """
        test_df = pd.DataFrame({'text':query_doc})
        test_df['tokenized'] = test_df['text'].apply(self.__apply_all)
        test_df['tokenized'] = test_df['tokenized'].apply(
            lambda text: [word for word in text if word in self.top_k_words])

        test_df['doc_dist'] = test_df['tokenized'].apply(self.__get_doc_distribution)
        most_sim_docs, sim_scores = [], []
        for doc_dist in test_df['doc_dist'].tolist():
            ids, scores = self.__get_most_similar_documents(doc_dist, k)
            most_sim_docs+=self.__df.loc[ids, 'text'].tolist()
            sim_scores+=scores
        return zip(most_sim_docs, sim_scores)
