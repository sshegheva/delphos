import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans


class PeerRecommender(object):
    def __init__(self):
        self.innovation_list = []
        self.innovation_texts = []
        self.tfidf_vectorizer = TfidfVectorizer(max_df = 0.75,
                                                max_features = 400000,
                                                min_df = 0.05,
                                                stop_words='english',
                                                use_idf=True,
                                                tokenizer=self.tokenize,
                                                ngram_range=(1,4),
                                                decode_error='ignore')



    def generate_innovation_text_list(self):
        pass

    def perform_tfidf(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.innovation_texts)
        self.terms = self.tfidf_vectorizer.get_feature_names()

    def perform_clustering(self):
        pass

    def print_cluster(self):
        pass

    def tokenize(self, text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text.decode('utf8', 'ignore'))
                  for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token) and "'" not in token:
                filtered_tokens.append(token)
        return filtered_tokens
