"""
how to actually get a recommender from the cluster
"""
import re
import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from matplotlib import pyplot as plt
import matplotlib as mpl


class PeerRecommender(object):
    def __init__(self):
        self.NUM_CLUSTERS = 10
        self.km = KMeans(n_clusters=self.NUM_CLUSTERS)
        self.tfidf_matrix = None
        self.terms = None
        self.presentation_frame = None
        self.cluster_plot_frame = None
        self.clusters = None
        self.innovation_list = []
        self.presentation_list = []
        self.presentation_texts = []
        self.tfidf_vectorizer = TfidfVectorizer(max_df=0.75,
                                                max_features=400000,
                                                min_df=0.05,
                                                stop_words='english',
                                                use_idf=True,
                                                tokenizer=self.tokenize,
                                                ngram_range=(1,4),
                                                decode_error='ignore')
        self.cluster_colours = {}


    def tokenize(self, text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text.decode('utf8', 'ignore'))
                  for word in nltk.word_tokenize(sent)]
        filtered_tokens = []
        for token in tokens:
            if re.search('[a-zA-Z]', token) and "'" not in token:
                filtered_tokens.append(token)
        return filtered_tokens

    def perform_tfidf(self):
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.presentation_texts)
        self.terms = self.tfidf_vectorizer.get_feature_names()

    def perform_clustering(self):
        self.km.fit(self.tfidf_matrix)
        self.clusters = self.km.labels_.tolist()
        presentations = {'presentation': self.presentation_list,
                         'text': self.presentation_texts,
                         'cluster': self.clusters,
                         'innovation': self.innovation_list}
        self.presentation_frame = pd.DataFrame(presentations,
                                               index=[self.clusters],
                                               columns=['presentation', 'cluster', 'text', 'innovation'])

    def recommend_peers(self, person):
        self.presentation_frame
        
    def presentation_team(self, presentation):
        return "team members"

    def print_clusters(self, num_terms=10):
        order_centroids = self.km.cluster_centers_.argsort()[:, ::-1]
        for i in range(self.NUM_CLUSTERS):
            print("Cluster %d words: " % i)
            top_words = [str(self.terms[term_index]) for term_index in order_centroids[i, :num_terms]]
            print("Top words: %s" % ','.join(top_words))
            print("")
            for book, title in zip(self.presentation_frame.ix[i]['book'], self.presentation_frame.ix[i]['chapter']):
                print("Presentation: %s, Innovation: %s" % (title, book))
            print("")

    def generate_cluster_plot_frame(self):
        mds = MDS(n_components=2, dissimilarity="precomputed", random_state=1)
        dist = 1 - cosine_similarity(self.tfidf_matrix)
        pos = mds.fit_transform(dist)
        xs, ys = pos[:, 0], pos[:, 1]
        cluster_data = dict()
        cluster_data["x"] = xs
        cluster_data["y"] = ys
        cluster_data["label"] = self.clusters
        cluster_data["presentation"] = self.presentation_list
        cluster_data["innovation_list"] = self.innovation_list
        self.cluster_plot_frame = pd.DataFrame(cluster_data)

    def plot_all_clusters(self):
        groups = self.cluster_plot_frame.groupby('innovation')
        fig, ax = plt.subplots(figsize=(17, 9))
        ax.margins(0.05)

        for name, group in groups:
            ax.plot(group.x, group.y, marker='o', linestyle='', ms=8,
                    color=self.cluster_colours[name], mec='none', label=name)
            ax.set_aspect('auto')
            ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
            ax.tick_params(axis='y', which='both', left='off', top='off', labelleft='off')

        ax.legend(numpoints=1)
        for i in range(len(self.cluster_plot_frame)):
            ax.text(self.cluster_plot_frame.ix[i]['x'],
                    self.cluster_plot_frame.ix[i]['y'],
                    self.cluster_plot_frame.ix[i]['presentation'], size=8)

        plt.savefig('all_clusters.png')
        plt.close()
