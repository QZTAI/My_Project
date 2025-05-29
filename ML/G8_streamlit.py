import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN, KMeans, AgglomerativeClustering
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from collections import defaultdict
from collections import Counter
from sklearn.manifold import TSNE
from collections import Counter
from bokeh.plotting import figure, show
from bokeh.models import Label
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from collections.abc import Iterable
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from gensim.models import Word2Vec
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from scipy.stats import kde
import networkx as nx
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import plotly.express as px
from os import path
from PIL import Image
from nltk import ngrams
from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
import pyLDAvis.gensim as gensimvis
import gensim
import pyLDAvis
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import matplotlib.pyplot as plt  
from nltk import pos_tag
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import SilhouetteVisualizer
import streamlit as st
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.corpora import Dictionary
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
# from utils.visualization import plot_figure

def preprocessing(data):
    # Lower case
    data = data.str.lower()
    # Remove Email
    data = data.apply(lambda x: cleaning_email(x))
    # Remove URL
    data = data.apply(lambda x: cleaning_URLs(x))
    # Remove Punctuations
    data = data.apply(lambda text: cleaning_punctuations(text))
    # Remove Number
    data = data.apply(lambda x: cleaning_numbers(x))
    # Remove Stopwords
    data = data.apply(lambda text: cleaning_stopwords(text))
    # Tokenizer
    data = data.apply(tokenizer.tokenize)
    # Lemmatizer
    data = data.apply(lambda x: lemmatizer_on_text(x))
    # Remove Veb
    data = data.apply(Keep_nouns)
    # Recombine Words
    data = data.apply(lambda y: ' '.join(y) if isinstance(y, Iterable) else '')
    return data

# Remove Email
def cleaning_email(df):
    return re.sub('@[^\s]+', ' ', df)

# Remove URL
def cleaning_URLs(df):
    return re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',df)

# Remove Punctuations
english_punctuations = string.punctuation
def cleaning_punctuations(text):
    translator = str.maketrans('', '', english_punctuations)
    return text.translate(translator)

# Remove Number 
def cleaning_numbers(data):
    return re.sub('[0-9]+', '', data)

# Remove Stopwords
stopwords_list = stopwords.words('english')
STOPWORDS = set(stopwords_list)
def cleaning_stopwords(text):
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

# Tokenisation
tokenizer = RegexpTokenizer(r'\w+')

# Lemmatizer
def lemmatizer_on_text(words):
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(word) for word in words]

# Function to Keep nouns using POS tagging
def Keep_nouns(tokenized_words):
    tagged_words = pos_tag(tokenized_words)
    words_without_nouns = [word for word, tag in tagged_words if tag == 'NN']
    return words_without_nouns if words_without_nouns else []

# Sampling
def sampling_data(data):
    sample_size=1067
    data = data.sample(n=sample_size, random_state=42)
    data.reset_index(drop=True, inplace=True)
    return data

def find_best_eps(data, min_eps=0.1, max_eps=1.0, step=0.05):
        best_score = -1
        best_eps = None
        for eps in np.arange(min_eps, max_eps, step):
            dbscan = DBSCAN(eps=eps, min_samples=3).fit(data)
            labels = dbscan.labels_
            if len(set(labels)) == 1 and -1 in labels:
                continue
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_eps = eps
        return best_eps, best_score

st.title('Topic Modelling on Twitter Comments')
    
header = ["target", "id", "date", "flag", "user", "text"]
# Read the CSV file with specified column names
df = pd.read_csv("x_dataset.csv", encoding="ISO-8859-1", names=header)

num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3)
sentiment = st.sidebar.selectbox('Comments Sentiment', ['Postive', 'Negative'])
sample_size=490

if sentiment == 'Postive':
    x = df[(df["target"] == 4)]["text"]
else:
    x = df[(df["target"] == 0)]["text"]
    
x = sampling_data(x)
text_data = preprocessing(x)  

# Vectorize the text data
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(text_data)

best_eps, best_score = find_best_eps(X_text)

# Clustering algorithms
clustering_algorithms = {
    "KMeans": KMeans(n_clusters=num_clusters),
    "Agglomerative Hierarchical": AgglomerativeClustering(n_clusters=num_clusters),
    "Latent Dirichlet Allocation": LatentDirichletAllocation(n_components=num_clusters),
    "Latent Semantic Analysis": TruncatedSVD(n_components=num_clusters),
    "DBSCAN": DBSCAN(eps=best_eps, min_samples=3),
}

selected_algorithm = st.sidebar.selectbox("Topic Modelling Algorithms", list(clustering_algorithms.keys()))

algorithm = clustering_algorithms[selected_algorithm]

if selected_algorithm =="DBSCAN":
    num_clusters = None
    algorithm.fit(X_text)
    labels = algorithm.labels_
    coords=X_text.toarray()
    
    no_clusters = len(np.unique(labels) )
    no_noise = np.sum(np.array(labels) == -1, axis=0)
    # Dimensionality reduction for visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(X_text.toarray())
    unique_labels = np.unique(labels)
    
    # Create DataFrame for Plotly Express
    db_data = pd.DataFrame({'Dimension 1': reduced_features[:, 0], 'Dimension 2': reduced_features[:, 1], 'Cluster': labels})
    
    # Plot using Plotly Express
    import plotly.express as px
    fig = px.scatter(db_data, x='Dimension 1', y='Dimension 2', color='Cluster', title='DBSCAN Clustering')

    # Show plot in Streamlit
    st.plotly_chart(fig)
    
    # Extracting keywords
    cluster_keywords = defaultdict(list)
    for idx, label in enumerate(labels):
        words = text_data[idx].split()
        cluster_keywords[label].extend(words)

    
    # Extracting keywords
    cluster_keywords = defaultdict(list)
    for idx, label in enumerate(labels):
        words = text_data[idx].split()
        cluster_keywords[label].extend(words)

    st.header("Cluster Keywords:")
    for cluster, keywords in cluster_keywords.items():
        # Count occurrences of each word
        word_counts = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1
        # Sort words by frequency and select top 5
        top_keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
        st.write(f"Cluster {cluster}: {', '.join(top_keywords)}")

elif selected_algorithm == "KMeans":    
    svd = TruncatedSVD(n_components=2)
    data = svd.fit_transform(X_text)
    algorithm=KMeans(n_clusters=num_clusters)
    algorithm.fit(data)
    labels = algorithm.labels_
    
    cluster_data = pd.DataFrame({'Dimension 1': data[:, 0], 'Dimension 2': data[:, 1], 'Cluster': labels})
    
    # Plot using Plotly Express
    import plotly.express as px
    fig = px.scatter(cluster_data, x='Dimension 1', y='Dimension 2', color='Cluster', title=f'Interactive Scatter Plot of Clusters (k={num_clusters})')

    # Show plot in Streamlit
    st.plotly_chart(fig)
    
    cluster_keywords = defaultdict(list)
    for idx, label in enumerate(labels):
        words = text_data[idx].split()
        cluster_keywords[label].extend(words)

    st.header("Topic Keywords:")
    for cluster, keywords in cluster_keywords.items():
        # Count occurrences of each word
        word_counts = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1
        # Sort words by frequency and select top 5
        top_keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
        st.write(f"Cluster {cluster}: {', '.join(top_keywords)}")

elif selected_algorithm == "Agglomerative Hierarchical":    
    svd = TruncatedSVD(n_components=2)
    data = svd.fit_transform(X_text)
    algorithm = AgglomerativeClustering(n_clusters=num_clusters)
    labels = algorithm.fit_predict(data)
    
    cluster_data = pd.DataFrame({'Dimension 1': data[:, 0], 'Dimension 2': data[:, 1], 'Cluster': labels})
    
    # Plot using Plotly Express
    import plotly.express as px
    fig = px.scatter(cluster_data, x='Dimension 1', y='Dimension 2', color='Cluster', title=f'Interactive Scatter Plot of Clusters (k={num_clusters})')

    # Show plot in Streamlit
    st.plotly_chart(fig)

    cluster_keywords = defaultdict(list)
    for idx, label in enumerate(labels):
        words = text_data[idx].split()
        cluster_keywords[label].extend(words)

    st.header("Topic Keywords:")
    for cluster, keywords in cluster_keywords.items():
        # Count occurrences of each word
        word_counts = defaultdict(int)
        for word in keywords:
            word_counts[word] += 1
        # Sort words by frequency and select top 5
        top_keywords = sorted(word_counts, key=word_counts.get, reverse=True)[:5]
        st.write(f"Cluster {cluster}: {', '.join(top_keywords)}")
    

elif selected_algorithm == "Latent Dirichlet Allocation":
    lda_model = LatentDirichletAllocation(n_components=num_clusters)  # Create a new LDA model
    lda_model.fit(X_text)  # Fit the LDA model to the data
    topics = lda_model.transform(X_text)

    topic_labels = np.argmax(topics, axis=1)

    lda_data = pd.DataFrame({'Topic 1': topics[:, 0], 'Topic 2': topics[:, 1], 'Cluster': topic_labels})
    
    # Plot using Plotly Express
    import plotly.express as px
    fig = px.scatter(lda_data, x='Topic 1', y='Topic 2', color='Cluster', title='Latent Dirichlet Allocation')

    # Show plot in Streamlit
    st.plotly_chart(fig)

    feature_names = vectorizer.get_feature_names_out()
    n_top_words = 2  # Number of top words to display for each topic
    st.header("Topic Keywords:")
    for topic_idx, topic in enumerate(lda_model.components_):
        top_features_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        st.write(f"Topic {topic_idx}: {', '.join(top_features)}")

elif selected_algorithm == "Latent Semantic Analysis":
    small_count_vectorizer = CountVectorizer()
    small_text_sample = text_data
    
    small_document_term_matrix = small_count_vectorizer.fit_transform(small_text_sample)
    
    lsa_model = TruncatedSVD(n_components=num_clusters, random_state=42)
    lsa_topic_matrix = lsa_model.fit_transform(small_document_term_matrix)
    
    def get_top_n_words(n, keys, document_term_matrix, count_vectorizer):
        top_word_indices = []
        for topic in range(len(set(keys))):
            temp_vector_sum = 0
            for i in range(len(keys)):
                if keys[i] == topic:
                    temp_vector_sum += document_term_matrix[i]
            temp_vector_sum = temp_vector_sum.toarray()
            top_n_word_indices = np.flip(np.argsort(temp_vector_sum)[0][-n:], 0)
            top_word_indices.append(top_n_word_indices)
        top_words = []
        for topic in top_word_indices:
            topic_words = []
            for index in topic:
                temp_word_vector = np.zeros((1, document_term_matrix.shape[1]))
                temp_word_vector[:, index] = 1
                the_word = count_vectorizer.inverse_transform(temp_word_vector)[0][0]
                topic_words.append(the_word)
            top_words.append(" ".join(topic_words))
        return top_words
    
    def get_keys(topic_matrix):
        keys = topic_matrix.argmax(axis=1).tolist()
        return keys
    
    def keys_to_counts(keys):
        count_pairs = Counter(keys).items()
        categories = [pair[0] for pair in count_pairs]
        counts = [pair[1] for pair in count_pairs]
        return (categories, counts)
    
    lsa_keys = get_keys(lsa_topic_matrix)
    lsa_categories, lsa_counts = keys_to_counts(lsa_keys)
    
    tsne_lsa_model = TSNE(n_components=2, perplexity=50, learning_rate=100,
                          n_iter=2000, verbose=1, random_state=0, angle=0.75)
    tsne_lsa_vectors = tsne_lsa_model.fit_transform(lsa_topic_matrix)
    
    n_topics = num_clusters
    colormap = np.array([
        "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
        "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
        "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
        "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5"])
    colormap = colormap[:n_topics]
    
    def get_mean_topic_vectors(keys, two_dim_vectors):
        mean_topic_vectors = []
        for t in range(n_topics):
            articles_in_that_topic = []
            for i in range(len(keys)):
                if keys[i] == t:
                    articles_in_that_topic.append(two_dim_vectors[i])
            
            if articles_in_that_topic:  # Check if the list is not empty
                articles_in_that_topic = np.vstack(articles_in_that_topic)
                mean_article_in_that_topic = np.mean(articles_in_that_topic, axis=0)
                mean_topic_vectors.append(mean_article_in_that_topic)
        return mean_topic_vectors
    
    top_3_words_lsa = get_top_n_words(3, lsa_keys, small_document_term_matrix, small_count_vectorizer)
    lsa_mean_topic_vectors = get_mean_topic_vectors(lsa_keys, tsne_lsa_vectors)
    
    # Plot using Plotly Express
    import plotly.express as px
    df = pd.DataFrame({'Dimension 1': tsne_lsa_vectors[:, 0], 'Dimension 2': tsne_lsa_vectors[:, 1], 'Cluster': lsa_keys})
    fig = px.scatter(df, x='Dimension 1', y='Dimension 2', color='Cluster', title='t-SNE Clustering of {} LSA Topics'.format(n_topics), color_discrete_sequence=colormap)
  
    
    # Show plot in Streamlit
    st.plotly_chart(fig)
  
    feature_names = small_count_vectorizer.get_feature_names_out()
    n_top_words = 5

    st.header("Topic Keywords:")
    for dimension_idx, dimension in enumerate(lsa_model.components_):
        top_features_idx = dimension.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_idx]
        st.write(f"Topic {dimension_idx}: {', '.join(top_features)}")
        