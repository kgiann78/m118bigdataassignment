from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import random
import numpy as np
from collections import defaultdict, Counter
import csv
from pyspark import SparkContext, SparkConf
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from string import punctuation
import matplotlib.pyplot as plt
import time
import pandas as pd
import zipfile
from sklearn import preprocessing

# Stem tokens
# Stemmers remove morphological affixes from words, leaving only the word stem.
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

# Tokenize text
# Tokenizers divide strings into lists of substrings.
def tokenize(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

# map document by distance
# Measure similarity according to the cosine distance
# and return the index with the shortest distance
def doc_mapper(doc):
    max_similarity = -1
    for i in range(len(centroids_prev)):
        temp_similarity = cosine_similarity([doc], [centroids_prev[i]])[0][0]
        if max_similarity <= temp_similarity:
            max_similarity = temp_similarity
            result = i
    return result, doc

# Produce a kmeans sampling in order to create initial centroids
def sample_kmeans(spark_context, sample_Xsvd, c_prev):
    # Distribute a local collection to form an RDD
    doc_parallelize_sample = spark_context.parallelize(sample_Xsvd, 250)
    finish = False

    while not finish:
        # Use map to the paralleled sample
        map_doc = doc_parallelize_sample.map(doc_mapper)

        # Hashmap of (key, value) pairs.
        # Values here will be the indices that correspond to the distance.
        cluster_counter = map_doc.countByKey()

        # Apply reduceByKey to return a distributed map of the centroids
        # with pairs of keys and new values
        temp_centroids = map_doc.reduceByKey(lambda q, p: (q + p)).collectAsMap()

        # create new centroids
        centroids_next = []
        for key, value in temp_centroids.iteritems():
            centroids_next.append(value / cluster_counter[key])

        # Continue until centroids now match.
        # In the end a new centroids array will be produced
        if not np.array_equal(centroids_next, c_prev):
            c_prev = centroids_next
        else:
            finish = True

    return c_prev


if __name__ == "__main__":

    # A SparkContext represents the connection to a Spark cluster,
    # and can be used to create RDDs, accumulators and broadcast variables
    # on that cluster.
    spark_conf = SparkConf().setAppName("kmeans_pyspark").setMaster("local[*]")
    sc = SparkContext(conf=spark_conf)

    start = int(round(time.time() * 1000))

    #
    #   Prepare data for applying KMeans
    #

    zf = zipfile.ZipFile('../Datasets-2016.zip')
    files = zf.open('train_set.csv')
    df = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content', 'Category'])
    # Concatenate Title and Content as text
    df['text'] = df['Title'].map(str)+df['Content']

    # Convert a collection of raw documents to a matrix of TF-IDF features.
    # TF-IDF: How important a word is to a document in a collection
    tfidf_vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english')

    # Learn vocabulary and idf, return term-document matrix for concatenated text
    X_tfidf = tfidf_vect.fit_transform(df['text'][1:])

    # Encode labels with value between 1 to n (where n the count of values).
    y_names = set(df['Category'][1:].values)
    le = preprocessing.LabelEncoder()

    # Learn vocabulary and idf, return term-document matrix for Categories
    y = le.fit_transform(df['Category'][1:])

    finished = False
    max_iter = 10
    # Dimensionality reduction using truncated SVD for the term-document matrix of the concatenated text produced.
    model = TruncatedSVD(n_components=100).fit(X_tfidf)
    # Perform dimensionality reduction on X_tfidf
    X_svd = model.transform(X_tfidf)

    # Extract a random sample from the reduced X_tfidf
    sample_Xsvd = random.sample(X_svd, int(len(X_svd)*0.30))
    # Get random centroids
    centroids_prev = random.sample(sample_Xsvd, 5)
    # Apply sampling KMeans to produce final centroids that
    # are going to be used for the actual KMeans
    centroids_prev = sample_kmeans(sc, sample_Xsvd, centroids_prev)

    # Parallelize X_svd
    doc_parallelize = sc.parallelize(X_svd, 250)

    #
    #   Apply KMeans to data
    #

    j = 0

    while not finished and j <= max_iter:
        map_doc = doc_parallelize.map(doc_mapper)

        cluster_counter = map_doc.countByKey()
        temp_centroids = map_doc.reduceByKey(lambda q, p: (q + p)).collectAsMap()
        centroids_next = []
        for key, value in temp_centroids.iteritems():
            centroids_next.append(value / cluster_counter[key])

        if not np.array_equal(centroids_next, centroids_prev):
            centroids_prev = centroids_next
        else:
            finished = True
        j += 1

    doc_dict = map_doc.keys().collect()

    # Create/populate cluster for centroid
    i = 0
    cluster_dict = defaultdict(list)
    for d in doc_dict:
        cluster_dict[d].append(y[i])
        i += 1

    end = int(round(time.time() * 1000))

    print "Execution Time: "+str(end-start)

    #
    #   Export results
    #

    column_labels = []
    row_labels = []
    data = np.zeros(shape=(len(y_names), len(y_names)))
    with open('clustering_KMeans.csv', 'w') as csvfile:
        fieldnames = [' ']
        for names in y_names:
            fieldnames.append(names)
            row_labels.append(names)

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        i = 0
        for key in cluster_dict:
            temp_cluster_dict = {' ': 'Cluster'+str(key+1)}
            column_labels.append('Cluster'+str(key+1))
            counter = Counter(cluster_dict[key])
            j = 0
            for k, value in counter.iteritems():
                temp_cluster_dict[le.inverse_transform(k)] = value/float(sum(counter.values()))
                data[i][j] = value/float(sum(counter.values()))
                j += 1
            writer.writerow(temp_cluster_dict)
            i += 1

    #
    #   Create plot for results
    #

    data = np.array(data)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('clustering_KMeans.png')