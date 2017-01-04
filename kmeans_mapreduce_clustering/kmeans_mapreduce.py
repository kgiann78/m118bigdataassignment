from sklearn.datasets import load_files
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


def sample_kmeans(spark_context, sample_Xsvd, c_prev):
    doc_parallelize_sample = spark_context.parallelize(sample_Xsvd, 250)
    finish = False

    def doc_mapper(doc):
        max_similarity = -1
        for i in range(len(c_prev)):
            temp_similarity = cosine_similarity([doc], [c_prev[i]])[0][0]
            if max_similarity <= temp_similarity:
                max_similarity = temp_similarity
                result = i

        return result, doc

    while not finish:
        map_doc = doc_parallelize_sample.map(doc_mapper)

        cluster_counter = map_doc.countByKey()
        temp_centroids = map_doc.reduceByKey(lambda q, p: (q + p)).collectAsMap()
        centroids_next = []
        for key, value in temp_centroids.iteritems():
            centroids_next.append(value / cluster_counter[key])

        if not np.array_equal(centroids_next, c_prev):
            c_prev = centroids_next
        else:
            finish = True

    return c_prev


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    stemmer = PorterStemmer()
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

if __name__ == "__main__":
    spark_conf = SparkConf().setAppName("kmeans_pyspark").setMaster("local[*]")
    sc = SparkContext(conf=spark_conf)

    start = int(round(time.time() * 1000))

    data_train = load_files('../Dataset/')
    tfidf_vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english')
    X_tfidf = tfidf_vect.fit_transform(data_train.data)
    y = data_train.target
    y_names = data_train.target_names
    finished = False
    max_iter = 10
    #model = TruncatedSVD(n_components=100).fit(X_tfidf)
    #X_svd = model.transform(X_tfidf)

    #sample_Xsvd = random.sample(X_svd, int(len(X_svd)*0.30))
    #centroids_prev = random.sample(sample_Xsvd, 5)
    #centroids_prev = sample_kmeans(sc, sample_Xsvd, centroids_prev)

    #centroids_prev = random.sample(X_svd, 5)

    doc_parallelize = sc.parallelize(X_tfidf, 250)

    def doc_mapper(doc):
        max_similarity = -1
        for i in range(len(centroids_prev)):
            temp_similarity = cosine_similarity([doc], [centroids_prev[i]])[0][0]
            if max_similarity <= temp_similarity:
                max_similarity = temp_similarity
                result = i
        return result, doc


    j = 0
    #while not finished:
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
    i = 0
    cluster_dict = defaultdict(list)
    for d in doc_dict:
        cluster_dict[d].append(y[i])
        i += 1

    print 'max_iter: '+str(j)
    print centroids_prev
    end = int(round(time.time() * 1000))

    print "Execution Time: "+str(end-start)

    column_labels = []
    row_labels = []
    data = np.zeros(shape=(len(y_names), len(y_names)))
    with open('clustering_KMeans_tfidf.csv', 'w') as csvfile:
        fieldnames = [' ']
        for names in y_names:
            fieldnames.append(names[:-1])
            row_labels.append(names[:-1])

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        i = 0
        for key in cluster_dict:
            temp_cluster_dict = {' ': 'Cluster'+str(key+1)}
            column_labels.append('Cluster'+str(key+1))
            counter = Counter(cluster_dict[key])
            j = 0
            for k, value in counter.iteritems():
                temp_cluster_dict[y_names[k][:-1]] = value/float(sum(counter.values()))
                data[i][j] = value/float(sum(counter.values()))
                j += 1
            writer.writerow(temp_cluster_dict)
            i += 1

    data = np.array(data)
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(data, cmap=plt.cm.Blues)

    ax.set_xticks(np.arange(data.shape[0])+0.5, minor=False)
    ax.set_yticks(np.arange(data.shape[1])+0.5, minor=False)

    ax.invert_yaxis()
    ax.xaxis.tick_top()

    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(column_labels, minor=False)
    plt.savefig('clustering_KMeans_tfidf.png')