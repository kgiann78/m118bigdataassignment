from svm import svm
from random_forest import random_forest
from neural_networks import neural_networks
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import csv
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from string import punctuation
import numpy as np
from gensim.models import Word2Vec
from sklearn.decomposition import TruncatedSVD
import pandas as pd
import zipfile
from sklearn import preprocessing
import string

stemmer = PorterStemmer()

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
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems

def avg_feature_vector(words, model, num_features, index2word_set):

    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec

if __name__ == "__main__":

    #
    #   Prepare data
    #
    zf = zipfile.ZipFile('../Datasets-2016.zip')
    files = zf.open('train_set.csv')
    df = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content', 'Category'])

    # Concatenate Title and Content into text
    df['text'] = df['Title'].map(str)+df['Content']

    #
    #   BOW classification
    #

    #   Create the tf-idf "vectorizer" for words
    tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
    # Produce the tf-idf vector
    X_tfidf = tfidf_vect.fit_transform(df['text'][1:])


    #
    # tf-idf stemmer for neural networks
    #

    # Create the tf-idf vectorized stemmer
    tfidf_vect_stemmer = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english')
    # Produce the stemmed tf-idf vector
    X_tfidf_stemmer = tfidf_vect_stemmer.fit_transform(df['text'][1:])

    # Create and produce labels from 1 to n
    le = preprocessing.LabelEncoder()
    y = le.fit_transform(df['Category'][1:])
    classes = list(set(y))
    n_classes = len(classes)

    #
    #   SVD
    #
    model = TruncatedSVD(n_components=90).fit(X_tfidf)
    X_svd = model.transform(X_tfidf)

    #
    #   Create and populate words' vector
    #
    doc_word = []
    for i in range(1, df['text'].count()):
        doc_word.append(str(df['text'][i]).translate(None, string.punctuation).split())

    #
    #   W2V
    #
    num_features = 100
    min_word_count = 50
    num_workers = 20
    context = 300

    word2vec_model = Word2Vec(doc_word, workers=num_workers,
                              size=num_features, min_count=min_word_count,
                              window=context)
    word2vec_model.init_sims(replace=True)
    index2word_set = word2vec_model.index2word

    word2vec_model.init_sims(replace=True)

    # Get the average
    X_w2v = np.zeros(shape=(len(y), num_features))
    i = 0
    for sen in doc_word:
        X_w2v[i] = avg_feature_vector(sen, model=word2vec_model, num_features=num_features,
                                      index2word_set=index2word_set)
        i += 1

    #
    #   Plot classification
    #

    plt.figure(figsize=(12, 12))
    lw = 2
    labels = ['Random Forest BOW Mean ROC (area = %0.12f)',
              'Random Forest SVD Mean ROC (area = %0.12f)',
              'Random Forest W2V Mean ROC (area = %0.12f)',
              'SVM BOw Mean ROC (area = %0.12f)',
              'SVM SVD Mean ROC (area = %0.12f)',
              'SVM W2V Mean ROC (area = %0.12f)',
              'Neural Network Mean ROC (area = %0.12f)']

    colors = ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'deeppink']

    metrics_list = ['Accuracy', 'Precision', 'Recall', 'F-Measure', 'AUC']
    fieldnames = ['Statistic Measure', 'SVM (BoW)', 'Random Forest (BoW)', 'SVM (SVD)', 'Random Forest (SVD)',
                  'SVM (W2V)', 'Random Forest (W2V)', 'Neural Networks (SVD)']

    # produce svm and random_forests for classifications
    svm_bow_metrics = svm(X_tfidf, y, classes, n_classes, labels[0], colors[0], plt)
    svm_svd_metrics = svm(X_svd, y, classes, n_classes, labels[1], colors[1], plt)
    svm_w2v_metrics = svm(X_w2v, y, classes, n_classes, labels[2], colors[2], plt)
    random_forest_bow_metrics = random_forest(X_tfidf, y, classes, n_classes, labels[3], colors[3], plt)
    random_forest_svd_metrics = random_forest(X_svd, y, classes, n_classes, labels[4], colors[4], plt)
    random_forest_w2v_metrics = random_forest(X_w2v, y, classes, n_classes, labels[5], colors[5], plt)

    # produce neural networks
    neural_networks_metrics = neural_networks(X_tfidf_stemmer, y, classes, n_classes, labels[6], colors[6], plt)

    with open('EvaluationMetric_10fold.csv', 'w') as csvfile:

        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
        writer.writeheader()
        for key in metrics_list:
            writer.writerow({'Statistic Measure': key,
                             'SVM (BoW)': svm_bow_metrics[key],
                             'Random Forest (BoW)': random_forest_bow_metrics[key],
                             'SVM (SVD)': svm_svd_metrics[key],
                             'Random Forest (SVD)': random_forest_svd_metrics[key],
                             'SVM (W2V)': svm_w2v_metrics[key],
                             'Random Forest (W2V)': random_forest_w2v_metrics[key],
                             'Neural Networks (SVD)': neural_networks_metrics[key]})

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('roc_10fold.png')
