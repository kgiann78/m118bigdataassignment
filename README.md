# README #

M118 Big Data Assignment

Python Version: 2.7\
Scikit Learn version: 0.18\
Gensim Version: 0.13.3\
WordCloud Version: 1.2.1

__Available Modules__

All algorithms use the path of Dataset.zip, which contains 2 csv files:
* train_set.csv: Format [RowNum Id Title Content Category] (tab seperated)
* test_set.csv: Format [RowNum Id Title Content] (tab seperated)

<!-- -->
1. WordCloud
2. Three types of inputs - BoW(Tf-idf), SVD, Average Word Vector
3. Classification (SVM, Random Forest, Neural Network)
4. Standalone classification implementations (10-fold CV)
5. Benchmarking all implementations for all inputs (10-fold Cross Validation)
6. KMeans Map Reduce Clustering\
   Preprocess: Remove stop words, stemming\
   Input: SVD vectors from tf-idf Vectors of the documents\
   First Step: Map Reduce K-Means for a random sample, 30% of actual dataset, to get more suitable centroids\
   Second Step: Map Reduce K-Means for full dataset\
   spark-submit kmeans_mapreduce.py (spark folder @ PATH)
7. test_categories.py - Categorize new documents\
   Preprocess: Remove stop words, stemming\
   Input Data: SVD vectors from Tf-idf Vectors of the documents\
   Train Algorithm: Neural Networks