import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
import csv
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.datasets import load_files
import pandas as pd
import zipfile
from sklearn import preprocessing

stemmer = PorterStemmer()


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = word_tokenize(text)
    tokens = [i for i in tokens if i not in punctuation]
    stems = stem_tokens(tokens, stemmer)
    return stems


zf = zipfile.ZipFile('Datasets-2016.zip')
files = zf.open('test_set.csv')
dataframe = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content'])
dataframe['test'] = dataframe['Title'].map(str)+dataframe['Content']


files = zf.open('train_set.csv')
df = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content', 'Category'])
df['text'] = df['Title'].map(str)+df['Content']


tfidf_vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english')
tfidf_model = tfidf_vect.fit(df['text'][1:].tolist()+dataframe['test'][1:].tolist())
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Category'][1:])
classes = list(set(y))
n_classes = len(classes)

X_all = tfidf_model.transform(df['text'][1:].tolist()+dataframe['test'][1:].tolist())

X_tfidf = tfidf_model.transform(df['text'][1:].tolist())

X_test = tfidf_model.transform(dataframe['test'][1:].tolist())


model = TruncatedSVD(n_components=100).fit(X_all)
X_svd = model.transform(X_tfidf)

X_train = X_svd

X_test = model.transform(X_test)


clf = MLPClassifier(solver='adam', max_iter=300, activation='logistic',
                    alpha=1e-5, hidden_layer_sizes=(50, ), warm_start=True, random_state=1)

clf.fit(X_train, y)
predict = clf.predict(X_test)


fieldnames = ['Test_Document_ID', 'Predicted_Category']
with open('testSet_categories.csv', 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter='\t')
    writer.writeheader()
    i = 1
    for p in le.inverse_transform(predict):
        writer.writerow({'Test_Document_ID': dataframe['Id'][i],
                         'Predicted_Category': p})
        i += 1
