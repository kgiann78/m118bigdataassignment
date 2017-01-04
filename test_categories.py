import zipfile
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neural_network import MLPClassifier
import csv
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from string import punctuation
from sklearn.datasets import load_files

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
files.readline()
test_document_id = []
test_document_text = []
for f in files.readlines():
    text = f.split('\t')
    test_document_id.append(text[1])
    test_document_text.append(text[2]+'\n'+text[3])


data_train = load_files('Dataset/')
tfidf_vect = TfidfVectorizer(tokenizer=tokenize, analyzer='word', stop_words='english')
tfidf_model = tfidf_vect.fit(data_train.data+test_document_text)
y = data_train.target

X_all = tfidf_model.transform(data_train.data+test_document_text)

X_tfidf = tfidf_model.transform(data_train.data)

X_test = tfidf_model.transform(test_document_text)


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
    i = 0
    for p in predict:
        writer.writerow({'Test_Document_ID': test_document_id[i],
                         'Predicted_Category': data_train.target_names[p][:-1]})
        i += 1
