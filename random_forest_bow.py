from numpy import mean
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from numpy import mean

data_train = load_files('Dataset/')
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfidf = tfidf_vect.fit_transform(data_train.data)
y = data_train.target

# Text management (experimenting on elements gathered from vectorizing the given text)
# #print vector shape
# print X_tfidf.shape
# #print vector for each doc
# print X_tfidf
# #print vector for each doc
# print data_train.target


clf = RandomForestClassifier() #TODO na paikse ligo me tis parmetrous
cv = StratifiedKFold(n_splits=10)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'green', 'darkred', 'darkgreen'])
accuracy = []
precision = []
f1 = []
recall = []
for (train, test), color in zip(cv.split(X_tfidf, y), colors):
    clf.fit(X_tfidf[train], y[train])
    predict =clf.predict(X_tfidf[test])
    # Compute ROC curve and area the curve
    accuracy.append(metrics.accuracy_score(y[test], predict))
    precision.append(metrics.precision_score(y[test], predict, average='macro')) #TODO na dw ti paizei me to average....!!!!
    f1.append(metrics.f1_score(y[test], predict, average='macro'))
    recall.append(metrics.recall_score(y[test], predict, average='macro'))

print mean(accuracy)
print mean(precision)
print mean(f1)
print mean(recall)

#TODO
'''
y = label_binarize(y, classes=[0, 1, 2, 3])
print metrics.roc_auc_score(target, predicted, average='samples')
'''