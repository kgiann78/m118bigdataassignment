from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from numpy import mean
from sklearn.preprocessing import label_binarize

data_train = load_files('Dataset/')
tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfidf = tfidf_vect.fit_transform(data_train.data)
y = data_train.target
classes = list(set(y))
n_classes = len(classes)


clf = RandomForestClassifier()
cv = StratifiedKFold(n_splits=10)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'green', 'darkred', 'darkgreen'])
accuracy = []
precision = []
f1 = []
recall = []
for (train, test), color in zip(cv.split(X_tfidf, y), colors):
    clf.fit(X_tfidf[train], y[train])
    predict = clf.predict(X_tfidf[test])
    proba_ = clf.predict_proba(X_tfidf[test])

    accuracy.append(accuracy_score(y[test], predict))
    precision.append(precision_score(y[test], predict, average='macro'))
    f1.append(f1_score(y[test], predict, average='macro'))
    recall.append(recall_score(y[test], predict, average='macro'))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = label_binarize(y[test], classes=classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], proba_[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

print mean(accuracy)
print mean(precision)
print mean(f1)
print mean(recall)