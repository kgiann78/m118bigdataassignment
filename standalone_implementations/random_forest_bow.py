from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from sklearn import preprocessing

zf = zipfile.ZipFile('../Datasets-2016.zip')
files = zf.open('train_set.csv')
df = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content', 'Category'])
df['text'] = df['Title'].map(str)+df['Content']

tfidf_vect = TfidfVectorizer(analyzer='word', stop_words='english')
X_tfidf = tfidf_vect.fit_transform(df['text'][1:])
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Category'][1:])
classes = list(set(y))
n_classes = len(classes)

clf = RandomForestClassifier()
cv = StratifiedKFold(n_splits=10)
colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'green', 'darkred', 'darkgreen'])
accuracy = []
precision = []
f1 = []
recall = []

plt.figure(figsize=(12, 12))
lw = 2
j = 1
mean_tpr_final = 0.0
mean_fpr_final = np.linspace(0, 1, 100)

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

    '''
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    '''
    # Compute macro-average ROC curve and ROC area

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    mean_tpr_final += interp(mean_fpr_final, fpr["macro"], tpr["macro"])
    mean_tpr_final[0] = 0.0
    # Plot all ROC curves
    '''
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average '+str(j)+' fold ROC curve (area = {0:0.12f})'
                   ''.format(roc_auc["micro"]),
             color=color, linestyle=':', linewidth=4)
    '''
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average '+str(j)+' fold ROC curve (area = {0:0.12f})'
                   ''.format(roc_auc["macro"]),
             color=color, linestyle='-.', linewidth=4)
    j += 1

print np.mean(accuracy)
print np.mean(precision)
print np.mean(f1)
print np.mean(recall)

mean_tpr_final /= cv.get_n_splits(X_tfidf, y)
mean_tpr_final[-1] = 1.0
mean_auc = auc(mean_fpr_final, mean_tpr_final)

print mean_auc

plt.plot(mean_fpr_final, mean_tpr_final, color='deeppink', linestyle='-',
         label='Mean ROC (area = %0.12f)' % mean_auc, lw=lw)

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")

plt.savefig('random_forest_bow_roc.png')