from gensim.models import Word2Vec
from sklearn.model_selection import StratifiedKFold
from itertools import cycle
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy import interp
import matplotlib.pyplot as plt
import pandas as pd
import zipfile
from sklearn import preprocessing
import string
import time


def avg_feature_vector(words, model, num_features, index2word_set):
    # function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0

    # list containing names of words in the vocabulary
    # index2word_set = set(model.index2word) this is moved as input param for performance reasons
    for word in words:
        if word in index2word_set:
            nwords += 1
            featureVec = np.add(featureVec, model[word])

    if (nwords > 0):
        featureVec = np.divide(featureVec, nwords)
    return featureVec


zf = zipfile.ZipFile('../Datasets-2016.zip')
files = zf.open('train_set.csv')
df = pd.read_csv(files, sep='\t', names=['RowNum', 'Id', 'Title', 'Content', 'Category'])
df['text'] = df['Title'].map(str)+df['Content']
start = int(round(time.time() * 1000))
le = preprocessing.LabelEncoder()
y = le.fit_transform(df['Category'][1:])
classes = list(set(y))
n_classes = len(classes)

doc_word = []
for i in range(1, df['text'].count()):
    doc_word.append(str(df['text'][i]).translate(None, string.punctuation).split())

num_features = 100    # Word vector dimensionality
min_word_count = 50   # Minimum word count
num_workers = 20       # Number of threads to run in parallel
context = 300          # Context window size
downsampling = 1e-2  # Downsample setting for frequent words

word2vec_model = Word2Vec(doc_word, workers=num_workers,
            size=num_features, min_count=min_word_count,
            window=context)
word2vec_model.init_sims(replace=True)
index2word_set = word2vec_model.index2word


# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
word2vec_model.init_sims(replace=True)

X_w2v = np.zeros(shape=(len(y), num_features))
i = 0
for sen in doc_word:
    X_w2v[i] = avg_feature_vector(sen, model=word2vec_model, num_features=num_features, index2word_set=index2word_set)
    i += 1

clf = RandomForestClassifier(warm_start=True)
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

for (train, test), color in zip(cv.split(X_w2v, y), colors):
    clf.fit(X_w2v[train], y[train])
    predict = clf.predict(X_w2v[test])
    proba_ = clf.predict_proba(X_w2v[test])

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

mean_tpr_final /= cv.get_n_splits(X_w2v, y)
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

plt.savefig('random_forest_w2v_roc.png')
end = int(round(time.time() * 1000))

print "Execution Time: "+str(end-start)