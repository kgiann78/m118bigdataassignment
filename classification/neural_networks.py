from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, f1_score, recall_score
from itertools import cycle
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import label_binarize
from sklearn.neural_network import MLPClassifier
import numpy as np
from scipy import interp


def neural_networks(X_tfidf, y, classes, n_classes, label, color_plot, plt):
    model = TruncatedSVD(n_components=100).fit(X_tfidf)
    X_svd = model.transform(X_tfidf)

    clf = MLPClassifier(solver='adam', max_iter=300, activation='logistic',
                        alpha=1e-5, hidden_layer_sizes=(50,), warm_start=True, random_state=1)

    cv = StratifiedKFold(n_splits=10)
    colors = cycle(
        ['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange', 'red', 'green', 'darkred', 'darkgreen'])
    accuracy = []
    precision = []
    f1 = []
    recall = []

    lw = 2
    j = 1
    mean_tpr_final = 0.0
    mean_fpr_final = np.linspace(0, 1, 100)
    for (train, test), color in zip(cv.split(X_svd, y), colors):
        clf.fit(X_svd[train], y[train])
        predict = clf.predict(X_svd[test])
        proba_ = clf.predict_proba(X_svd[test])

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

        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        mean_tpr_final += interp(mean_fpr_final, fpr["macro"], tpr["macro"])
        mean_tpr_final[0] = 0.0

        j += 1

    mean_tpr_final /= cv.get_n_splits(X_svd, y)
    mean_tpr_final[-1] = 1.0
    mean_auc = auc(mean_fpr_final, mean_tpr_final)

    plt.plot(mean_fpr_final, mean_tpr_final, color=color_plot, linestyle='-',
             label=label % mean_auc, lw=lw)

    return {'Accuracy': np.mean(accuracy), 'Precision': np.mean(precision), 'Recall': np.mean(recall),
            'F-Measure': np.mean(f1), 'AUC': mean_auc}
