import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics


def compute_roc(gc_label, gc_predict, draw):

    # gc_predict = normalization(gc_predict)
    gc_label = gc_label.flatten().astype(float)
    gc_predict = gc_predict.flatten().astype(float)
    if draw:
        score = draw_roc_curve(gc_label, gc_predict / gc_predict.max())
    else:
        score = metrics.roc_auc_score(gc_label, gc_predict / gc_predict.max())
    return score


def normalization(GC):
    max_values = np.max(GC, axis=1, keepdims=True)
    GC = GC / max_values
    return GC


def draw_roc_curve(label, predict):
    FPR, TPR, P = metrics.roc_curve(label, predict)
    plt.plot(FPR, TPR, 'b*-', label='roc')
    plt.plot([0, 1], [0, 1], 'r--', label="45°")
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    AUC_score = metrics.auc(FPR, TPR)
    return AUC_score
