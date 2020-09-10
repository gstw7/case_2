import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, precision_score, classification_report
from sklearn.dummy import DummyClassifier


def eval_thresh(y_real, y_proba):
    """
    Essa função gera um dataframe com vários thresholds e as respectivas métricas de acuráccia: precision, recall e f1
    """
    recall_score_thresh = []
    precision_score_thresh = []
    f1_score_thresh = []
    for thresh in np.arange(0,1,0.001):
        y_thresh = [1 if x >= thresh  else 0 for x in y_proba ]
        recall_score_thresh.append(recall_score(y_real, y_thresh))
        precision_score_thresh.append(precision_score(y_real, y_thresh))
        f1_score_thresh.append(f1_score(y_real, y_thresh))
    dict_metrics = {'threshold':np.arange(0,1,0.001),'recall_score':recall_score_thresh,\
                    'precision_score':precision_score_thresh,'f1_score':f1_score_thresh}
    df_metrics = pd.DataFrame(dict_metrics)
    return df_metrics 

def plot_metrics(df):

    """
    Essa função plota as métricas que estão no dataframe df (métricas vs thresholds)
    """
    plt.plot(df['threshold'],df['recall_score'], '-.')
    plt.plot(df['threshold'],df['precision_score'], '-.')
    plt.plot(df['threshold'],df['f1_score'],'-.')
    plt.legend(['recall','precision','f1_score'])
    plt.xlabel("Threshold")
    plt.ylabel("Metric")
    plt.show()