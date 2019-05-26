"""Action"""

from sklearn import metrics
#import numpy as np


def get_nmi(y_ture, y_pred):
    """Get normalized_mutual_info_score
    """
    return metrics.normalized_mutual_info_score(y_ture, y_pred)

def get_ari(y_true, y_pred):
    """Get metrics.adjusted_rand_score
    """
    return metrics.adjusted_rand_score(y_true, y_pred)

#def get_acc(y_true, y_pred):
#    """Get accuracy
#    """
