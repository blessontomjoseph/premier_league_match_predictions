import scipy
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


def isolation_forest(data, iforest):
    # iforest in an instance of isolation_forest
    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data)
    outlier_index = np.where(pred == -1)
    outlier_values = data.iloc[outlier_index]
    return outlier_index, outlier_values


def iqr(data, features):
    all = np.array([])
    for feature in features:
        sub_data = data[feature]
        q1 = np.quantile(sub_data, .25)
        q3 = np.quantile(sub_data, .75)
        iqr = q3-q1
        r_1 = q1-1.5*iqr
        r_2 = q3+1.5*iqr
        all = np.append(all, sub_data[sub_data > r_2].index.values)
        all = np.append(all, sub_data[sub_data > r_2].index.values)
    df = pd.DataFrame(all.reshape(-1, 1), columns=['id'])
    index = df['id'].unique()
    return index



def lof(data,lof):
    pred = lof.fit_predict(data)
    outlier_index = np.where(pred == -1)
    outlier_values = data.iloc[outlier_index]
    return outlier_index,outlier_values