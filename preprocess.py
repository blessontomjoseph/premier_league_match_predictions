import pandas as pd
import numpy as np
from collections import namedtuple
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

path = "epl_23-f.csv"
df = pd.read_csv(path, encoding='windows-1254')


def isolation_forest(data, iforest):
    # iforest in an instance of isolation_forest
    # Returns 1 of inliers, -1 for outliers
    pred = iforest.fit_predict(data)
    outlier_index = np.where(pred == -1)
    outlier_values = data.iloc[outlier_index]
    return outlier_index, outlier_values


iforest = IsolationForest(n_estimators=100, max_samples='auto',
                          contamination=0.1, max_features=1.0,
                          bootstrap=False, n_jobs=-1, random_state=1)


def preprocess(data):
    data.columns = [col.lower() for col in data.columns]
    data.datetime = pd.to_datetime(data.datetime).dt.date
    data.drop(data.loc[data['season'].str.contains(
        '1993|1994|1995|1996|1997|1998|1999', regex=True)].index, inplace=True)
    data.sort_values(['datetime', 'hometeam', 'awayteam',
                     'referee'], ascending=True, inplace=True)
    data.reset_index(drop=True, inplace=True)
    points_h_map = {'H': 3, 'D': 1, 'A': 0}
    points_a_map = {'H': 0, 'D': 1, 'A': 3}
    data['hp'] = data['ftr'].map(points_h_map)
    data['ap'] = data['ftr'].map(points_a_map)
    data['hhp'] = data['htr'].map(points_h_map)
    data['hap'] = data['htr'].map(points_a_map)
    index, values = isolation_forest(data.select_dtypes('number'), iforest)
    data.drop(index[0], axis=0, inplace=True)
    return data


pack = {}
home = ['hometeam', 'fthg', 'hthg', 'hs',
        'hst', 'hc', 'hf', 'hy', 'hr', 'hp', 'hhp']
away = ['awayteam', 'ftag', 'htag', 'as',
        'ast', 'ac', 'af', 'ay', 'ar', 'ap', 'hap']
objects = ['season', 'datetime', 'ftr', 'htr', 'referee']
feature_names = ['ftg', 'htg', 's', 'st', 'c', 'f', 'y', 'r', 'p', 'hp']

data = preprocess(df)
data = data.reindex(columns=objects+home+away)

home_data = data[objects+home]
away_data = data[objects+away]

pack['data'] = data
pack['home_data'] = home_data
pack['away_data'] = away_data
pack['home'] = home
pack['away'] = away
pack['objects'] = objects
pack['feature_names'] = feature_names
