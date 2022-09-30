import pandas as pd
import numpy as np
from collections import namedtuple

path = "epl_23-f.csv"
df = pd.read_csv(path, encoding='windows-1254')

def preprocess(data):
    data.columns=[col.lower() for col in data.columns]
    data.datetime = pd.to_datetime(data.datetime).dt.date
    data.drop(data.loc[data['season'].str.contains('1993|1994|1995|1996|1997|1998|1999',regex=True)].index,inplace=True)
    data.sort_values(['datetime','hometeam','awayteam','referee'],ascending=True,inplace=True) 
    data.reset_index(drop=True,inplace=True)
    points_h_map = {'H': 3, 'D': 1, 'A': 0}
    points_a_map = {'H': 0, 'D': 1, 'A': 3}
    data['hp'] = data['ftr'].map(points_h_map)
    data['ap'] = data['ftr'].map(points_a_map)
    return data


pack={}
home=['hometeam','fthg','hthg','hs','hst','hc','hf','hy','hr','hp'] 
away=['awayteam','ftag','htag','as','ast','ac','af','ay','ar','ap']
objects =['season', 'datetime', 'ftr', 'htr', 'referee']
feature_names = ['ftg', 'htg', 's', 'st', 'c', 'f', 'y', 'r', 'p']

data=preprocess(df)
data = data.reindex(columns=objects+home+away)

home_data=data[objects+home]
away_data=data[objects+away]

pack['data']=data
pack['home_data']=home_data
pack['away_data']=away_data
pack['home'] = home
pack['away']=away
pack['objects']=objects
pack['feature_names']=feature_names



