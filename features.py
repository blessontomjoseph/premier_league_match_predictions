import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import namedtuple
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class Features:
    def __init__(self, home, away, objects, feature_names, home_data, away_data, data):
        self.home = home
        self.away = away
        self.objects = objects
        self.feature_names = feature_names
        self.home_data1 = home_data.copy()
        self.away_data1 = away_data.copy()
        self.home_data = home_data.copy()
        self.away_data = away_data.copy()
        self.data = data

    def rol_hac(self):
        # rolling home and away combined
        """making a rolling of all attributes representing a team offense"""

        self.away_data1.columns = self.objects + self.home  # changed away to look like home
        combined_data = pd.concat([self.home_data1, self.away_data1],
                                  ignore_index=True)  # cancatted it
        combined_data['identification'] = np.array([np.ones(self.home_data1.shape[0]), np.zeros(
            self.home_data1.shape[0])], dtype=np.int64).reshape(-1, 1)  # making an identifier to split later
        # combined data-very important its data raveled vertically

        rol_hac = pd.DataFrame(index=combined_data.index)
        features_to_rol = self.home[1:]
        for col in features_to_rol:
            feature = combined_data.groupby('hometeam', as_index=False)[col].rolling(
                window=10, center=True, min_periods=5).mean().shift(1).fillna(method='bfill')[col]
            rol_hac[col] = feature
        home_df = rol_hac[combined_data.identification == 1]
        away_df = rol_hac[combined_data.identification ==
                          0].reset_index(drop=True)
        # hr-home rolling but its offensive
        home_df.columns = [i+'_hr'for i in self.feature_names]
        # away rolling but offensive
        away_df.columns = [i+'_ar'for i in self.feature_names]
        # suffixes are for overlapping columns
        rol_features = home_df.join(away_df)
        return rol_features

    def rol_hac_d(self):
        """this features are rolling home and away combined attributes(attributes like home shots,home fouls away yellow acrds etc )
        conceded by home and away teams until the previous match
        hs_c=shots conceded by the  home team until the previous match
        ha_c=shots conceded by the away team until the previous match ,i think you get the idea
        """

        self.home_data.columns = self.objects+self.away  # changed away to look like home
        combined_data = pd.concat([self.away_data, self.home_data],
                                  ignore_index=True)  # cancatted it
        combined_data['identification'] = np.array([np.ones(self.home_data.shape[0]), np.zeros(
            self.home_data.shape[0])], dtype=np.int64).reshape(-1, 1)  # making an identifier to split later
        # ones represent what is home teams defence and zeros represent away teams defence

        rol_hac = pd.DataFrame(index=combined_data.index)
        features_to_rol = self.away[1:]
        for col in features_to_rol:
            feature = combined_data.groupby('awayteam', as_index=False)[col].rolling(
                window=10, center=True, min_periods=5).mean().shift(1).fillna(method='bfill')[col]
            rol_hac[col] = feature  # c indicates conceded
        home_df = rol_hac[combined_data.identification == 1]
        # -chr-conceded home rolling
        home_df.columns = [i+'_chr'for i in self.feature_names]
        away_df = rol_hac[combined_data.identification ==
                          0].reset_index(drop=True)
        # -car-conceded away rolling
        away_df.columns = [i+'_car'for i in self.feature_names]
        rol_features = home_df.join(away_df)
        return rol_features

    def other_features(self, data):
        data['day'] = data['datetime'].apply(lambda x: x.isoweekday())
        data.ftr = data.ftr.map({'H': 2, 'A': 1, 'D': 0})
        data.htr = data.htr.map({'H': 2, 'A': 1, 'D': 0})
        return data

    def execute(self):
        rollin_features_a = self.rol_hac()
        rollin_features_d = self.rol_hac_d()
        object_data = self.data[self.objects+['hometeam', 'awayteam']]
        data = object_data.join([rollin_features_a, rollin_features_d])
        data_all_features = self.other_features(data)
        return data_all_features


def encoder(data, features):
    from sklearn.preprocessing import OrdinalEncoder
    oe = OrdinalEncoder()
    data[features] = oe.fit_transform(data[features])
    return data


def features_targets(data, selected_features=None):
    # data = data.select_dtypes('number')  #selecting only numerical features for starters
    container = namedtuple('container', ['trainx', 'trainy'])
    if selected_features is not None:
        features = data[selected_features]
    else:
        features=data
    # these not rolling original results for reference dont need anymore!
    ind_feats = features.drop(['ftr', 'htr','datetime','referee'], axis=1)
    ind_feats = encoder(ind_feats, ['hometeam', 'awayteam', 'season'])
    datas = container(ind_feats, features['ftr'])
    return datas


def mutual_information(x, y, mask=None):
    """function calculates the mi score in descendinhg trend given x and y"""
    if mask is not None:
        mi = mutual_info_classif(x.iloc[:, :mask], y)
        mi = pd.DataFrame(mi, columns=['mi_score'], index=x.columns[:mask])
    elif mask is None:
        mi = mutual_info_classif(x, y)
        mi = pd.DataFrame(mi, columns=['mi_score'], index=x.columns)

    mi = mi.sort_values("mi_score", ascending=False)
    return mi


def pca_ing(x, standardize=True):
    """function standardizes the data is not standardized and performs pca and outputs its componets in a df also loadings"""
    if standardize:
        sc = StandardScaler()
        x_scaled = sc.fit_transform(x)
        x = pd.DataFrame(x_scaled, columns=x.columns)
    pca = PCA()
    x_pca = pca.fit_transform(x)
    components = [f'pca_{i}' for i in x.columns.values]
    x_pca = pd.DataFrame(x_pca, columns=components)
    loadings = pd.DataFrame(
        pca.components_.T, columns=components, index=x.columns)
    return x_pca, loadings


def auto_best_features(x, y,  n_features, standardize_on_pca=True):
    """best n_features(having most mi scores) among x and its pca version n_features=-1 for all features """
    x_pca, _ = pca_ing(x, standardize=standardize_on_pca)
    x.reset_index(drop=True, inplace=True)
    all_features = x.join(x_pca)
    mutual_info = mutual_information(all_features, y)
    selected_cols = mutual_info.index.values[:n_features]
    return all_features[selected_cols]


def plotmi(mi):
    sns.set_style('darkgrid')
    plt.figure(figsize=(5, 20), dpi=100)
    sns.barplot(mi['mi_score'], mi.index)
