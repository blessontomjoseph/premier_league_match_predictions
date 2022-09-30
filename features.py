import numpy as np
import pandas as pd

class Features:
    def __init__(self,home,away,objects,feature_names,home_data,away_data,data):
        self.home=home
        self.away=away
        self.objects=objects
        self.feature_names=feature_names
        self.home_data1=home_data.copy()
        self.away_data1=away_data.copy()
        self.home_data = home_data.copy()
        self.away_data = away_data.copy()
        self.data=data
        

    def rol_hac(self):
        # rolling home and away combined
        """making a rolling of all attributes representing a team offense"""

        self.away_data1.columns = self.objects+self.home  # changed away to look like home
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
        away_df = rol_hac[combined_data.identification == 0].reset_index(drop=True)
        # hr-home rolling but its offencive
        home_df.columns = [i+'_hr'for i in self.feature_names]
        # away rolling but offencive
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
        away_df = rol_hac[combined_data.identification == 0].reset_index(drop=True)
        # -car-conceded away rolling
        away_df.columns = [i+'_car'for i in self.feature_names]
        rol_features = home_df.join(away_df)
        return rol_features



    def other_features(self,data):
        data['day']=data['datetime'].apply(lambda x: x.isoweekday())
        return data

    def execute(self):
        rollin_features_a=self.rol_hac()
        rollin_features_d=self.rol_hac_d()
        object_data=self.data[self.objects+['hometeam','awayteam']]
        data=object_data.join([rollin_features_a,rollin_features_d])
        data_all_features=self.other_features(data)
        return data_all_features    
