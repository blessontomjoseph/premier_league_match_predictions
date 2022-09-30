from hyperopt import fmin,hp,tpe,Trials
from hyperopt.pyll.base import scope
from functools import partial
from sklearn.model_selection import KFold,cross_val_score
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


# parameters for tuning in models in model.py
params={'n_estimators':scope.int(hp.quniform('n_estimators', 10, 100, 1)), 
               'max_depth':scope.int(hp.quniform('max_depth', 1, 50, 1)),
               'min_samples_split':scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
               'min_samples_leaf':scope.int(hp.quniform('min_samples_leaf', 2, 20, 1)),
               'max_features':hp.quniform('max_features', 0.1, 1, 0.1)}


class bayes_estimation:
    def  __init__(self,params,trainx,trainy,valx,valy,model):
        self.params=params
        self.trainx=trainx
        self.trainy=trainy
        self.valx=valx
        self.valy=valy
        self.model=model
      
    @staticmethod
    def objective(params,trainx,trainy,valx,valy,model):
        model[-1].set_params(**params) 
        model.fit(trainx,trainy) 
        preds=model.predict(valx)
        out=accuracy_score(preds,valy)
        return out
    
    def parameters(self):
        obj_func=partial(self.objective, trainx=self.trainx, trainy=self.trainy, valx=self.valx, valy=self.valy, model=self.model)
        trials=Trials()
        result = fmin(fn=obj_func,
                    space=self.params,
                    algo=tpe.suggest,
                    trials=trials,
                    max_evals=10
                        )
        
        return result
    
def format_result(result):
    """formats result paraameters from bayesian search into any desired form- usually a change in data type"""
    result['n_estimators'] = int(result['n_estimators'])
    result['min_samples_split'] = int(result['min_samples_split'])
    result['min_samples_leaf'] = int(result['min_samples_leaf'])
    result['max_depth'] = int(result['max_depth'])
    return result
