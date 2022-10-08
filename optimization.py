from time import time
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import make_scorer
from xgboost import XGBClassifier, DMatrix
from skopt.callbacks import DeadlineStopper, DeltaYStopper
from skopt.space import Real, Categorical, Integer
from functools import partial
import pprint
import pandas as pd
import tqdm

from skopt import BayesSearchCV


model = XGBClassifier(random_state=0, booster='gbtree', objective='multi:softprob', tree_method='hist',
                      eval_metric=accuracy_score, verbosity=0, num_class=3, grow_policy='lossguide')

# eval_metric='mlogloss' checkout


scoring = make_scorer(partial(accuracy_score), greater_is_better=True)

overdone_control = DeltaYStopper(delta=0.0001)
time_limit_control = DeadlineStopper(total_time=60*60*1)

search_spaces = {'learning_rate': Real(0.01, 1.0, 'uniform'),
                 'max_depth': Integer(1, 12),
                 'subsample': Real(0.1, 1.0, 'uniform'),
                 'colsample_bytree': Real(0.1, 1.0, 'uniform'),
                  'reg_lambda': Real(1e-9, 300., 'uniform'),
                #   'reg_alpha': Real(1e-9, 100., 'uniform'),  # L1 regularization not needed making results shite
                 'n_estimators': Integer(1, 500),
                #  'num_class':Categorical([3]),
                 }

# num_iterations=1000,
# max_depth=10,
# feature_fraction=0.7,
# scale_pos_weight=1.5,


def optimizer(trainx, trainy, title, callbacks=True):

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=0) #controllable split size-take care
    cv_strategy = skf.split(trainx, trainy)
    optimizer = BayesSearchCV(estimator=model, search_spaces=search_spaces, scoring=scoring, cv=cv_strategy, n_iter=120,
                              n_points=1, n_jobs=1, iid=False, return_train_score=False, refit=False, optimizer_kwargs={'base_estimator': 'GP'}, random_state=0)

    # params=optimizer.get_params()
    # params['num-class']=3

    
    start = time()
    if callbacks:
        optimizer.fit(trainx, trainy, callback=[overdone_control, time_limit_control])
    else:
        optimizer.fit(trainx, trainy)

    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f " + u"\u00B1" +
          " %.3f") % (time() - start, len(optimizer.cv_results_['params']), best_score, best_score_std))
    print('Best parameters:')
    pprint.pprint(best_params)
    return best_params
