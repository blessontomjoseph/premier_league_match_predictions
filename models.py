from sklearn import metrics,ensemble
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd



pipeline_rfclf=Pipeline([('model',ensemble.RandomForestClassifier())])


class Train:
    def __init__(self,trainx,trainy,valx,testx,s_tr,s_vl,s_ts,model,model_parameters): 
       
        self.trainx=trainx
        self.valx=valx
        self.testx=testx
        self.trainy=trainy
        self.s_tr=s_tr
        self.s_vl=s_vl
        self.s_ts=s_ts
        self.model=model[-1].set_params(**model_parameters)
        
    def model_eval(self):
        self.model.fit(self.trainx,self.trainy)        
        train_preds=self.model.predict(self.trainx)
        val_preds=self.model.predict(self.valx)
        test_preds=self.model.predict(self.testx)
        
        train_acc=self.single_out(train_preds,self.trainy)
        val_acc=self.single_out(val_preds,self.valy)
        test_acc=self.single_out(test_preds,self.testy)        
        return train_acc,val_acc,test_acc
    
    @staticmethod
    def single_out(preds,real):
        return metrics.accuracy_score(real,preds)

    
        