import sys
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import ndcg_score
import pickle
import json
class PromxAgent:
    def __init__(self,pstdata,dataname):
        # read YAML
        #with open(parafile, 'r') as file:
        #    parameters = yaml.safe_load(file)
        self.test = 3
        self.abs_len = 5000
        self.pstdata = pstdata
        self.main_papers = pstdata.main_papers
        self.dataname = dataname
        self.node_index_list = pstdata.node_index_list
    def loadjson(self,path):
        with open(path, 'r') as f:
            fi = json.load(f)
        return fi
    def logger(self,experName):
        self.experName = experName
        self.f = open('logresult-Promx/'+"PTS"+"__"+self.experName+'.log', 'a')
        sys.stdout = self.f
        print("experName .... ",self.experName)

    def compute_mAP(self,labels,outputs):
        y_true = np.asarray(labels)
        y_pred = np.asarray(outputs)
        AP = []
        for i in range(y_true.shape[0]):
            AP.append(average_precision_score(y_true[i],y_pred[i]))
        return np.mean(AP)

    def compute_ndcg(self,labels,outputs):
        y_true = np.asarray(labels)
        y_pred = np.asarray(outputs)
        return ndcg_score(y_true,y_pred)
    
    def dumppkl(self,fname,res,folder = 'predictPTS/'):
        with open(folder+fname, 'wb') as file:
            pickle.dump(res, file,pickle.HIGHEST_PROTOCOL)
    def dumpjson(self,fname,res,folder = 'predictPTS/'):
        with open(folder+fname,"w") as f:
            json.dump(res,f)