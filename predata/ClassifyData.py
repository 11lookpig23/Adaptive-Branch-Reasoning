import random
import Levenshtein
import re
from pydantic import BaseModel, Field
import json

categories = {
        'cora': ["Rule Learning", "Neural Networks", "Case Based", "Genetic Algorithms", "Theory", "Reinforcement Learning", "Probabilistic Methods"],
        'pubmed': ["Type 1 diabetes", "Type two diabetes", "Experimentally induced diabetes"]
    }


from sklearn import metrics

class predres:
    def __init__(self,predict_result) -> None:
        self.predict_result= predict_result
    def countacc(self):
        correct_predictions = 0
        for key, value in self.predict_result.items():
            if value["tf"]:
                correct_predictions += 1
        accuracy = correct_predictions / len(self.predict_result)
        self.acc = accuracy
        return accuracy

    def cal_metrics(self):
        y_pred = []
        y_true = []
        for key, value in self.predict_result.items():
            y_true.append(categories['cora'].index(value['ideal_answ']))
            try:
                y_pred.append(categories['cora'].index(value['ori_answ']))
            except:
                y_pred.append(random.randint(1,6))
        f1_mac = metrics.f1_score(y_true, y_pred, average='macro')
        f1_wei = metrics.f1_score(y_true, y_pred, average='weighted')
        acc = metrics.accuracy_score(y_true, y_pred)
        return {"f1_m":f1_mac,"f1_wei":f1_wei,"acc":acc}

    def get_pred(self,li):
        pred_val = [self.predict_result[str(k)]["tf"] for k in li]
        wrval = [not n for n in pred_val]
        return wrval
    def save(self,folder,fname):
        with open(folder+fname, 'w') as file:
            json.dump(self.predict_result,file)

class dataClassif:
    def __init__(self,data,text,nodes,dataname):
        self.data = data
        self.text = text
        self.nodes = nodes
        self.dataname = dataname
        if dataname=='cora':
            self.nclass = 7
        elif dataname=='pubmed':
            self.nclass = 3

    def sample_citanode(self,retr):
        exampdic = {}
        for i,id_ in enumerate(self.nodes):
            citenei = retr.get_hops(id_)
            exms = retr.select_citations(citenei,4,False)
            exampdic[id_] = exms
        return exampdic

    def cal_metrics(self,predict_result):
        y_pred = []
        y_true = []
        for key, value in predict_result.items():
            y_true.append(categories['cora'].index(value['ideal_answ']))
            try:
                y_pred.append(categories['cora'].index(value['ori_answ']))
            except:
                y_pred.append(random.randint(0,6))
        f1_mac = metrics.f1_score(y_true, y_pred, average='macro')
        f1_wei = metrics.f1_score(y_true, y_pred, average='weighted')
        acc = metrics.accuracy_score(y_true, y_pred)
        return {"f1_m":f1_mac,"f1_wei":f1_wei,"acc":acc}
    def has_label(self,idx):
        return (self.data.train_mask[idx] or self.data.val_mask[idx])
    
    def get_data(self,idx,attr):
        if attr=='title':
            return self.text['title'][idx]
        elif attr=='abs':
            return self.text['abs'][idx]
        elif attr=='label':
            return self.text['label'][idx]
        else:
            return 'NONE'

    def set_id(self,idx):
        self.title = self.text['title'][idx]
        self.abs = self.text['abs'][idx]
        self.label = self.text['label'][idx]

