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
            # 如果最后一个元素为True，表示预测正确
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
    def sample_testnode(self,n):
        random.seed(42)
        if n>=1000:
            return self.nodes
        else:
            test_node = random.sample(self.nodes,n)
            return test_node


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
    
    def extract_category_cora(self,response,categories,keywd="Category"):
        # 查找所有的"FINAL_SCORE"并记录位置
        matches = list(re.finditer(r'Category', response, re.IGNORECASE))
        # 如果找到了"FINAL_SCORE"，获取最后一个匹配的位置
        if matches:
            last_match = matches[-1]  # 获取最后一个匹配
            start_index = last_match.end()  # 获取匹配结束的位置            
            last_section0 = response[start_index:].strip()  # 去掉前后空格
            # 使用 split 方法获取需要的字符串，直到换行符或文本结束
            last_section_parts = last_section0.split('\n', 1)  # 只分割一次
            last_section = last_section_parts[0].strip()  # 获取第一个部分并去掉空格
            # 从提取的文本中找到最后一个数字
            last_found = "None"
            recid = -1
            for i,category in enumerate(categories):
                if category.lower() in last_section.lower():
                    last_found = category
                    recid = i
            if recid!=-1:
                return categories[recid]
        return last_found
    def extract_category(self,output_text, categories):
        # 使用正则表达式找到所有符合模式的部分
        pattern = r"(?i)\bcategory\s*:\s*([a-zA-Z ]+)"
        matches = list(re.finditer(pattern, output_text))

        if matches:
            # 倒序迭代匹配项，优先处理最后一个匹配项
            for match in reversed(matches):
                extracted_category = match.group(1).strip()
                for category in categories:
                    if Levenshtein.distance(extracted_category, category) <= 2:
                        return category
        return "None"
    '''
    def extract_category_med(self,output_text, categories):
        # 使用正则表达式找到所有符合模式的部分
        pattern = r"(?i)\bcategory\s*:\s*\{?\s*([a-zA-Z0-9 ]+)\s*\}?"
        matches = list(re.finditer(pattern, output_text))
        #print(matches)
        if matches:
            # 倒序迭代匹配项，优先处理最后一个匹配项
            for match in reversed(matches):
                extracted_category = match.group(1).strip()
                extracted_category = extracted_category.replace("2","two")
                for category in categories:
                    if Levenshtein.distance(extracted_category.lower(), category.lower()) <= 2:
                        return category
        return "None"
    '''
    def extract_category_med(self,output_text, categories):
        # 匹配的关键词列表，忽略大小写
        keywords = [
        "Type 1 diabetes", "Type one diabetes", "Type two diabetes", "Type 2 diabetes",
        "Experimentally induced diabetes"
    ]
        # 构建正则表达式进行匹配，忽略大小写
        pattern = re.compile(r'|'.join(re.escape(keyword) for keyword in keywords), re.IGNORECASE)
        
        # 查找所有匹配的字符串
        matches = pattern.findall(output_text)
        if matches:
            rt = matches[-1]
            rt = rt.replace("one","1")
            rt = rt.replace("2","two")

        else:
            rt = "None"
        
        # 返回最后一个匹配的字符串，如果有匹配则输出最后一个，否则输出None
        return rt
    def verify(self,response):
        ideal_answer = self.label
        print("Ideal_answer:", ideal_answer, end="\n")
        if self.dataname=='cora':
            pred = self.extract_category_cora(response,categories[self.dataname])
        elif self.dataname=='pubmed':
            pred = self.extract_category_med(response,categories[self.dataname])
        print("rpredict~~~~",pred,"\n")
        distance = Levenshtein.distance(pred.lower(), ideal_answer.lower())
        if distance<=2:
            tf=True
        else:
            tf=False
        print("Is prediction correct? ",tf, end="\n---------\n")
        return (pred,ideal_answer,tf)
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

