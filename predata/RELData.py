
import pandas as pd
import random
import numpy as np
import dill
import bert_score
from txtai import Embeddings
import re
relsim = pd.read_csv("dataset/relish/relish-similarity.csv")
relsim = relsim.rename(columns={'Unnamed: 0': 'id_'})
relishdata = pd.read_csv('dataset/relish/relish.csv')

class RelishData:
    def __init__(self,load):
        random.seed(42)
        if load:
            with open('dataset/relishdata.pkl', "rb") as file:
                self.relish = dill.load(file)
            return
        self.embeddings = Embeddings(path="allenai/scibert_scivocab_uncased")
        query_ids = relishdata['query_id'].drop_duplicates().tolist()
        random.shuffle(query_ids)
        self.data_all = query_ids
        self.data_test = query_ids[:150]
        self.data_train = query_ids[150:]
        self.main_papers = {}
        for pid in self.data_train:
            per = relish_paper(pid,'train',relishdata)
            self.main_papers[pid] = per
        for pid in self.data_test:
            per = relish_paper(pid,'test',relishdata)
            self.main_papers[pid] = per
        self.node_index_list = self.data_test
        self.nclass = 3
        self.save()

    def add_info(self,num_cal):
        self.nclass = 3
        
    def set_id(self,idx,label):
        self.thistabs = self.main_papers[idx].tabs
        self.label = label

    def find_interval(self,x):
        if 0 <= x < (2 / 3):
            return 0  # 第一个区间
        elif (2 / 3) <= x < (4 / 3):
            return 1  # 第二个区间
        elif (4 / 3) <= x <= 2:
            return 2  # 第三个区间
        else:
            return None  # 不在区间 [0, 2] 内
    def verify(self,response):
        score = self.label
        print("Ideal_score:", score, end="\n")
        txt,pred = self.extract_category(response)
        if pred == "None":
            print("No prediction found ... ",pred,txt,response)
            return ("None",score,False)
        try:
            pred1 = self.find_interval(float(pred))
        except Exception as e:
            print("No prediction found ... ",pred,txt,response)
            return ("None",score,False)
        try:
            tf = (int(pred1) == int(score))
        except:
            print("pred,score NonError ......., ",str(pred1),str(score))
            tf = (str(pred1) == str(score))
        print("prediction is ", pred,". Is prediction correct? ",tf, end="\n---------\n")
        return (pred,score,tf)

    def extract_category(self,response):
        txt1, pred1 = self.extract_single(response,'FINAL_SCORE')
        txt2, pred2 = self.extract_single(response,'Calculate')
        if pred1 == 'None' and pred2 == 'None':
            return txt1,"None"#str(random.randint(0, 2))  # 返回0, 1, 2的随机数作为字符串
        elif pred2 == 'None':
            return txt1,pred1  # 返回b
        elif pred1 == 'None':
            return txt2,pred2  # 返回a
        else:
            if pred1!=pred2:
                return txt2, pred2
            else:
                return txt1, pred1


    def extract_single(self,response,keywd):
        if keywd == 'FINAL_SCORE':
        # 查找所有的"FINAL_SCORE"并记录位置
            matches = list(re.finditer(r'FINAL_SCORE', response, re.IGNORECASE))
        else:
            matches = list(re.finditer(r'Calculate', response, re.IGNORECASE))

        # 如果找到了"FINAL_SCORE"，获取最后一个匹配的位置
        if matches:
            last_match = matches[-1]  # 获取最后一个匹配
            start_index = last_match.end()  # 获取匹配结束的位置            
            last_section0 = response[start_index:].strip()  # 去掉前后空格
            # 使用 split 方法获取需要的字符串，直到换行符或文本结束
            last_section_parts = last_section0.split('\n', 1)  # 只分割一次
            last_section = last_section_parts[0].strip()  # 获取第一个部分并去掉空格
            # 从提取的文本中找到最后一个数字
            last_number = re.findall(r'\b\d+\.\d+|\b\d+\b', last_section)  # 匹配有效的整数和小数
            final_number = last_number[-1] if last_number else "None"  # 获取最后一个数字
            
            return last_section, final_number
        return "None", "None"
    def save(self):
        with open('dataset/relishdata.pkl', 'wb') as f:
            dill.dump(self, f)

    def find_all_candirefs(self):
        candirefs = {}
        for mid in self.data_test:
            best_candi = self.find_candidate(mid)
            candi_ref = self.find_ref_of_candi(best_candi)
            candirefs[mid] = candi_ref
        self.candirefs = candirefs
        self.save()
        return candirefs
    def get_best_candi(self,bestcandi):
        if len(bestcandi)!=0:
            self.best_candi = bestcandi
        else:
            test2candi = {}
            for mid in self.data_test:
                best_candi = self.find_candidate(mid)
                test2candi[mid] = best_candi
            self.best_candi = test2candi
        return self.best_candi
    def find_candidate(self,mid):
        #mid = 22569528
        mids = str(mid)
        sorted_df = relsim.sort_values(by=mids, ascending=False).head(15)[['id_',mids]]
        sorted_df2 = sorted_df[sorted_df['id_'].isin(self.data_train)]
        candidates = []
        candi_id = []
        maintxt = self.main_papers[mid].tabs
        for j in range(len(sorted_df2)):
            pid = sorted_df2['id_'].iloc[j]
            tiabs = self.main_papers[pid].tabs
            candidates.append(tiabs)
            candi_id.append(pid)
        P, R, F1 = bert_score.score(candidates, [maintxt]*len(candidates), lang="en", verbose=True)
        #print("R",R)
        #R = bertscore(candidates, [maintxt]*len(candidates))['recall']
        max_idx = R.argmax()
        best_candi = candi_id[max_idx]
        return best_candi

    def find_ref_of_candi(self,best_candi):
        best_mainpa = self.main_papers[best_candi]
        main_txt = best_mainpa.tabs
        sidx = []
        df1 = relishdata[relishdata['query_id']==best_candi]
        for j in range(3):
            rrids = []
            rpas = []
            rscore = []
            df2 = df1[df1['score']==j]
            if len(df2)==0:
                continue
            #print("best_mainpa",best_candi)
            for refid in df2.cand_id:
                df3 = df2[df2['cand_id']==refid]
                rpas.append(df3.cand_text.iloc[0])
                rscore.append(df3.score.iloc[0])
                rrids.append(refid)
            sortindex = self.find_max_similarity(rpas,main_txt,1)
            sidx.extend([{rrids[i]:{"text":rpas[i],"score":rscore[i]}} for i in sortindex])
        return sidx

    def find_max_similarity(self,rpapers,paper,k):
        self.embeddings.index(rpapers)
        sort_ind = self.embeddings.search(paper, k)
        return [sort_ind[i][0] for i in range(k)]

class relish_paper:
    def __init__(self,pid,sets,relishdata):
        # paper_info is a id group
        paper_info = relishdata[relishdata['query_id']==pid]
        self.tabs = paper_info.iloc[0].query_text
        self.sets = sets
        if sets=='test':
            refs = self.sample_ref(paper_info)#paper_info.sample(n=8).reset_index()#self.sample_ref(paper_info)
            row_list = [refs.iloc[[i]] for i in range(len(refs))]
            self.refs = [self.get_ref_prop(rf) for rf in row_list]
        else:
            self.refids = paper_info.cand_id
    
    def get_ref_prop(self,dfref):
        res = dfref.iloc[0]
        return {res.cand_id:{"tabs":res.cand_text,"score":res.score}}

    def pair_elements(self):
        random.seed(42)
        random.shuffle(self.refs)
        return [self.refs[:4],self.refs[4:]]

    def sample_ref(self,df):
        random.seed(42)
        # df is a id group
        #df = relishdata[relishdata['query_id']==22569528]
        grp2 = df[df['score']==2]
        grp1 = df[df['score']==1]
        grp0 = df[df['score']==0]
        try:
            d2 = grp2.sample(n=1)
            grp2_1 = grp2.drop(d2.index)
        except ValueError:
            d2 = grp1.sample(n=1)
            grp2_1 = grp1.drop(d2.index)       
        try:
            d1 = grp1.sample(n=1)
            grp1_1 = grp1.drop(d1.index)
        except ValueError:
            d1 = grp2_1.sample(n=1)
            grp1_1 = grp2_1.drop(d1.index)
        try:
            d0 = grp0.sample(n=1)
            grp0_1 = grp0.drop(d0.index)
        except ValueError:
            d0 = grp2_1.sample(n=1)
            grp0_1 = grp2_1.drop(d0.index)
        weights = [0.36, 0.33, 0.31]

        # 根据权重随机选择一个 DataFrame
        dataframes = [grp2_1,grp1_1,grp0_1]
        # 从选中的 DataFrame 中随机选择一行
        ali = [0,1,2]
        selected_num = np.random.choice([0,1,2], p=weights)
        selected_df = dataframes[selected_num]
        try:
            d3 = selected_df.sample(n=1)
        except ValueError:
            ali.remove(selected_num)
            try:
                d3 = dataframes[ali[0]].sample(n=1)
            except ValueError:
                ali.remove(ali[0])
                d3 = dataframes[ali[0]].sample(n=1)
        return [d2,d1,d0,d3]