import itertools
from prompthub import PST_proximity,REL_proximity
import json
import numpy as np
import random
#from LLM import Inferencer
#Olla = Inferencer()

A1 = "GOAL"
A2 = "METHOD"
A3 = "IDEA"
A4 = "THEORY/EXPER"

class ClassRunner:
    def __init__(self, agent,retr):
        super().__init__(agent,retr)
        self.retr = retr
        self.boter = DoT_classify(self.dataset)


    def create_reason(self,save,name,start,end=10000):
        #doterdic = self.get_dotdic()
        if self.dname == 'pubmed':
            doterdic = self.all124()
        reasondict = self.create_reason_model(doterdic,self.train_list[start:end],cite_nei_dic[self.dname],curr_reason)
        if save:
            self.agent.dumpjson(name+".json",reasondict,folder = 'Genereason/')
        return reasondict

    def get_dotdic(self):
        if self.dname == 'cora':
            doterdic = self.dot_indices_cora()
        elif self.dname == 'pubmed':
            doterdic = self.dot_indices_pubmed()
        return doterdic

    def ABRP_model(self,reason_dict,asp_dict,id2exam,mode,):
        doterdic = self.get_dotdic()
        print("========================================")
        ### construct whole prompt
        for i,id_ in enumerate(self.test_data):
            self.dataset.set_id(id_)
            citenei = self.retr.get_hops(id_)
            exms = self.retr.select_citations(citenei,4)
            #asp_list = list(doterdic[str(id_)].keys())
            self.boter.set_idx(id_,exms)
            if self.dname == "cora":
                self.boter.set_aspect(doterdic[str(id_)])
            else:
                self.boter.set_aspect(asp_dict)
            print(id_,"**************************")
            if mode[:4] == "abrp":
                message = self.boter.few_shot(reason_dict[str(id2exam[id_])])
            elif mode == "bot":
                message = self.boter.zero_shot()

        return message



    def create_reason_model(self,prompter_dics,test_list,cite_neibor,curr_reason):
        #select_exm = self.agent.constructExamples(cite_neibor)
        reasondict = curr_reason
        doter = DoT_classify(self.dataset)
        ### construct whole prompt
        for i,id_ in enumerate(test_list):
            self.dataset.set_id(id_)
            doter.set_aspect(prompter_dics[str(id_)])
            asp_list = list(prompter_dics[str(id_)].keys())
            doter.set_idx(id_,[])
            message = doter.create_reason()

        return message


class PSTRunner:
    def __init__(self,agent):
        self.pster = PST_proximity(agent)
        self.agent = agent
        self.pstdata = self.agent.data
        self.mid_list = self.pstdata.node_index_list

        
    def generate_BRD_prompts(self,mid):
        # mid is in self.mid_list 
        best_candi = self.pstdata.find_candidate(mid)
        ref_ids =  self.pstdata.find_ref_of_candi(best_candi)
        self.pster.set_idx(best_candi,ref_ids,[])
        message = self.pster.Example_create_prompting()
        return message

    def DBRP_prompts(self,id_,BRD_file):
        # mid is in self.mid_list
        PST_dir = "dataset/PST/"
        self.llmres_exm = self.agent.loadjson(PST_dir+BRD_file)
        pair_test = self.pstdata.main_papers[id_].pair_test
        for input_id in pair_test:
            self.pster.set_idx(id_,input_id,examples = self.llmres_exm[id_])
            labelslist = self.pster.prompt['ref_labels']
            self.pster.set_aspect({A2:"",A3:"",A4:""})
            message = self.pster.COT_DCOM_promting_base()
        return message,labelslist


class RELRunner:
    def __init__(self, agent,retr):
        self.reler = REL_proximity(agent)
        self.reldata = self.agent.data
        self.mid_list = self.reldata.node_index_list
        self.agent = agent


    def generate_BRD_prompts(self,mid):
        # mid is in self.mid_list 
        best_candi = self.pstdata.find_candidate(mid)
        ref_ids =  self.pstdata.find_ref_of_candi(best_candi)
        self.reler.set_idx(best_candi,ref_ids,[])
        message = self.reler.Example_create_prompting()
        return message

    def DBRP_prompts(self,id_,BRD_file):
        # mid is in self.mid_list
        PST_dir = "dataset/PST/"
        self.llmres_exm = self.agent.loadjson(PST_dir+BRD_file)
        pair_test = self.pstdata.main_papers[id_].pair_test
        for input_id in pair_test:
            self.pster.set_idx(id_,input_id,examples = self.llmres_exm[id_])
            labelslist = self.pster.prompt['ref_labels']
            self.pster.set_aspect({A2:"",A3:"",A4:""})
            message = self.pster.COT_DCOM_promting_base()
        return message,labelslist
