import numpy as np
from utils.utils import process_and_compare_predictions, load_data, sample_test_nodes
import sys
import requests
requests.packages.urllib3.disable_warnings()
import yaml
import json
from txtai.embeddings import Embeddings
import pickle
from Retriever import dp
import time
import random


class linkInContextAgent:
    def __init__(self,parafile):
        # read YAML
        with open(parafile, 'r') as file:
            parameters = yaml.safe_load(file)
        ssize = parameters['samplesize']
        self.test = parameters['testsize']
        self.load_embedding = parameters['load_embedding']
        self.dataname = parameters['dataset_name']
        self.LLM = parameters['LLM']
        self.maxp1 = parameters['max_papers'][0]
        self.maxp2 = parameters['max_papers'][1]
        self.abs_len = parameters['abstract_len']
        self.shotnum = parameters['fewshot_num']
        self.method = parameters['method']
        self.style = parameters['style']
        ## create embed

        ## load data
        self.data, self.text = load_data(self.dataname, use_text=True, seed=42)
        if ssize == "all":
            self.sample_size = int(len(self.data.test_id)/2)
        else:
            self.sample_size = ssize
        ##
        idx_list = list(range(self.sample_size))
        node_indices = sample_test_nodes(self.data, self.text, self.sample_size, self.dataname)
        self.node_index_list = [node_indices[idx] for idx in idx_list]

        if parameters['emb'][0]:
            self.creatembed(parameters['emb'][1])
        else:
            self.embeddings = None
        
        self.resources = {'data':self.data,'text':self.text,'node_list':self.node_index_list,'embeddings':self.embeddings}
    def loadjson(self,path):
        with open(path, 'r') as f:
            fi = json.load(f)
        return fi

    def dumppkl(self,fname,res,folder = 'results/predictresult/'):
        with open(folder+fname, 'wb') as file:
            pickle.dump(res, file,pickle.HIGHEST_PROTOCOL)
    def dumpjson(self,fname,res,folder = 'results/predictresult/'):
        with open(folder+fname,"w") as f:
            json.dump(res,f)
    def logger(self,experName):
        self.experName = experName
        #self.jsonfile = open('jsonfile/'+self.dataname+"__"+self.experName+'.jsonl','a')
        self.f = open('logresult/'+self.dataname+"__"+self.experName+'.log', 'a')
        sys.stdout = self.f
        print("experName .... ",self.experName)

    def creatembed(self,index_file):
        if self.load_embedding:
            self.embeddings = Embeddings()
            self.embeddings.load(index_file)
        else:
            self.embeddings = self.createIndex(self.text)
            self.embeddings.save(index_file)
            
    def createIndex(self,alltext):
        alltxt = []
        for i in range(len(alltext['title'])):
            alltxt.append(alltext['title'][i]+"|| "+alltext['abs'][i])
        # Create embeddings model, backed by sentence-transformers & transformers
        embeddings = Embeddings(path="allenai/scibert_scivocab_uncased")
        # Index the list of te
        embeddings.index(alltxt)
        return embeddings

    def constructExamples(self,raw_exam_file):
        if self.method == "citation":
            with open(raw_exam_file, 'rb') as inp:
                examples = pickle.load(inp)
            return examples
        examples = {}
        with open(raw_exam_file, 'r') as inp:
            raw_exam= json.load(inp)

        if self.method == "cluster":
            for key,val in raw_exam.items():
                examples[int(key)] = [dp(v,-1,self.has_label(key)) for v in val]
        else:
            for key,val in raw_exam.items():
                examples[int(key)] = [dp(v,-1,self.has_label(key)) for v in val[1:self.shotnum+1]]
        return examples