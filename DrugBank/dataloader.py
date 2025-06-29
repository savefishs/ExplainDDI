from ast import arg
from calendar import c
import os
from networkx import parse_multiline_adjlist
from regex import P
import torch
import random
import numpy as np
from collections import defaultdict
import json
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import pandas as pd
import torchvision.transforms as transforms


class TSDDIDataset(Dataset): #DrugBank 无负边
    base_kg = None
    triplets = None # 共同属性 正边
    Kg_triplets = None # KG（DDI-KG + Bio-KG）
    def __init__(self, params ,data_type='train',saved_relation2id=None):
        # super().__init__()


        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])

        # -----------------------------------------------------------
        #EmerGNN
        self.task_dir = params.task_dir
        self.datatask = params.datatask
        self.datatype = data_type

        ddi_paths = {
            'train': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'train')),
            'valid': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'valid')),
            'test':  os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'test'))
        }
        kg_paths = {
            'train': os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'train')),
            'valid': os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'valid')),
            'test':  os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'test'))
        }
        self.load_ent_id()
        if TSDDIDataset.triplets is None:
            self.process_files_ddi(ddi_paths, saved_relation2id)
            self.process_files_kg(kg_paths, saved_relation2id)
        if TSDDIDataset.base_kg is None :
            self.shuffle_train()

            self.vKG = self.load_graph(np.concatenate([TSDDIDataset.triplets['train'], self.valid_kg], axis=0))
            if 'ATC' in params.datatask :
                self.tKG = self.load_graph(np.concatenate([TSDDIDataset.triplets['train'], self.test_kg], axis=0))
            else :
                self.tKG = self.load_graph(np.concatenate([TSDDIDataset.triplets['train'], TSDDIDataset.triplets['valid'], self.test_kg], axis=0))
        else :
            self.samples = TSDDIDataset.triplets[data_type]
            

    def process_files_ddi(self, file_paths, saved_relation2id=None):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id
        TSDDIDataset.triplets = {}
        
        self.train_ent = set()
        self.ent_pair = set()

        for file_type, file_path in file_paths.items():
            triplets = []
            with open(file_path)as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]

            for triplet in file_data:
                h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                entity2id.setdefault(h, h) 
                entity2id.setdefault(t, t)
                if not saved_relation2id :
                    relation2id.setdefault(r, r)
                if file_type == 'train':
                    self.train_ent.add(h)
                    self.train_ent.add(t)
                triplets.append([h, t, r])

            TSDDIDataset.triplets[file_type] = np.array(triplets, dtype='int')


        self.entity2id = entity2id
        self.relation2id = relation2id
        self.eval_ent = max(self.entity2id.keys()) + 1
        # self.eval_ent = 1710
        # self.eval_rel = len(self.relation2id)
        self.eval_rel = 86
        
    def load_ent_id(self, ):
        id2entity = dict()
        id2relation = dict()
        drug_set = json.load(open(os.path.join(self.task_dir, 'node2id.json'), 'r'))
        entity_set = json.load(open(os.path.join(self.task_dir, 'entity_drug.json'), 'r'))
        relation_set = json.load(open(os.path.join(self.task_dir, 'relation2id.json'), 'r'))
        for drug in drug_set:
            id2entity[int(drug_set[drug])] = drug
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent

        for rel in relation_set:
            id2relation[int(rel)] = relation_set[rel]
        
        self.id2entity = id2entity
        self.id2relation = id2relation

    def load_drug_atc(self, ):
        atc_data = pd.read_csv('../dataset/drug_class.csv')
        atc_data.fillna('False', inplace  = True)

        self.drug_atc = {}
        self.KnowDrug  = []
        for index, row in atc_data.iterrows():
            drug_id = row['drugbank_id']
            atc_codes = row['atc_codes'].split('|')  # 处理多个ATC代码
            categories = {code[0] for code in atc_codes}  # 提取每个ATC代码的首字母并去重
            self.drug_atc[drug_id] = categories
            self.KnowDrug.append(drug_id)
            
        
    def process_files_kg(self, kg_paths, saved_relation2id=None, ratio=1):
        TSDDIDataset.Kg_triplets = defaultdict(list)
        self.ddi_in_kg = set()
        print('pruned ratio of edges in KG: {}'.format(ratio))

        for file_type, file_path in kg_paths.items():
            with open(file_path) as f:
                file_data = [line.split() for line in f.read().split('\n')[:-1]]

                for triplet in file_data:
                    h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
                    self.entity2id.setdefault(h, h) 
                    self.entity2id.setdefault(t, t)
                    if not saved_relation2id:       
                        self.relation2id.setdefault(r, r)
                    TSDDIDataset.Kg_triplets[file_type].append([h, t, r])
                    if h in self.train_ent:
                        self.ddi_in_kg.add(h)
                    if t in self.train_ent:
                        self.ddi_in_kg.add(t)

        if ratio < 1:
            n_train = len(TSDDIDataset.Kg_triplets['train'])
            n_valid = len(TSDDIDataset.Kg_triplets['valid'])
            n_test = len(TSDDIDataset.Kg_triplets['valid'])
            TSDDIDataset.Kg_triplets['train'] = random.sample(TSDDIDataset.Kg_triplets['train'], int(ratio*n_train))
            TSDDIDataset.Kg_triplets['valid'] = random.sample(TSDDIDataset.Kg_triplets['valid'], int(ratio*n_valid))
            TSDDIDataset.Kg_triplets['test'] = random.sample(TSDDIDataset.Kg_triplets['test'], int(ratio*n_test))

        train_kg = TSDDIDataset.Kg_triplets['train']
        valid_kg = train_kg + TSDDIDataset.Kg_triplets['valid']
        test_kg  = valid_kg + TSDDIDataset.Kg_triplets['test']
        self.train_kg = np.array(train_kg, dtype='int')
        self.valid_kg = np.array(valid_kg, dtype='int')
        self.test_kg = np.array(test_kg, dtype='int')
        print("KG triplets: Train-{} Valid-{} Test-{}".format(len(train_kg), len(valid_kg), len(test_kg)))

        self.all_ent = max(self.entity2id.keys()) + 1
        self.all_rel = max(self.relation2id.keys()) + 1
        print(self.all_ent,self.all_rel)

    def load_graph(self, triplets):
        edges = self.double_triple(triplets)
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), 2*self.all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, 2*self.all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        n_ent = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(np.random.choice(list(self.ddi_in_kg), n_ent-int(n_ent*ratio)))
        all_triplet = np.array(TSDDIDataset.triplets['train'])
        self.samples =[]
        if  'S1' in self.datatask:
            fact_triplet = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h, t, r])
                elif h in train_ent or t in train_ent:
                    self.samples.append(TSDDIDataset.triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.samples = np.array(self.samples)
            kg_triplets = np.concatenate([fact_triplet, self.train_kg], axis=0)
            TSDDIDataset.base_kg = self.load_graph(kg_triplets)
        elif 'S2' in self.datatask:
            fact_triplet = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i]
                if h in train_ent and t in train_ent:
                    fact_triplet.append([h,t,r])
                elif h not in train_ent and t not in train_ent:
                    self.samples.append(TSDDIDataset.triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.samples = np.array(self.samples)
            kg_triplets = np.concatenate([fact_triplet, self.train_kg], axis=0)
            TSDDIDataset.base_kg = self.load_graph(kg_triplets)
        elif 'S0' in self.datatask:
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            all_triplet = all_triplet[rand_idx]
            n_fact = int(n_all * 0.8)
            kg_triplets = np.concatenate([all_triplet[:n_fact], self.train_kg], axis=0)
            TSDDIDataset.base_kg = self.load_graph(kg_triplets)
            self.samples = np.array(all_triplet[n_fact:].tolist())



    def double_triple(self, triplet):
        new_triples = []
        n_rel = self.all_rel
        for triple in triplet:
            h, t, r = triple
            new_triples.append([t, h, r])
            new_triples.append([h, t, r+n_rel])
        new_triples = np.array(new_triples)
        return new_triples


        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        
        d1, d2, label = self.samples[index]
        path1 = f'../dataset/DrugBank/drug_images/{self.id2entity[d1]}.png'
        path2 = f'../dataset/DrugBank/drug_images/{self.id2entity[d2]}.png'
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2,d1,d2,label
