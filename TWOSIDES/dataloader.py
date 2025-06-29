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
    Pos_triplets = None # 共同属性 正边
    Neg_triplets = None #  负边
    Fact_triplets = None # 加入到KG中的实际边
    Kg_triplets = None # KG（DDI-KG + Bio-KG）
    train_kg = None
    valid_kg = None
    test_kg  = None
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
        print("ddi path",ddi_paths['train'])
        kg_paths = {
            'train': os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'train')),
            'valid': os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'valid')),
            'test':  os.path.join(self.task_dir, '{}/{}_KG.txt'.format(params.datatask, 'test'))
        }
        self.load_ent_id()
        if TSDDIDataset.Pos_triplets is None:
            self.process_files_ddi(ddi_paths, saved_relation2id)
            self.process_files_kg(kg_paths, saved_relation2id)

        if TSDDIDataset.base_kg is None :
            self.shuffle_train()

            fact_triplets = []
            for triplet in TSDDIDataset.Pos_triplets['train']:
                h, t, r = triplet[0], triplet[1], triplet[2:-1]
                for s in np.nonzero(r)[0]:
                    fact_triplets.append([h,t,s])
            self.vKG = self.load_graph(np.array(fact_triplets), self.valid_kg)

            for triplet in TSDDIDataset.Pos_triplets['valid']:
                h, t, r = triplet[0], triplet[1], triplet[2:-1]
                for s in np.nonzero(r)[0]:
                    fact_triplets.append([h,t,s])
            self.tKG = self.load_graph(np.array(fact_triplets), self.test_kg)
        else :
            all_datas =np.concatenate([TSDDIDataset.Pos_triplets[data_type],TSDDIDataset.Neg_triplets[data_type]],axis=0) 
            self.samples = all_datas

            

    def process_files_ddi(self, file_paths, saved_relation2id=None):
        entity2id = {}
        relation2id = {} if saved_relation2id is None else saved_relation2id
        TSDDIDataset.Pos_triplets = {}
        TSDDIDataset.Neg_triplets = {}
        
        self.train_ent = set()
        self.ent_pair = set()
        
        for file_type, file_path in file_paths.items():
            triplets = []
            pos_triplet = []
            neg_triplet = []
            with open(file_path)as f:
                file_data: list[list[str]] = [line.split('\t') for line in f.read().split('\n')[:-1]]

            for triplet in file_data:
                h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[-1])
                z1 = list(map(int, triplet[2].split(',')))
                z = [i for i, value in enumerate(z1) if value == 1]
                
                if len(z) >= 0 :
                    entity2id.setdefault(h, h) 
                    entity2id.setdefault(t, t)
                if not saved_relation2id :
                    for s in z:
                        relation2id.setdefault(s, s)
                if r==1:
                        # pos_triplet.append([x,y] + z1)
                    pos_triplet.append([h,t] + z1 +[1])
                    self.ent_pair.add((h,t))
                    self.ent_pair.add((t,h))
                else:
                        # neg_triplet.append([x,y] + z1)
                    neg_triplet.append([h,t] + z1 + [0])
                if file_type == 'train':
                    self.train_ent.add(h)
                    self.train_ent.add(t)
                triplets.append([h, t, r])

            TSDDIDataset.Pos_triplets[file_type] = np.array(pos_triplet, dtype='int')
            TSDDIDataset.Neg_triplets[file_type] = np.array(neg_triplet, dtype='int')


        self.entity2id = entity2id
        self.relation2id = relation2id
        self.eval_ent = max(self.entity2id.keys()) + 1 
        
        self.eval_rel = len(self.relation2id)
        print( len(self.relation2id))

    def load_ent_id(self, ):
        id2entity = dict()
        id2relation = dict()
        drug_set = json.load(open(os.path.join(self.task_dir, 'id2drug.json'), 'r'))
        entity_set = json.load(open(os.path.join(self.task_dir, 'entity2id.json'), 'r'))
        relation_set = json.load(open(os.path.join(self.task_dir, 'relation2id.json'), 'r'))
        for drug in drug_set:
            id2entity[int(drug)] = drug_set[drug]['cid']
        for ent in entity_set:
            id2entity[int(entity_set[ent])] = ent

        for rel in relation_set:
            id2relation[int(rel)] = relation_set[rel]
        
        self.id2entity = id2entity
        self.id2relation = id2relation

            
        
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

    def load_graph(self, triplets, kg_triplets=None):
        new_triplets = []
        for triplet in triplets:
            h, t, r = triplet
            new_triplets.append([t, h, 0])
            new_triplets.append([h, t, 0])
        if kg_triplets is not None:
            for triplet in kg_triplets:
                h, t, r = triplet
                r_inv = r + self.all_rel-self.eval_rel
                new_triplets.append([t, h, r])
                new_triplets.append([h, t, r_inv])
        edges = np.array(new_triplets)
        all_rel = 2*self.all_rel - self.eval_rel
        idd = np.concatenate([np.expand_dims(np.arange(self.all_ent),1), np.expand_dims(np.arange(self.all_ent),1), all_rel*np.ones((self.all_ent, 1))],1)
        edges = np.concatenate([edges, idd], axis=0)
        values = np.ones(edges.shape[0])
        adjs = torch.sparse_coo_tensor(indices=torch.LongTensor(edges).t(), values=torch.FloatTensor(values), size=torch.Size([self.all_ent, self.all_ent, all_rel+1]), requires_grad=False).cuda()
        return adjs

    def shuffle_train(self, ratio=0.8):
        n_ent = len(self.ddi_in_kg)
        train_ent = set(self.train_ent) - set(np.random.choice(list(self.ddi_in_kg), n_ent-int(n_ent*ratio)))
        all_triplet = np.array(TSDDIDataset.Pos_triplets['train'])
        self.samples =[]
        if  'S1' in self.datatask:
            fact_triplet = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i,0], all_triplet[i,1], all_triplet[i,2:-1]
                if h in train_ent and t in train_ent:
                    for s in np.nonzero(r)[0]:
                        fact_triplet.append([h, t, s])
                elif h in train_ent or t in train_ent:
                    self.samples.append(TSDDIDataset.Pos_triplets['train'][i])
                    self.samples.append(TSDDIDataset.Neg_triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.samples = np.array(self.samples)
            TSDDIDataset.base_kg = self.load_graph(fact_triplet, self.train_kg)
        elif 'S2' in self.datatask:
            fact_triplet = []
            for i in range(len(all_triplet)):
                h, t, r = all_triplet[i,0], all_triplet[i,1], all_triplet[i,2:-1]
                if h in train_ent and t in train_ent:
                    for s in np.nonzero(r)[0]:
                        fact_triplet.append([h, t, s])
                elif h not in train_ent and t not in train_ent:
                    self.samples.append(TSDDIDataset.Pos_triplets['train'][i])
                    self.samples.append(TSDDIDataset.Neg_triplets['train'][i])
            fact_triplet = np.array(fact_triplet)
            self.samples = np.array(self.samples)
            TSDDIDataset.base_kg = self.load_graph(fact_triplet, self.train_kg)

        elif 'S0' in self.datatask:
            n_all = len(all_triplet)
            rand_idx = np.random.permutation(n_all)
            n_fact = int(n_all * 0.8)
            pos_triplet = all_triplet[[rand_idx[n_fact:]]]
            neg_triplet = np.array(TSDDIDataset.Neg_triplets['train'][rand_idx[n_fact:]])

            facts = all_triplet[rand_idx[:n_fact]]
            
            fact_triplet = []
            for i in range(n_fact):
                x, y, z = facts[i,0], facts[i,1], facts[i,2:-1]
                for s in np.nonzero(z)[0]:
                    fact_triplet.append([x, y, s])

            TSDDIDataset.base_kg = self.load_graph(fact_triplet, self.train_kg)

            interleaved = [val for pair in zip(pos_triplet, neg_triplet) for val in pair] # 把

            self.samples = np.array(interleaved)


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
        
        d1, d2, *label = self.samples[index]
        path1 = f'../dataset/TWOSIDES/drug_images/{d1}.png'
        path2 = f'../dataset/TWOSIDES/drug_images/{d2}.png'
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2,d1,d2,label
