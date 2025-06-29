import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import json
import numpy as np

class MAEDDIDataset(Dataset):
    # entity_embbing = None # 节点embbing
    def __init__(self,params,datatype='train',KGE_mode = 'RotatE'):
        super().__init__()
        self.task_dir = params.task_dir
        ddi_paths = {
            'train': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'train')),
            'valid': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'valid')),
            'test':  os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'test'))
        }
        embeddings_path = {
            'train': os.path.join(params.embedding_path, KGE_mode,'entity_embedding.npy'),
            'valid':os.path.join(params.embedding_path, KGE_mode,'entity_embedding.npy'),
            'test': os.path.join(params.embedding_path, KGE_mode ,'entity_embedding.npy'),
            # "train": os.path.join(params.embedding_path, 'kgnn_embed.npy'),
            # "valid": os.path.join(params.embedding_path, 'kgnn_embed.npy'),
            # "test": os.path.join(params.embedding_path,  'kgnn_embed.npy')
        }
        self.load_ent_id()
        self.process_files_ddi(ddi_paths[datatype])
        # if MAEDDIDataset.entity_embbing is None :
            # MAEDDIDataset.entity_embbing = embeddings = np.load(params.embedding_path)
        self.entity_emb = np.load(embeddings_path[datatype])
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
    
    def process_files_ddi(self, file_path, saved_relation2id=None):


        self.train_ent = set()
        self.ent_pair = set()

        triplets = []
        with open(file_path)as f:
            file_data = [line.split() for line in f.read().split('\n')[:-1]]

        for triplet in file_data:
            h, t, r = int(triplet[0]), int(triplet[1]), int(triplet[2])
            triplets.append([h, t, r])

        self.samples = np.array(triplets, dtype='int')

        self.eval_ent = 1710
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
        self.all_ent = max(self.id2entity.keys()) + 1
        self.all_rel = max(self.id2relation.keys()) + 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        d1, d2, label = self.samples[index]
        path1 = f'/root/workspace/visualDDI/datasets/drug_images/{self.id2entity[d1]}.png'
        # print(self.id2entity[d1],self.id2entity[d2],label)
        path2 = f'/root/workspace/visualDDI/datasets/drug_images/{self.id2entity[d2]}.png'
        
        img1 = Image.open(path1)
        img2 = Image.open(path2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        embedding1 = self.entity_emb[d1]
        embedding2=self.entity_emb[d2]
        return img1, img2,embedding1,embedding2,label
    






class TSDDIDataset(Dataset):
    def __init__(self, fold='S1', data_type='train' ):
        super().__init__()
        self.samples = []
        path = f'datasets/TWOSIDES/{fold}/{data_type}_ddi.txt'
        print(path)
        self.samples = []
        with open(path, 'r') as f:
            f.readline()
            bar = tqdm(f)
            for idx, line in enumerate(bar):
                h, t, label, bin_label = line.strip().split('\t')
                self.samples.append([f'datasets/drug_images/{h}.png', f'datasets/drug_images/{t}.png', int(bin_label)])

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        d1, d2, label = self.samples[index]
        img1 = Image.open(d1)
        img2 = Image.open(d2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, label
