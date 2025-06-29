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
from InterDDI.DrugBank.model import Combin_Classifier
import pandas as pd
import torchvision.transforms as transforms


class DualInputViTWrapper(torch.nn.Module):
    def __init__(self, model, eval_ent, eval_rel, args, entity_vocab=None, relation_vocab=None):
        super(DualInputViTWrapper, self).__init__()
        self.model = model
        self.eval_ent = eval_ent
        self.eval_rel = eval_rel
        self.all_rel = args.all_rel
        self.args = args
        self.entity_vocab = entity_vocab
        self.relation_vocab = relation_vocab

    def forward(self,x,head,tail,KG):
        # 假设输入 x 是一个字典，包含 img1 和 img2
        img1 = x[:,:,:224,:]
        img2 = x[:,:,224:,:]
        result = self.model(img1,img2,head,tail,KG)
        # print(result)
        return result
    
    
    def visualize(self, triplet, KG, head_batch=True):
        h, t, r = triplet[0], triplet[1], triplet[2:]
        paths, weights = self.model.visualize_forward(h.unsqueeze(0), t.unsqueeze(0), r.unsqueeze(0), KG, 10, head_batch)
        outputs = []
        rel_weights = [0] * (self.all_rel - self.eval_rel)
        rel_freq = [0] * self.all_rel
        for path, weight in zip(paths, weights):
            out_str = '%4f\t' % weight
            for i in range(len(path)):
                h, t, r = path[i]
                # print(path[i])
                h_name = self.entity_vocab[h]
                t_name = self.entity_vocab[t]
                if r == 2*self.all_rel:
                    r_name = 'idd'
                else:
                    r_mod = r % self.all_rel
                    if r_mod >= self.eval_rel:
                        r_name = self.relation_vocab[r_mod]
                    else:
                        r_name = str(r % self.all_rel)
                    rel_freq[r_mod] += 1

                if r >= self.all_rel and r < 2*self.all_rel:
                    r_name += "_inv"
                if r >= self.eval_rel and r < self.all_rel:
                    rel_weights[r-self.eval_rel] += 1
                elif r >= self.all_rel+self.eval_rel and r < 2*self.all_rel:
                    rel_weights[r-self.eval_rel-self.all_rel] += 1

                if i == 0:
                    out_str += '< %s, %6s, %18s' % (h_name, r_name, t_name)
                else:
                    out_str += ', %6s, %18s' % (r_name, t_name)
            out_str += ' >\n'
            outputs.append(out_str)
        return outputs, np.array(rel_weights), np.array(rel_freq)

    def KG_relation_weights(self, triplets, KG):
        heads, tails = triplets[:,0], triplets[:,1]
        batch_size = self.args.batch_size
        num_batch = len(heads) // batch_size + int(len(heads)%batch_size>0)
        rel_weights = [[] for i in range(self.args.length)]
        self.model.eval()
        for i in range(num_batch):
            start = i * batch_size
            end = min((i+1)*batch_size, len(heads))
            batch_h = heads[start:end].cuda()
            batch_t = tails[start:end].cuda()
            relations = self.model.get_attention_weights(batch_h, batch_t, KG)
            for l in range(self.args.length):
                rel_weights[l].append(relations[l])
       
        all_weights = 0
        for l in range(self.args.length):
            rel_weight = np.concatenate(rel_weights[l], axis=0) # [N, n_rel]
            rel_weight = np.mean(rel_weight, axis=0)    # n_rel
            kg_weight = rel_weight[self.eval_rel:self.all_rel]
            kg_weight += rel_weight[self.all_rel+self.eval_rel:2*self.all_rel]
            kg_weight /= 2
            all_weights += kg_weight
            print(l, list(np.round(kg_weight, 2)))
        print(list(np.round(all_weights/self.args.length, 2)))

    def Path_Statistics(self, triplet, KG, head_batch=True):
        h, t, tr = triplet[0], triplet[1], triplet[2:]
        paths, weights = self.model.visualize_forward(h.unsqueeze(0), t.unsqueeze(0), tr.unsqueeze(0), KG, 10, head_batch)
        results = []

        for path, weight in zip(paths, weights):
            if weight is None or float(weight) == 0:
                return results
            entities = []
            relations = []
            path_len = len(path)
            for i in range(path_len):
                h, t, r = path[i]
                # if h == t :
                #     continue
                h_name = self.entity_vocab[h]
                t_name = self.entity_vocab[t]
                if r == 2*self.all_rel:
                    r_name = 'idd'
                else:
                    r_mod = r % self.all_rel
                    if r_mod >= self.eval_rel:
                        r_name = self.relation_vocab[r_mod]
                    else:
                        r_name = str(r % self.all_rel)
                entities.append(h_name)
                relations.append(r_name)
                if i == (path_len-1) :
                    entities.append(t_name)
            while len(relations) < 3:
                relations.append('')
            while len(entities) < 4:
                entities.insert(-1, '')  # 保证 tail 在末尾

            result = {
                'weight': float(weight),
                'head': entities[0],
                'rel1': relations[0],
                'mid1': entities[1],
                'rel2': relations[1],
                'mid2': entities[2],
                'rel3': relations[2],
                'tail': entities[-1],
                'true_rel': tr.item()
            }
            results.append(result)
        return results


def reshape_transform(tensor, height=28, width=14):
    # print(f"Original tensor shape: {tensor.shape}")
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

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
        if params.kg_true :
            ddi_paths = {
                'train': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'train')),
                'valid': os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'valid')),
                'test':  os.path.join(self.task_dir, '{}/{}_ddi.txt'.format(params.datatask, 'test'))
            }
        else : 
            ddi_paths = {
                'train': os.path.join(self.task_dir, '{}/{}/{}.txt'.format('Case',params.datatask, 'otherddi')),
                'valid': os.path.join(self.task_dir, '{}/{}/{}.txt'.format('Case',params.datatask, 'part1')),
                'test':  os.path.join(self.task_dir, '{}/{}/{}.txt'.format('Case',params.datatask, 'part2'))
            }
        print("ddi path",ddi_paths['train'])
        print("ddi path",ddi_paths['valid'])
        print("ddi path",ddi_paths['test'])
        kg_paths = {
            'train': os.path.join(self.task_dir, 'S0/{}_KG.txt'.format('train')),
            'valid': os.path.join(self.task_dir, 'S0/{}_KG.txt'.format('valid')),
            'test':  os.path.join(self.task_dir, 'S0/{}_KG.txt'.format('test'))
        }
        self.load_ent_id()
        # self.load_drug_atc()
        if  params.force_reload or TSDDIDataset.triplets is None:
            print('This times load ddi and kg')
            self.process_files_ddi(ddi_paths, saved_relation2id)
            self.process_files_kg(kg_paths, saved_relation2id)
            # self.load_ent_id()
            
            self.VKG = self.load_graph(np.concatenate([TSDDIDataset.triplets['train'],TSDDIDataset.triplets['test'], self.train_kg], axis=0))
            if params.kg_true :
                self.VKG = self.load_graph(self.train_kg)
            self.TKG = self.load_graph(np.concatenate([TSDDIDataset.triplets['train'],TSDDIDataset.triplets['valid'], self.train_kg], axis=0))

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
        atc_data = pd.read_csv('/root/workspace/visualDDI/datasets/drug_class.csv')
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
        n_all = len(all_triplet)
        rand_idx = np.random.permutation(n_all)
        all_triplet = all_triplet[rand_idx]
        n_fact = int(n_all * 0.8)
        kg_triplets = np.concatenate([all_triplet[:n_fact], self.train_kg], axis=0)
        TSDDIDataset.base_kg = self.load_graph(kg_triplets)
        self.samples = np.array(all_triplet[n_fact:].tolist())
        # for VisualDDI
        # self.samples = np.array(all_triplet.tolist())


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
        path1 = f'/root/workspace/visualDDI/datasets/drug_images/{self.id2entity[d1]}.png'
        path2 = f'/root/workspace/visualDDI/datasets/drug_images/{self.id2entity[d2]}.png'
        img1 = Image.open(path1)
        img1np = np.float32(np.array(img1)) / 255
        img2 = Image.open(path2)
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2,img1np,self.samples[index]


