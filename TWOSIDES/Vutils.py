from turtle import pos
from rdkit import Chem
from rdkit.Chem import Draw
from regex import P
from sympy import N
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score,average_precision_score
# from dataloader import DDIDataset, ImageMolDDI
from torch.utils.data import DataLoader
import numpy as np
import pickle
import torch
import torch.nn as nn
import os
import torchvision
from PIL import Image
import torch
import random


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loadSmilesAndSave(smis, path):
    '''
        smis: e.g. COC1=C(C=CC(=C1)NS(=O)(=O)C)C2=CN=CN3C2=CC=C3
        path: E:/a/b/c.png

        ==============================================================================================================
        demo:
            smiless = ["OC[C@@H](NC(=O)C(Cl)Cl)[C@H](O)C1=CC=C(C=C1)[N+]([O-])=O", "CN1CCN(CC1)C(C1=CC=CC=C1)C1=CC=C(Cl)C=C1",
              "[H][C@@](O)(CO)[C@@]([H])(O)[C@]([H])(O)[C@@]([H])(O)C=O", "CNC(NCCSCC1=CC=C(CN(C)C)O1)=C[N+]([O-])=O",
              "[H]C(=O)[C@H](O)[C@@H](O)[C@@H](O)[C@H](O)CO", "CC[C@H](C)[C@H](NC(=O)[C@H](CC1=CC=C(O)C=C1)NC(=O)[C@@H](NC(=O)[C@H](CCCN=C(N)N)NC(=O)[C@@H](N)CC(O)=O)C(C)C)C(=O)N[C@@H](CC1=CN=CN1)C(=O)N1CCC[C@H]1C(=O)N[C@@H](CC1=CC=CC=C1)C(O)=O"]

            for idx, smiles in enumerate(smiless):
                loadSmilesAndSave(smiles, "{}.png".format(idx+1))
        ==============================================================================================================

    '''
    mol = Chem.MolFromSmiles(smis)
    img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
    img.save(path)


    

def evaluate_multi_class(y_pred, labels):
    y_pred = np.argmax(y_pred, axis=1)
    f1 = f1_score(labels, y_pred, average='macro')

    recall = recall_score(labels, y_pred, average='macro')
    precision = precision_score(labels, y_pred, average='macro')
    acc = accuracy_score(labels, y_pred)
    return f1, recall, precision, acc

def evaluate_multi_label(y_pred, labels):
    y_pred = y_pred
    labels = labels
    auc_score = roc_auc_score(labels, y_pred)
    
    aupr_score = average_precision_score(labels, y_pred)
    
    y_label = [1 if i > 0.5 else 0 for i in y_pred]
    acc = accuracy_score(labels, y_label)
    
    return acc, auc_score, aupr_score

def eval_mae_loader(model, loader, device,KG):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2,d1,d2,label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = torch.stack(label, dim=0)
        label = label.T.to(device)
        label = label.float()
        with torch.no_grad():
            pred = model(img1, img2, d1,d2,KG)
            pred = torch.softmax(pred, dim=1)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_multi_class(np.array(Y_pre), np.array(Y_true))

def eval_mae_loader_ts(model, loader, device,KG):
    model.eval()
    # model.cuda()
    model = model.to(device)
    Y_pre = []
    Y_true = []
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2,d1,d2, label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        label = torch.stack(label, dim=0)
        # print(label.shape)
        label = label.T.to(device)
        label = label.float()
        with torch.no_grad():
            pred = model(img1, img2,d1,d2,KG)
            pred = torch.sigmoid(pred)
            Y_pre.extend(list(pred.cpu().detach().numpy()))
            Y_true.extend(list(label[:,-1].cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))
    return evaluate_multi_label(np.array(Y_pre), np.array(Y_true))

def eval_mae_loader_ts_mutil(model, loader, device,KG):
    model.eval()
    # model.cuda()
    model = model.to(device)

    pred_labels = []
    rel_labels = []
    pred_class = {}
    bar = tqdm(enumerate(loader))
    for b_idx, batch in bar:
        img1, img2,d1,d2, label = batch
        img1 = img1.to(device)
        img2 = img2.to(device)
        d1 = d1.to(device)
        d2 = d2.to(device)
        label = torch.stack(label, dim=0)
        label = label.T.to(device)
        label = label.float()
        with torch.no_grad():
            pred = model(img1, img2,d1,d2,KG)
            pred=torch.sigmoid(pred)
            pred_labels.append(list(pred.cpu().detach().numpy()))
            rel_labels.append(list(label.cpu().detach().numpy()))
        bar.set_description('Evaluating: {}/{}'.format(str(b_idx+1), len(loader)))

    pred_labels = np.concatenate(pred_labels)
    rel_labels = np.concatenate(rel_labels)
    is_pos = rel_labels[:, -1] > 0
    pos_scores = pred_labels[is_pos]                                          
    neg_scores = pred_labels[~is_pos]


    label_pos = rel_labels[is_pos, :-1]  # shape (num_pos, 200)
    label_neg = rel_labels[~is_pos, :-1]  # shape (num_neg, 200)


    index = None

    eval_rel = 200
    for r in range(eval_rel):
        index = label_pos[:, r] > 0
        index_neg = label_neg[:, r] > 0
        pred_class[r] = {'score': list(pos_scores[index, r]) + list(neg_scores[index, r]),
                    'preds': list((pos_scores[index,r] > 0.5).astype('int')) + list((neg_scores[index,r]>0.5).astype('int')),
                    'label': [1] * np.sum(index) + [0] * np.sum(index)}
       
    roc_auc = []
    prc_auc = []
    ap = []
    for r in range(eval_rel):
        label = pred_class[r]['label']
        score = pred_class[r]['score']
        if len(label)==0:
            continue
        sort_label = np.array(sorted(zip(score, label), reverse=True))
        roc_auc.append(roc_auc_score(label, score))
        prc_auc.append(average_precision_score(label, score))
        k = int(len(label)//2)
        apk = np.sum(sort_label[:k,1])
        ap.append(apk/k)

    pos_mask = label_pos > 0
    neg_mask = label_neg > 0
    pos_acc = pos_scores[pos_mask]
    neg_acc = neg_scores[pos_mask]
    pos_acc = (pos_acc > 0.5).astype('int')
    neg_acc = (neg_acc > 0.5).astype('int') 

    pos_acc = np.sum(pos_acc) / len(pos_acc)
    neg_acc = np.sum(neg_acc) / len(neg_acc)  #  反向acc 因为neg_mask是负例，所以neg_acc是负例的错误率
    return np.mean(roc_auc), np.mean(prc_auc), np.mean(ap)

    # return evaluate_multi_label(np.array(Y_pre), np.array(Y_true))
