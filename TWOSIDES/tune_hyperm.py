from ast import arg
import os
import argparse
import sys
import math
from turtle import pos
from regex import E, P
import torch
import torch.utils
import importlib
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from zmq import device
import warnings
# from Vmae_model import *
from Vutils import setup_seed, eval_mae_loader_ts_mutil
import random
# for reload change
from mae_model import *
from model import Combin_Classifier
import numpy as np
from dataloader import TSDDIDataset
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, partial
import json
if __name__ == '__main__':
    # 参数设置
    sys.argv = [
        "notebook",  # 通常为脚本名称，随意填写
        "--epoch", "100",
        "--gpu", "0",
        "--n_dim","32",
        "--datatask", "S1_1"  ,#"S2_1",
        "--feat",'M',    ]
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Parser for VEDDI")
    parser.add_argument("--task_dir",type=str,default='/root/workspace/EmerGNN/TWOSIDES/data',help = 'directory to DrugBank dataset' )
    parser.add_argument("--datatask",type=str,default='S1_1',help = 'directory to EmerGNN dataset' )
    parser.add_argument("--batch_size",type=int,default=64,help = 'batch_size')
    parser.add_argument('--n_dim', type=int, default=32, help='set embedding dimension')
    parser.add_argument("--base_learning_rate",type=float,default=0.03,help='learning rate value')
    parser.add_argument('--epoch', type=int, default=100, help='number of training epochs')
    parser.add_argument('--weight_decay', type=float, default=7e-4, help='set weight decay value')
    parser.add_argument('--gpu',type=int,default=0,help='set the GPU use from this model,default is 0')
    parser.add_argument('--length', type=int, default=3, help='length of the path')
    parser.add_argument('--feat', type=str, default='M', help='type of the S1/S2-S3')
    parser.add_argument('--pretrain_model', type=str, default=None, help='pretain_model path')
    parser.add_argument('--file_name', type=str, default=None, help='file name for all ')
    parser.add_argument('--visualDDI', type=str, default=None, help='path of visualDDI ')
    parser.add_argument('--model_type', type=str, default=None, help='type of model')
    args = parser.parse_args()
    # args.all_ent
    torch.cuda.set_device(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    print("this time train for the ",args.file_name)
    # 数据集读取
    train_dataset = TSDDIDataset(args,data_type= 'train')

    Ent= train_dataset.eval_ent
    Rel= train_dataset.eval_rel
    VKG = train_dataset.vKG
    TKG = train_dataset.tKG
    valid_dataset = TSDDIDataset(args,data_type= 'valid')

    if "ATC" in args.datatask:
        VKG = TKG
        print("Now ATC,TKG = VKG")

    # 参数补充
    args.all_ent, args.all_rel = train_dataset.all_ent,train_dataset.all_rel
    args.eval_rel = train_dataset.eval_rel
    args.warmup_epoch= 2
    args.seed = 1234

    def run_model(params):
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        args.base_learning_rate = params['lr']
        args.weight_decay = params['lamb']
        args.batch_size = params['n_batch']
        args.model_type = params['model_type'] 
        batch_size = args.batch_size
    # 模型预训练权重加载
        if args.pretrain_model is None:
            # type load :load parmer
            print("load model form new type") 
            pretrain_model = torch.load('/root/workspace/visualDDI/ckpts/pretrain/vit-t-mae_8layers_patch16.pt')
            ermergnn = torch.load(f'/root/workspace/EmerGNN/TWOSIDES/{args.datatask}_saved_model.pt')
            model = Combin_Classifier(pretrain_model.encoder,parmer=args,Ent=Ent,num_classes=Rel).to(device)
            model.load_state_dict(ermergnn,strict=False)
        else :
            print("load model from pretrain")
            model = torch.load(args.pretrain_model, map_location='cuda')
        # 打印模型的所有层和对应的权重
        for name, param in model.named_parameters():
            if "transformer_block" in name:
                param.requires_grad = False    

        # 模型训练
        loss_fn = torch.nn.BCELoss()
        optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5*(math.cos(epoch / args.epoch * math.pi) + 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
        best_val_acc = 0
        k= 0
        optim.zero_grad()
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = False, num_workers = 4)

        valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle = False, num_workers = 4)
        
        try: 
            for e in range(args.epoch):
                if k > 5 :
                    break
                model.train()
                # for Combine
                train_dataset.shuffle_train()
                train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = True, num_workers = 4)
                KG = train_dataset.base_kg
                for img1, img2,d1,d2,label in tqdm(iter(train_dataloader)):
                    img1 = img1.to(device)
                    img2 = img2.to(device)
                    d1 = d1.to(device)
                    d2 = d2.to(device)
                    # print(np.array(label).shape)
                    label = torch.stack(label, dim=0)
                    label = label.T.to(device)
                    label = label.float()
                    # print(label.shape)
                    logits = model(img1, img2, d1, d2, KG)
                    
                    label_r  = label[:, :-1]
                    is_pos = label[:, -1] > 0    # shape (32,)
                    label_r = label_r.float()
                    scores = torch.sigmoid(logits)

                    label_pos = label_r[is_pos]  # shape (num_pos, 200)
                    label_neg = label_r[~is_pos]  # shape (num_neg, 200)
                    p_scores = scores[is_pos]    # shape (num_pos, 200)
                    n_scores = scores[~is_pos]   # shape (num_neg, 200)
                    pos_mask = label_pos > 0  # shape (num_pos, 200)
                    neg_mask = label_neg > 0 # shape (num_neg, 200)
                    p_scores = p_scores[pos_mask]    # shape (num_pos,)
                    n_scores = n_scores[neg_mask]    # shape (num_neg,)
                    p_labels = torch.ones_like(p_scores)
                    n_labels = torch.zeros_like(n_scores)
                    all_scores = torch.cat([p_scores, n_scores], dim=0).to(device)  # shape (num_pos + num_neg,)
                    all_labels = torch.cat([p_labels, n_labels], dim=0).to(device)  
                    loss = loss_fn(all_scores, all_labels)
                    loss.backward()
                    optim.step()
                    optim.zero_grad()
                scheduler.step()

                roc_auc, prc_auc, ap= eval_mae_loader_ts_mutil(model, valid_dataloader, device,VKG) 
                if prc_auc > best_val_acc:
                    k=0
                    best_val_acc = prc_auc

                else :
                    k+=1
        except RuntimeError as e:
            print(e)
            return 0
        print(best_val_acc)
        with open(os.path.join('/root/workspace/Fusionfordeal/TWOSIDES/results', 'hyperm_tune.txt'), 'a+') as f:
            f.write(json.dumps({
            **params, 
            "val_acc": best_val_acc
        }) + '\n')
        return -best_val_acc


    space = {
        "lr": hp.choice("lr", [1e-2, 1e-3, 1e-4]),
        "lamb": hp.choice("lamb", [1e-4, 5e-2, 1e-1]),
        "n_batch": hp.choice("n_batch", [32, 64,128]),
        "model_type": hp.choice("model_type",['A','B','C'])
    }

    trials = Trials()
    best = fmin(run_model, space, algo=partial(tpe.suggest, n_startup_jobs=40), max_evals=81, trials=trials)
    print(best)
    with open(os.path.join('/root/workspace/Fusionfordeal/TWOSIDES/results', 'hyperm_tune.txt'), 'a+') as f:
        f.write(json.dumps(best, default=str) + '\n')
                