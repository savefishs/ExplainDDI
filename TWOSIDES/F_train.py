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
from Vutils import setup_seed, eval_mae_loader_ts_mutil
import random
from mae_model import *
from model import Combin_Classifier
import numpy as np
from dataloader import TSDDIDataset

if __name__ == '__main__':
    # 参数设置
    sys.argv = [
        "notebook",  # 通常为脚本名称，随意填写
        "--batch_size", "32",   
        "--base_learning_rate", "0.001",
        "--epoch", "50",
        "--weight_decay","1e-4" ,  
        "--gpu", "0",
        "--n_dim","32",
        "--datatask", "S1_1"  ,
        "--feat",'M',
        "--file_name","S1_1" 
    ]
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Parser for VEDDI")
    parser.add_argument("--task_dir",type=str,default='../dataset/TWOSIDES/data',help = 'directory to DrugBank dataset' )
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
    args = parser.parse_args()
    # args.all_ent
    torch.cuda.set_device(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size

    print("this time train for the ",args.file_name)
    # 数据集读取
    train_dataset = TSDDIDataset(args,data_type= 'train')
    Ent= train_dataset.eval_ent
    Rel= train_dataset.eval_rel
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = False, num_workers = 4)
    KG = train_dataset.base_kg

    valid_dataset = TSDDIDataset(args,data_type= 'valid')
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle = False, num_workers = 4)

    test_dataset = TSDDIDataset(args,data_type= 'test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle = False, num_workers = 4)

    
    VKG = train_dataset.vKG
    TKG = train_dataset.tKG
    if "ATC" in args.datatask:
        VKG = TKG
        print("Now ATC,TKG = VKG")

    # 参数补充
    args.all_ent, args.all_rel = train_dataset.all_ent,train_dataset.all_rel
    args.eval_rel = train_dataset.eval_rel
    args.warmup_epoch= 2

    # 模型预训练权重加载
    if args.pretrain_model is None:
        print("load model form new type") 
        pretrain_model = torch.load('./model/vit-t-mae_8layers_patch16.pt')
        ermergnn = torch.load(f'./model/{args.datatask}_saved_model.pt')
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
    writer = SummaryWriter(os.path.join('logs', './type', args.file_name+'-cls'))

    loss_fn = torch.nn.BCELoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5*(math.cos(epoch / args.epoch * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
 
    best_val_acc = 0
    step_count = 0
    k= 0
    optim.zero_grad()
    # best_test_acc =0.94
    for e in range(args.epoch):
        model.train()
        losses = []
        acces = []
        pre_list = []
        label_list = []
        train_dataset.shuffle_train()
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = True, num_workers = 4)
        KG = train_dataset.base_kg
        for img1, img2,d1,d2,label in tqdm(iter(train_dataloader)):
            img1 = img1.to(device)
            img2 = img2.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)
            label = torch.stack(label, dim=0)
            label = label.T.to(device)
            label = label.float()
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

            # 拼接
            all_scores = torch.cat([p_scores, n_scores], dim=0).to(device)  # shape (num_pos + num_neg,)
            all_labels = torch.cat([p_labels, n_labels], dim=0).to(device)  

            loss = loss_fn(all_scores, all_labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            if step_count %50 == 0:
                writer.add_scalar('cls/batch_loss', loss.item(), global_step=step_count)
            step_count += 1
        scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_train_loss}')

        roc_auc, prc_auc, ap= eval_mae_loader_ts_mutil(model, valid_dataloader, device,VKG) 

        out_str = f"epoch {e} Valid: roc_auc: {roc_auc}  prc_auc :{prc_auc} ap :{ap}   "  
        
        with open(os.path.join('results', args.file_name+f'_lr_{args.base_learning_rate}_batch_{args.batch_size}_lamb_{args.weight_decay}.txt'), 'a+') as f:
            f.write(out_str + '\n')
        k+=1
        if prc_auc > best_val_acc:
            k=0
            best_val_acc = prc_auc
            print(f'saving best model with aupr {prc_auc} at {e} epoch!')   
            torch.save(model,'./model/'+ args.file_name+'.pt')



        writer.add_scalar('cls/Train_loss', avg_train_loss, global_step=e)
        writer.add_scalar('cls/Valid_aupr',prc_auc , global_step=e)
        print(f'roc_auc: {roc_auc}  prc_auc :{prc_auc} ap :{ap}')
        if k >  10:
            break

    model = torch.load('./model/'+ args.file_name+'.pt', map_location='cuda')
    with open(os.path.join('results', args.file_name+f'_lr_{args.base_learning_rate}_batch_{args.batch_size}_lamb_{args.weight_decay}.txt'), 'a+') as f:
        f.write(str(eval_mae_loader_ts_mutil(model, test_dataloader, device,TKG)))

