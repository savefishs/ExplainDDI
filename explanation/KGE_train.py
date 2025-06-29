import os
import argparse
import sys
import math
import torch
import torch.utils
import importlib
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from zmq import device
import warnings
from Vutils import eval_mae_loader_KGE, setup_seed, eval_mae_loader,evaluate_multi_class_F1
import random
from mae_model import *
from KGE_model import Combin_Classifier
import numpy as np
from KGE_dataloader import MAEDDIDataset, TSDDIDataset

if __name__ == '__main__':
    # 参数设置
    sys.argv = [
        "notebook",  # 通常为脚本名称，随意填写
        "--batch_size", "128",   #"32",
        "--base_learning_rate", "0.005",
        "--epoch", "100",
        "--weight_decay","5e-2" ,  #"1e-4", 
        "--gpu", "0",
        "--n_dim","64",
        # "--pretrain_model","/root/workspace/Fusionfordeal/DrugBank/model/KGE_S0_RotatE.pt",
        "--datatask", "S0"  ,#"S2_1",
        "--feat",'E',
        "--embedding_path","/root/workspace/KGE/models",
        "--file_name","KGE_S0_KGNN_2"  # 命名规则 模型类型 ： Combin_type_x / visualDDI  数据集：S1 / S2
    ]
    # KGE_model = "RGCN" 
    KGE_model = "KGNN"
    # KGE_model = "TransE" 

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description="Parser for VEDDI")
    parser.add_argument("--task_dir",type=str,default='/root/workspace/EmerGNN/DrugBank/data',help = 'directory to DrugBank dataset' )
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
    parser.add_argument('--embedding_path', type=str, default=None, help='embedding_path')
    parser.add_argument('--file_name', type=str, default=None, help='file name for all ')
    parser.add_argument('--visualDDI', type=str, default=None, help='path of visualDDI ')
    args = parser.parse_args()
    # args.all_ent
    torch.cuda.set_device(args.gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = args.batch_size

    print("this time train for the ",args.file_name)
    # 数据集读取
    

    train_dataset = MAEDDIDataset(args,datatype= 'train', KGE_mode= KGE_model)

    Ent= train_dataset.eval_ent
    Rel= train_dataset.eval_rel
    print(Ent,Rel)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = True, num_workers = 4)

    valid_dataset = MAEDDIDataset(args,datatype= 'valid',KGE_mode =KGE_model)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size,shuffle = True, num_workers = 4)

    test_dataset = MAEDDIDataset(args,datatype= 'test',KGE_mode =KGE_model)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size,shuffle = True, num_workers = 4)

    # 参数补充
    args.all_ent, args.all_rel = train_dataset.all_ent,train_dataset.all_rel
    args.warmup_epoch= 5

    # 模型预训练权重加载
    if args.pretrain_model is None:
        pretrain_model = torch.load('/root/workspace/visualDDI/ckpts/pretrain/vit-t-mae_8layers_patch16.pt')
        model = Combin_Classifier(pretrain_model.encoder,parmer=args,Ent=Ent,num_classes=Rel).to(device)

    else :
        print("load model from pretrain")
        model = torch.load(args.pretrain_model, map_location='cuda')
    # 打印模型的所有层和对应的权重
    for name, param in model.named_parameters():
        if "transformer_block" in name:
            param.requires_grad = False    
        # print(f"Layer name: {name}, Parameter shape: {param.shape}") 
    
    # 模型训练
    writer = SummaryWriter(os.path.join('logs', './type', args.file_name+'-cls'))
    ##visualDDI op
    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5*(math.cos(epoch / args.epoch * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)
    ## EmerGNN
    # optim= torch.optim.Adam(model.parameters(), lr=args.base_learning_rate, weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='max')
    best_val_acc = 0
    step_count = 0
    k= 0
    optim.zero_grad()

    for e in range(args.epoch):
        model.train()
        losses = []
        acces = []
        pre_list = []
        label_list = []
        # for Combine
        train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size,shuffle = True, num_workers = 4)
        for img1, img2,d1,d2,label in tqdm(iter(train_dataloader)):
            img1 = img1.to(device)
            img2 = img2.to(device)
            d1 = d1.to(device)
            d2 = d2.to(device)
            label = label.to(device)
            logits = model(img1, img2, d1, d2)
            ## 使用VisualDDI的loss
            loss = loss_fn(logits,label)
            loss.backward()
            optim.step()
            optim.zero_grad()
            losses.append(loss.item())
            if step_count %50 == 0:

                batch_f1 = evaluate_multi_class_F1(list(torch.softmax(logits, dim=1).cpu().detach().numpy()),list(label.cpu().detach().numpy()))
                writer.add_scalar('cls/batch_loss', loss.item(), global_step=step_count)
                writer.add_scalar('cls/batch_F1', batch_f1, global_step=step_count)
            pred = torch.softmax(logits, dim=1)
            pre_list.extend(list(pred.cpu().detach().numpy()))
            label_list.extend(list(label.cpu().detach().numpy()))
            step_count += 1
        train_F1 = evaluate_multi_class_F1(pre_list,label_list)
        scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        print(f'In epoch {e}, average training loss is {avg_train_loss}')

        f1, recall, precision, acc = eval_mae_loader_KGE(model, valid_dataloader, device) 

        out_str = f"epoch {e} Valid: f1: {f1}  recall :{recall} precision :{precision}  acc: {acc} "  
        
        with open(os.path.join('results', args.file_name+f'_lr_{args.base_learning_rate}_batch_{args.batch_size}_lamb_{args.weight_decay}.txt'), 'a+') as f:
            f.write(out_str + '\n')
        k+=1
        if f1 > best_val_acc:
            k=0
            best_val_acc = f1
            print(f'saving best model with F1 {best_val_acc} at {e} epoch!')   
            # print(f1, recall, precision, acc)    
            torch.save(model,'./model/'+ args.file_name+'.pt')



        writer.add_scalar('cls/Train_loss', avg_train_loss, global_step=e)
        writer.add_scalar('cls/Train_F1', train_F1, global_step=e)
        writer.add_scalar('cls/Valid_F1', f1, global_step=e)
        print(f'acc:{acc}, f1:{f1}, recall:{recall}, pre:{precision}')
        if k >  30:
            break

    model = torch.load('./model/'+ args.file_name+'.pt', map_location='cuda')
    with open(os.path.join('results', args.file_name+f'_lr_{args.base_learning_rate}_batch_{args.batch_size}_lamb_{args.weight_decay}.txt'), 'a+') as f:
        f.write(str(eval_mae_loader_KGE(model, test_dataloader, device)))

