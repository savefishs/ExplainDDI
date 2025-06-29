from cProfile import label
import os
import argparse
import sys
import math
from numpy.typing._generic_alias import NDArray
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
# from dataloader import MAEDDIDataset
# warnings.filterwarnings("ignore")
from PIL import Image
from mae_model import *
# from utils import setup_seed, eval_mae_loader
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.utils
from zmq import device
import warnings
from mae_model import *
from model import Combin_Classifier
import numpy as np
import pandas as pd
import json
from Grad_dataloader import TSDDIDataset,reshape_transform,DualInputViTWrapper
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


device = 'cuda' if torch.cuda.is_available() else 'cpu'


if __name__ == '__main__':
    sys.argv = [
        "notebook",  # 通常为脚本名称，随意填写
        "--batch_size", "1",   #"32",
        "--gpu", "0",
        "--n_dim","64",
        "--pretrain_model","/root/workspace/Fusionfordeal/DrugBank/model/Expert_S0.pt",
        "--feat",'M',
        "--length","3",
        "--datatask","S0",
    ]
    # x1,y1,x2,y2 =9, 46, 63, 122
    
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
    parser.add_argument('--file_name', type=str, default=None, help='file name for all ')
    parser.add_argument('--visualDDI', type=str, default=None, help='path of visualDDI ')
    parser.add_argument('--all_rel', type=int, default=None, help='all_rel ')
    parser.add_argument('--mode', type=str, default=None, help='all_rel ')
    args = parser.parse_args()
    batch_size = args.batch_size

    load_batch_size =  batch_size
    model = torch.load('/root/workspace/Fusionfordeal/DrugBank/model/Expert_S0.pt', map_location='cuda')

    args.kg_true = True
                      
    dataset = TSDDIDataset(args,data_type= 'valid')
    data_loader = torch.utils.data.DataLoader(dataset,batch_size,shuffle = False, num_workers = 4)

    eval_ent= dataset.eval_ent
    eval_rel = dataset.eval_rel
    entity_vocab = dataset.id2entity
    relation_vocab = dataset.id2relation
    args.all_rel = dataset.all_rel

    VKG = dataset.VKG
    # triplets = dataset.samples
    print(VKG.shape)
    new_model = DualInputViTWrapper(model,eval_ent,eval_rel,args,entity_vocab,relation_vocab)
    new_model.eval()

    path_data = []
    # k = 0
    for img1, img2,img1np,triplet in tqdm(iter(data_loader)):
        triplet = triplet.to("cuda")
        # print(triplets)
        d1 = triplet[0][0]
        d2 = triplet[0][1]
        r= triplet[0][2]
        outputs = new_model.Path_Statistics(triplet[0],VKG)
        path_data.extend(outputs)
        # k +=1
        # if k > 100 :
        #     break
    
    all_path = pd.DataFrame(path_data)
    
    all_path.to_csv('/root/workspace/Fusionfordeal/visualDDiXAI/path_noDDI.csv',index=False)

