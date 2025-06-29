from cProfile import label
import os
import argparse
import sys
import math
from numpy.typing._generic_alias import NDArray
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from PIL import Image
from mae_model import *
import torchvision.transforms as transforms
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch
import torch.utils
from zmq import device
import warnings

from model import Combin_Classifier
import numpy as np
import pandas as pd
import json
from Grad_dataloader import TSDDIDataset,reshape_transform,DualInputViTWrapper
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

device = 'cuda' if torch.cuda.is_available() else 'cpu'



drug_bboxes = {
    # 143 : (185, 41, 213, 74),
    # 394 : (18, 128, 41, 151),
    # 482 : (109, 59, 138, 87),
    # 585 : (9, 46, 63, 122),
    # 878 : (7, 80, 98, 142),
    # 299 : (11, 98, 33, 111),
    # 137 : (21, 100, 50, 135),
    # 321 : (16, 25, 87, 114),
    # 1088: (11, 149, 42, 179),
    # 1089: (11, 110, 33, 119),
    # 1269: (8, 88, 101, 150),
    # 368 : (10, 143, 26, 165),
    # 436 : (11, 72, 36, 97),
}


def reshape_transform(tensor, height=28, width=14):
    # print(f"Original tensor shape: {tensor.shape}")
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))
    result = result.transpose(2, 3).transpose(1, 2)
    return result

def getcat_img(img1,img2):
    img1=img1.to(device)
    img2=img2.to(device)
    img1=img1.unsqueeze(0)
    img2=img2.unsqueeze(0)
    input_tensors = torch.cat((img1,img2),dim=2)
    return input_tensors


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

    args.kg_true = False


    model = torch.load('/root/workspace/Fusionfordeal/DrugBank/model/Expert_S0.pt', map_location='cuda') 
    # return
    jet_cmap = plt.get_cmap("jet")
    jet_colors = jet_cmap(np.linspace(0, 1, 256))  # 256个颜色点

    # 将前 N 个颜色（蓝色部分）改为白色
    N = 64  # 调整 N 的值（范围 0~256），控制替换的蓝色范围 只把其透明度提高 
    # jet_colors[:N] = [0, 0, 139/255,0.5]  
    jet_colors[:N, 3] = 0.30  # 设置透明度为0
    # 创建修改后的颜色映射
    custom_jet = LinearSegmentedColormap.from_list("custom_jet", jet_colors)

    file_path = "norm2"

    print(file_path)

    k=0

    model_pair = ['A','B']
    # for drug_id, (x1, y1, x2, y2) in drug_bboxes.items() :
    for drug_id,bboxes  in drug_bboxes.items():
        for function_num, (x1, y1, x2, y2) in enumerate(bboxes):
        # data reading 
            args.datatask = drug_id
            args.force_reload = True
            Case_parta_dataset =  TSDDIDataset(args,data_type= 'valid')
            PartA_dataloader = torch.utils.data.DataLoader(Case_parta_dataset,batch_size,shuffle = True, num_workers = 4)
            eval_ent= Case_parta_dataset.eval_ent
            eval_rel = Case_parta_dataset.eval_rel
            entity_vocab = Case_parta_dataset.id2entity
            relation_vocab = Case_parta_dataset.id2relation
            args.all_rel = Case_parta_dataset.all_rel
            TKG = Case_parta_dataset.TKG
            VKG = Case_parta_dataset.VKG

            args.force_reload = False
            Case_partb_dataset =  TSDDIDataset(args,data_type= 'test')
            PartB_dataloader = torch.utils.data.DataLoader(Case_partb_dataset,batch_size,shuffle = True, num_workers = 4)
            
            # model setting
            new_model = DualInputViTWrapper(model,eval_ent,eval_rel,args,entity_vocab,relation_vocab)
            new_model.eval()
            target_layers = [new_model.model.transformer_block1[0].norm2,new_model.model.transformer_block1[1].norm2,new_model.model.transformer_block2[0].norm2,new_model.model.transformer_block2[1].norm2,new_model.model.transformer_block3[0].norm2,new_model.model.transformer_block3[1].norm2,new_model.model.transformer_block4[0].norm2,new_model.model.transformer_block4[1].norm1]

            # grad setting
            Grad_list = []
            for i in range(8):
                cam = GradCAM(model=new_model, target_layers=[target_layers[i]], reshape_transform=reshape_transform)
                Grad_list.append(cam)
            Cam_all = GradCAM(model=new_model, target_layers=target_layers, reshape_transform=reshape_transform)
            Grad_list.append(Cam_all)

            # csv setting
            threshold_tx = []
            d1_list = []
            d2_list = []
            r_list = []    
            # continue
            for mode in model_pair : 
                if mode == 'A':
                    train_dataloader = PartA_dataloader
                    U_KG = VKG
                    print("PartA")
                else:
                    train_dataloader = PartB_dataloader
                    U_KG = TKG
                    print("PartB")

                for img1, img2,img1np,triplet in tqdm(iter(train_dataloader)):
                    img1np = img1np.squeeze(0)
                    # print(img1.shape)
                    img1np = np.array(img1np)
                    d1 = triplet[0][0]
                    d2 = triplet[0][1]
                    r= triplet[0][2]
                    triplet = triplet.to("cuda")
                    
                    # print(d1,d2,r)
                    img1case = np.zeros_like(img1np)
                    input_tensor = torch.cat((img1,img2),dim=2)
                    # we set too type of grad-cam : eigen_smooth =True or False
                    for layer_s in range(9):
                        # drug_cam = Cam_all(input_tensor=input_tensor,targets=[ClassifierOutputTarget(r)],eigen_smooth=True)
                        # drug_cam = Grad_list[layer_s](input_tensor=input_tensor,d1=d1.unsqueeze(0),d2=d2.unsqueeze(0),KG=U_KG,targets=[ClassifierOutputTarget(r)],eigen_smooth=True)
                        drug_cam = Grad_list[layer_s](input_tensor=input_tensor,d1=d1.unsqueeze(0),d2=d2.unsqueeze(0),KG=U_KG,targets=[ClassifierOutputTarget(r)])
                        drug_cam = drug_cam[0,:224,:]

                        threshold = []
                        threshold_pce = [99,97,95,93,90,85,80,75]
                        for i in threshold_pce: #即 95,90,85,80,75
                            threshold.append(np.percentile(drug_cam, i))

                        anchor_region = drug_cam[y1:y2, x1:x2]    # 认为锚框中有20%的点为 top10%  10% -top5% 5% top-3% 3% top-1%
                        total_pixels = anchor_region.size
                        threshold_temple = []
                        #统计锚框中大于阈值threshold的像素点数量
                        threshold_temple.append(layer_s)
                        for th in threshold:
                            above_threshold = anchor_region > th
                            num_above = np.sum(above_threshold)
                            percentage = num_above / total_pixels * 100
                            threshold_temple.append(percentage)  #形状为(4,)
                        threshold_temple.append(anchor_region.mean())
                        threshold_temple.append(drug_cam.mean())
                        threshold_temple.append(drug_cam.std())
                        d1_list.append(d1.item())
                        d2_list.append(d2.item())
                        r_list.append(r.item())
                        threshold_tx.append(threshold_temple)  #threshold_tx 的形状为 （N,4） 
                        drug_cam = (drug_cam - drug_cam.min()) / (drug_cam.max() - drug_cam.min())
                        Image.fromarray(show_cam_on_image(img1np,drug_cam,use_rgb=True)).save(f"/root/workspace/Fusionfordeal/visualDDiXAI/Grad_Case/Case_drug/{d1}/{function_num+1}/{d2}_{r}_{layer_s}.png")

                # 把threshold_tx转换为DataFrame
            threshold_tx = pd.DataFrame(threshold_tx)
            # 设置列名
            threshold_tx.columns = ['target_layers','threshold_99', 'threshold_97','threshold_95', 'threshold_93','threshold_90', 'threshold_85', 'threshold_80','threshold_75','abox-mean',"grad-mean","grad-std"]
            # 将数据添加到DataFrame中
            threshold_tx['d1'] = d1_list
            threshold_tx['d2'] = d2_list
            threshold_tx['r'] = r_list
            # #调整列的顺序
            threshold_tx = threshold_tx[['d1', 'd2', 'r','target_layers','threshold_99', 'threshold_97','threshold_95','threshold_93', 'threshold_90', 'threshold_85', 'threshold_80','threshold_75','abox-mean',"grad-mean","grad-std"]]
            
            threshold_tx.to_csv("/root/workspace/Fusionfordeal/visualDDiXAI/Grad_Case/Case_drug"+ f'/{d1}/'+f'threshold_layers_drug{function_num}.csv',index = False  )
