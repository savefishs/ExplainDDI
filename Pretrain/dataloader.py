import os
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
import json
class PretrainDataset(Dataset):
    def __init__(self, args):
        super().__init__()
        
        self.samples = args.scale
        self.range_data = range(1, args.scale+1)

        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        
    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        d1 = index+1
        d2 = random.choice(self.range_data)

        img1 = Image.open(f'datasets/pretrain/img/{d1}.png')
        img2 = Image.open(f'datasets/pretrain/img/{d2}.png')
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2
