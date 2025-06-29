from math import e
import torch
import timm
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np
from torchdrug.layers import functional
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torch_scatter import scatter_add

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
# from models import EmerGNN
# from layers import Gating

def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape ## patch_num, 2, 192
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes, remain_T

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)

        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img1, img2):
        patches1 = self.patchify(img1)
        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')

        patches2 = self.patchify(img2)
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')
        
        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches1, forward_indexes1, backward_indexes1, remain_T = self.shuffle(patches1)
        patches2 = take_indexes(patches2, forward_indexes1)

        patches2 = patches2[remain_T:]
        patches = torch.cat([patches1, patches2], dim=0)
        # patches2, forward_indexes2, backward_indexes2 = self.shuffle(patches2)


        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)        
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes1
    
class Expert(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Expert, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return F.relu(self.fc(x))

class Combin_Classifier(torch.nn.Module):
    
    
    def __init__(self, encoder : MAE_Encoder, parmer, Ent,num_classes =65, emb_dim=192, num_head = 3):
        super(Combin_Classifier, self).__init__()
        # combining 
        # self.combing_token =  torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
    
        self.embing_n =2
        # VisuallDII
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.shuffle = encoder.shuffle
        self.transformer_block1 = nn.Sequential(*[block for block in encoder.transformer[:2]])
        self.transformer_block2 = nn.Sequential(*[block for block in encoder.transformer[2:4]])
        self.transformer_block3 = nn.Sequential(*[block for block in encoder.transformer[4:6]])
        self.transformer_block4 = nn.Sequential(*[block for block in encoder.transformer[6:]]) 
        self.layer_norm = encoder.layer_norm

        self.n_dim = parmer.n_dim
        if parmer.feat == 'M':
            self.Went = nn.Linear(2048, self.n_dim)
        else:
            self.Went = nn.Linear(1024, parmer.n_dim)
            self.Went = nn.Linear(200, self.n_dim)
        # expert
        self.experts = nn.ModuleList([Expert(emb_dim + self.n_dim *self.embing_n, 512) for _ in range(4)])
        self.gate_network = nn.Sequential(
            nn.Linear(emb_dim + self.n_dim *self.embing_n, 4),
            nn.Softmax(dim=-1)  # 输出一个概率分布
        )  
        self.output_layer = nn.Linear(512, 86)

        
    def forward(self, img1, img2, head, tail):
        # Visuall DDI
        
        patches1 = self.patchify(img1)
        patches2 = self.patchify(img2)

        patches1 = rearrange(patches1, 'b c h w -> (h w) b c')
        patches2 = rearrange(patches2, 'b c h w -> (h w) b c')

        patches1 = patches1 + self.pos_embedding
        patches2 = patches2 + self.pos_embedding

        patches = torch.cat([patches1, patches2], dim=0)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)

        patches = rearrange(patches, 't b c -> b t c')
        features = self.transformer_block1(patches)
        # features = rearrange(features, 'b t c -> t b c')
        # return features

        features = self.transformer_block2(features)
        features = self.transformer_block3(features)
        features = self.layer_norm(self.transformer_block4(features))
        features = rearrange(features, 'b t c -> t b c')
        # if entry embbing = 64
        # entry_embbing = torch.cat([head,tail],dim=1)
        # else
        entry_embbing = torch.cat([self.Went(head),self.Went(tail)],dim=1)
        # entry_embbing = torch.cat([head,tail],dim=1)
        input_re = torch.cat([features[0],entry_embbing],dim =1)
        gate_weights = self.gate_network(input_re)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(input_re) for expert in self.experts], dim=1)
        weighted_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)
        output = self.output_layer(weighted_output)
        return output