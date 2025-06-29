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
        
        if parmer.feat == 'E':
            self.embing_n = 4
            print('use feature E')
        else :
            self.embing_n = 2
            print('use feature M')
        self.embbing_tun =nn.ModuleList([nn.Linear(parmer.n_dim *self.embing_n, emb_dim) for _ in range(parmer.length)])
        # self.attention_layer = nn.MultiheadAttention(emb_dim, num_heads=3)
        for linear_layer in self.embbing_tun:
            trunc_normal_(linear_layer.weight, std=.02)
            if linear_layer.bias is not None:
                linear_layer.bias.data.fill_(0)

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
        self.head = torch.nn.Linear(emb_dim + parmer.n_dim *self.embing_n , 86)

        # EmerGNN
        self.parmer = parmer
        self.eval_ent = Ent
        self.eval_rel = num_classes
        self.all_ent = parmer.all_ent
        self.all_rel = 2*parmer.all_rel  + 1
        self.L = parmer.length
        self.n_dim = parmer.n_dim
        with open('/root/workspace/EmerGNN/DrugBank/data/DB_molecular_feats.pkl', 'rb') as f:
            x = pickle.load(f, encoding='utf-8')
            mfeat = []
            for y in x['Morgan_Features']:
                mfeat.append(y)
            mfeat = np.array(mfeat)

        if parmer.feat == 'M':
            self.ent_kg = nn.Parameter(torch.FloatTensor(mfeat), requires_grad=False)
            self.Went = nn.Linear(1024, parmer.n_dim)
            # self.Wr = nn.Linear(2 * parmer.n_dim, num_classes)
        else:
            self.ent_kg = nn.Embedding(Ent, parmer.n_dim)
            # self.Wr = nn.Linear(4 * parmer.n_dim, num_classes)
        # self.Wr = nn.Linear(2 * parmer.n_dim, num_classes)
        self.rel_kg = nn.ModuleList([nn.Embedding(self.all_rel, parmer.n_dim) for _ in range(self.L)])

        self.linear = nn.ModuleList([nn.Linear(parmer.n_dim, parmer.n_dim) for _ in range(self.L)])
        self.W = nn.Linear(parmer.n_dim, 1)
        self.act = nn.ReLU()

        self.relation_linear = nn.ModuleList([nn.Linear(2 * parmer.n_dim, 5) for _ in range(self.L)])
        self.attn_relation = nn.ModuleList([nn.Linear(5, self.all_rel) for _ in range(self.L)])

        # expert
        self.experts = nn.ModuleList([Expert(emb_dim + parmer.n_dim *self.embing_n, 512) for _ in range(4)])
        self.gate_network = nn.Sequential(
            nn.Linear(emb_dim + parmer.n_dim *self.embing_n, 4),
            nn.Softmax(dim=-1)  # 输出一个概率分布
        )  
        self.output_layer = nn.Linear(512, 86)

        
    def forward(self, img1, img2, head, tail, KG):
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

        def EmerGNN():
        # EmerGNN
            if self.parmer.feat == 'E':
                head_embed = self.ent_kg(head)
                tail_embed = self.ent_kg(tail)
            else:
                head_embed = self.Went(self.ent_kg[head])
                tail_embed = self.Went(self.ent_kg[tail])
            n_ent = self.all_ent
            
            tail_hid = torch.zeros(self.L, len(tail), self.n_dim).cuda()
            head_hid = torch.zeros(self.L, len(tail), self.n_dim).cuda()
            # propagate from u to v
            hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
            hiddens[head, torch.arange(len(head)).cuda()] = head_embed
            ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
            for l in range(self.L):
                hiddens = hiddens.view(n_ent, -1)
                relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
                relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
                rel_embed = self.rel_kg[l].weight       # (1, n_rel, n_dim)
                relation_input = relation_weight * rel_embed    #(batch_size, n_rel, n_dim)
                relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
                relation_input = relation_input.transpose(0,1).flatten(1)
                hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
                hiddens = hiddens.view(n_ent * len(head), -1)
                hiddens = self.linear[l](hiddens)
                hiddens = self.act(hiddens)
                # print(hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))].shape)
                tail_hid[l]=hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))]
            # tail_hid = hiddens.view(n_ent, len(tail), -1)[tail, torch.arange(len(tail))]

            # propagate from v to u
            hiddens = torch.FloatTensor(np.zeros((n_ent, len(head), self.n_dim))).cuda()
            hiddens[tail, torch.arange(len(tail)).cuda()] = tail_embed
            for l in range(self.L):
                hiddens = hiddens.view(n_ent, -1)
                relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
                relation_weight = torch.sigmoid(relation_weight).unsqueeze(2)
                rel_embed = self.rel_kg[l].weight       # (1, n_rel, n_dim)
                relation_input = relation_weight * rel_embed    #(batch_size, n_rel, n_dim)
                relation_input = relation_input.view(head_embed.size(0), -1, self.n_dim)
                relation_input = relation_input.transpose(0,1).flatten(1)
                hiddens = functional.generalized_rspmm(KG, relation_input, hiddens, sum='add', mul='mul')
                hiddens = hiddens.view(n_ent * len(head), -1)
                hiddens = self.linear[l](hiddens)
                hiddens = self.act(hiddens)
                head_hid[l] = hiddens.view(n_ent, len(head), -1)[head, torch.arange(len(head))]
            # head_hid = hiddens.view(n_ent, len(head), -1)[head, torch.arange(len(head))] 
            if self.parmer.feat == 'E':
                head_expanded = head_embed.unsqueeze(0).repeat(3, 1, 1)
                tail_expanded = tail_embed.unsqueeze(0).repeat(3, 1, 1)
                embeddings = torch.cat([head_expanded, head_hid, tail_hid, tail_expanded], dim=2)
            else:
                embeddings = torch.cat([head_hid, tail_hid], dim=2)
            return embeddings

        emergnn_future = torch.jit.fork(EmerGNN)
        emergnn_future = emergnn_future.wait()
        # print(emergnn_future.shape)
        features = self.transformer_block2(features)
        features = self.transformer_block3(features)
        features = self.layer_norm(self.transformer_block4(features))
        features = rearrange(features, 'b t c -> t b c')

        input_re = torch.cat([features[0],emergnn_future[2]],dim =1)
        gate_weights = self.gate_network(input_re)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(input_re) for expert in self.experts], dim=1)
        weighted_output = torch.sum(gate_weights.unsqueeze(-1) * expert_outputs, dim=1)

        output = self.output_layer(weighted_output)
        return output
    
    
    def visualize_forward(self, head, tail, rela, KG, num_beam=5, head_batch=True):
        assert head.numel() == 1 and head.ndim == 1

        if self.parmer.feat == 'E':
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        else:
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])

        edge_index = KG._indices().t()
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        step_weights = []
        edges_rel = KG._indices()[2]
        for l in range(self.L):
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight)
            edge_weights = relation_weight.squeeze(0)[edges_rel]
            step_weights.append(edge_weights)

        paths, weights = self.visualize(head, tail, edge_index, step_weights, num_beam=num_beam, head_batch=head_batch)
        
        return paths, weights


    def visualize(self, head, tail, edge_index, edge_weights, num_beam=5, head_batch=True):
        n_ent = self.all_ent
        inputs = torch.full((n_ent, num_beam), float('-inf')).cuda()
        if head_batch:
            inputs[head, 0] = 0
        else:
            inputs[tail, 0] = 0
        distances = []
        back_edges = []

        for i in range(len(edge_weights)):
            if head_batch:
                edge_mask = edge_index[:,0] != tail
            else:
                edge_mask = edge_index[:,0] != head
            edge_step = edge_index[edge_mask]
            node_in, node_out = edge_step.t()[:2]

            message = inputs[node_in] + edge_weights[i][edge_mask].unsqueeze(-1)  # [n_edge_step, num_beam]   this is the accumulated beam score
            msg_source = edge_step.unsqueeze(1).expand(-1, num_beam, -1)    # [n_edge_step, num_beam, 3]

            is_duplicate = torch.isclose(message.unsqueeze(-1), message.unsqueeze(-2)) & (msg_source.unsqueeze(-2) == msg_source.unsqueeze(-3)).all(dim=-1)
            is_duplicate = is_duplicate.float() - torch.arange(num_beam, dtype=torch.float).cuda() / num_beam     # [n_edge, num_beam, num_beam]
            prev_rank = is_duplicate.argmax(dim=-1,keepdim=True)    # [n_edge, num_beam, 1]
            msg_source = torch.cat([msg_source, prev_rank], dim=-1) # [n_edge, num_bearm, 4]

            node_out, order = node_out.sort()
            node_out_set = torch.unique(node_out)   
            # sort message w.r.t. node_out
            message = message[order].flatten()
            msg_source = msg_source[order].flatten(0, -2)

            size = scatter_add(torch.ones_like(node_out), node_out, dim_size=n_ent)     # [n_ent]       # how many in-edges per node
            sizes = size[node_out_set] * num_beam
            arange = torch.arange(len(sizes)).cuda()
            msg2out = arange.repeat_interleave(sizes)

            # deduplicate
            is_duplicate = (msg_source[1:] == msg_source[:-1]).all(dim=-1)
            is_duplicate = torch.cat([torch.zeros(1, dtype=torch.bool).cuda(), is_duplicate])
            message = message[~is_duplicate]
            msg_source = msg_source[~is_duplicate]
            msg2out = msg2out[~is_duplicate]
            size = scatter_add(torch.ones_like(msg2out), msg2out, dim_size=len(node_out_set))
            
            if not torch.isinf(message).all():
                distance, rel_index = functional.variadic_topk(message, size, k=num_beam)
                abs_index = rel_index + (size.cumsum(0) - size).unsqueeze(-1)
                back_edge = msg_source[abs_index]
                distance = distance.view(len(node_out_set), num_beam)
                back_edge = back_edge.view(len(node_out_set), num_beam, 4)
                distance = scatter_add(distance, node_out_set, dim=0, dim_size=n_ent)
                back_edge = scatter_add(back_edge, node_out_set, dim=0, dim_size=n_ent)
            else:
                distance = torch.full((n_ent, num_beam), float('-inf')).cuda()
                back_edge = torch.zeros(n_ent, num_beam, 4).long().cuda()
           
            distances.append(distance)
            back_edges.append(back_edge)
            inputs = distance

        # get topk_average_length
        k = num_beam
        paths = []
        weights = []

        for i in range(len(distances)):
            if head_batch:
                distance, order = distances[i][tail].flatten(0,-1).sort(descending=True)
                back_edge = back_edges[i][tail].flatten(0, -2)[order]
            else:
                distance, order = distances[i][head].flatten(0,-1).sort(descending=True)
                back_edge = back_edges[i][head].flatten(0, -2)[order]
            for d, (h,t,r,prev_rank) in zip(distance[:k].tolist(), back_edge[:k].tolist()):
                if d == float('-inf'):
                    break
                path = [(h,t,r)]
                for j in range(i-1, -1, -1):
                    h, t, r, prev_rank = back_edges[j][h, prev_rank].tolist()
                    path.append((h, t, r))
                paths.append(path[::-1])
                weights.append(d / len(path))

        if paths:
            weights, paths = zip(*sorted(zip(weights, paths), reverse=True)[:k])
        
        return paths, weights

    def get_attention_weights(self, head, tail, KG):
        if self.parmer.feat == 'E':
            head_embed = self.ent_kg(head)
            tail_embed = self.ent_kg(tail)
        else:
            head_embed = self.Went(self.ent_kg[head])
            tail_embed = self.Went(self.ent_kg[tail])
        
        outputs = []
        ht_embed = torch.cat([head_embed, tail_embed], dim=-1)
        for l in range(self.L):
            relation_weight = self.attn_relation[l](nn.ReLU()(self.relation_linear[l](ht_embed)))
            relation_weight = torch.sigmoid(relation_weight)
            outputs.append(relation_weight.cpu().data.numpy())
        return outputs
    
    