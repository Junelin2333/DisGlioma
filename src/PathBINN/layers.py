import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from typing import Optional
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_geometric.nn import SAGPooling
from torch_geometric.utils import dense_to_sparse


class PathwayMoE(nn.Module):
    def __init__(self, embed_size, num_pathway, num_prompts):
        super(PathwayMoE, self).__init__()

        self.embed_dim = embed_size
        self.num_pathway = num_pathway
        self.num_prompts = num_prompts
        #------ init PathwayMoE -----

        self.attn2 = nn.MultiheadAttention(self.embed_dim, num_heads=1, batch_first=True)  
        self.norm2 = nn.RMSNorm(self.embed_dim) 

        self.prompt = nn.Parameter(torch.empty((self.num_pathway, self.num_prompts, self.embed_dim)), requires_grad=True)
        self.register_buffer('initialized', torch.zeros(num_pathway, dtype=torch.bool))
        
        self.indicate = nn.Sequential(
            nn.Linear(self.embed_dim, self.num_prompts, bias=True),
            nn.Softmax(dim=-1),
        ) 

        self._init()
        #------ finish init pathway -----

    def _init(self):
        # Initialize the prompts with a normal distribution
        nn.init.normal_(self.prompt, mean=0.01, std=0.02)
    
    def forward(self, gene_embed, value_embed, router):

        bsz, seq_len, dim = gene_embed.shape
        output = []
        for i in range(self.num_pathway):
            gene_mask = router[0, i, ...]
            idx = torch.nonzero(gene_mask, as_tuple=True)[0]
            
            symbol_x = gene_embed[:, idx, :] 
            value_x = value_embed[:, idx, :]
            gene_x = value_x + symbol_x

            if not self.initialized[i].item():
                with torch.no_grad():
                    noise = torch.randn(1, self.num_prompts, self.embed_dim, device=self.prompt.device)
                    avg_mean = torch.mean(gene_x, dim=1, keepdim=True) # [1, 1, dim]
                    
                    repeated_v_mean = noise * 0.01 + torch.mean(avg_mean, dim=0, keepdim=False)
                    self.prompt[i, :, :].copy_(repeated_v_mean.squeeze(0))
                    self.initialized[i] = True
            
            prompt = repeat(self.prompt[i, :, :], 'p d -> b p d', b=bsz)
     
            attn, _ = self.attn2.forward(prompt, gene_x, gene_x)
            attn = self.norm2(attn)
            prompt = prompt + attn

            value_mean = torch.mean(value_x, dim=1, keepdim=False)
            indicate_score = self.indicate(value_mean)

            Po = prompt * indicate_score.unsqueeze(dim=-1)
            Po = torch.sum(Po, dim=1, keepdim=True)
            output.append(Po)

        output = torch.cat(output, dim=1)
        return output
    

class GraphConv(torch.nn.Module):
    def __init__(self, in_channels, hidden_dim, ratio=0.5):
        super(GraphConv, self).__init__()

        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.gelu = nn.GELU()
        self.drop = nn.Dropout(ratio)
        self.conv2 = SAGEConv(hidden_dim, in_channels)

    def forward(self, graph, adj_matrix):
        bsz, _, _ = graph.shape
        output = []

        for i in range(bsz):
            x = graph[i,...]
            adj = adj_matrix[i,...]

            edge_index, edge_weight = dense_to_sparse(adj)
            layer1 = self.gelu(self.conv1.forward(x, edge_index))
            
            edge_index, edge_weight = dense_to_sparse(adj)
            layer1 = self.conv2.forward(layer1, edge_index,)
            layer1 = self.drop(layer1)

            output.append(layer1)

        output = torch.stack(output, dim=0)
        return output
    
