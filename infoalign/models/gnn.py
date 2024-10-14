import os
import torch
import torch.nn as nn

from torch.distributions import Normal, Independent
from torch.nn.functional import softplus
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool

from .conv import GNN_node, GNN_node_Virtualnode

class PretrainedGNN(nn.Module):
    def __init__(
        self,
        num_layer=5,
        emb_dim=300,
        gnn_type="gin",
        drop_ratio=0.5,
        graph_pooling="max",
        norm_layer="batch_norm",
    ):
        super(PretrainedGNN, self).__init__()

        ### GNN to generate node embeddings
        gnn_name = gnn_type.split("-")[0]
        if "virtual" in gnn_type:
            self.graph_encoder = GNN_node_Virtualnode(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        else:
            self.graph_encoder = GNN_node(
                num_layer,
                emb_dim,
                JK="last",
                drop_ratio=drop_ratio,
                residual=True,
                gnn_name=gnn_name,
                norm_layer=norm_layer,
            )
        ### Poolinwg function to generate whole-graph embeddings
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        else:
            raise ValueError("Invalid graph pooling type.")
        
        self.dist_net = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, 2 * emb_dim, bias=True)
        )
    
    def forward(self, batched_data):
        h_node, _ = self.graph_encoder(batched_data)
        h_graph = self.pool(h_node, batched_data.batch)

        mu, _ = self.dist_net(h_graph).chunk(2, dim=1)
        return mu

    def load_pretrained_graph_encoder(self, model_path):
        if not os.path.exists(model_path):
            try:
                from huggingface_hub import hf_hub_download
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                hf_hub_download(repo_id="liuganghuggingface/InfoAlign-Pretrained", 
                                filename="pretrain.pt", 
                                local_dir=os.path.dirname(model_path),
                                local_dir_use_symlinks=False)
                print('Model downloaded successfully to', model_path)
            except Exception as e:
                raise RuntimeError(f"Failed to download the model: {str(e)}")

        saved_state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        graph_encoder_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('graph_encoder.')}
        graph_encoder_state_dict = {key.replace('graph_encoder.', ''): value for key, value in graph_encoder_state_dict.items()}
        self.graph_encoder.load_state_dict(graph_encoder_state_dict)
        # Load dist_net state dictionary
        dist_net_state_dict = {key: value for key, value in saved_state_dict.items() if key.startswith('dist_net.')}
        dist_net_state_dict = {key.replace('dist_net.', ''): value for key, value in dist_net_state_dict.items()}
        self.dist_net.load_state_dict(dist_net_state_dict)
        self.freeze_graph_encoder()

    def freeze_graph_encoder(self):
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.dist_net.parameters():
            param.requires_grad = False