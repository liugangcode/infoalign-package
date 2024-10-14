import torch

from .models.gnn import PretrainedGNN
from .utils import convert_smiles_to_graph
from torch_geometric.loader import DataLoader

class ModelConfig:
    def __init__(self):
        self.emb_dim = 300
        self.model = "gin-virtual"
        self.norm_layer = "batch_norm"
        self.num_layer = 5
        self.prior = 1.0e-09
        self.readout = "sum"
        self.threshold = 0.8
        self.walk_length = 4

class InfoAlignRepresenter:
    def __init__(self, model_path):
        self.config = ModelConfig()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.model = PretrainedGNN(
            gnn_type=self.config.model,
            num_layer=self.config.num_layer,
            emb_dim=self.config.emb_dim,
            graph_pooling=self.config.readout,
            norm_layer=self.config.norm_layer,
        ).to(device)
        self.model.load_pretrained_graph_encoder(model_path)
        self.model.freeze_graph_encoder()
        self.model.eval()

    def predict(self, smiles_data, batch_size=32):
        pyg_graph_list = convert_smiles_to_graph(smiles_data)
        loader = DataLoader(pyg_graph_list, batch_size=batch_size, shuffle=False)
        representations = []
        with torch.no_grad():
            for data in loader:
                data = data.to(self.device)
                representation = self.model(data)
                representations.append(representation)
        return torch.cat(representations, dim=0).cpu().numpy()