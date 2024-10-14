import ast
import os
import os.path as osp

import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector

import copy
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem

def convert_smiles_to_graph(smiles_data):
    pyg_graph_list = []
    # check is str or list
    if isinstance(smiles_data, str):
        smiles_list = [smiles_data]
    else:
        smiles_list = smiles_data
    for smiles in tqdm(smiles_list, desc="Converting molecules to graphs"):
        graph = smiles2graph(smiles)
        g = Data()
        g.num_nodes = graph["num_nodes"]
        g.edge_index = torch.from_numpy(graph["edge_index"])

        del graph["num_nodes"]
        del graph["edge_index"]

        if graph["edge_feat"] is not None:
            g.edge_attr = torch.from_numpy(graph["edge_feat"])
            del graph["edge_feat"]

        if graph["node_feat"] is not None:
            g.x = torch.from_numpy(graph["node_feat"])
            del graph["node_feat"]

        try:
            g.fp = torch.tensor(graph["fp"], dtype=torch.int8).view(1, -1)
            del graph["fp"]
        except:
            pass

        pyg_graph_list.append(g)
    
    return pyg_graph_list


def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """
    try:
        mol = Chem.MolFromSmiles(smiles_string)
        # atoms
        atom_features_list = []
        # atom_label = []
        for atom in mol.GetAtoms():
            atom_features_list.append(atom_to_feature_vector(atom))
            # atom_label.append(atom.GetSymbol())

        x = np.array(atom_features_list, dtype=np.int64)
        # atom_label = np.array(atom_label, dtype=np.str)

        # bonds
        num_bond_features = 3  # bond type, bond stereo, is_conjugated
        if len(mol.GetBonds()) > 0:  # mol has bonds
            edges_list = []
            edge_features_list = []
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = bond_to_feature_vector(bond)
                # add edges in both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
            edge_index = np.array(edges_list, dtype=np.int64).T

            # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
            edge_attr = np.array(edge_features_list, dtype=np.int64)

        else:  # mol has no bonds
            edge_index = np.empty((2, 0), dtype=np.int64)
            edge_attr = np.empty((0, num_bond_features), dtype=np.int64)

        graph = dict()
        graph["edge_index"] = edge_index
        graph["edge_feat"] = edge_attr
        graph["node_feat"] = x
        graph["num_nodes"] = len(x)

        return graph

    except:
        return None