# The Package for InfoAlign: Learning Molecular Representation in a Cell

**InfoAlign** is a package for learning molecular representations from bottleneck information, derived from molecular structures, cell morphology, and gene expressions. For more detailed information, please refer to our [paper](https://arxiv.org/abs/2406.12056v3).

This package uses a pretrained model based on the method described in the [paper](https://arxiv.org/abs/2406.12056v3). It takes molecules as input (e.g., a single SMILES string or a list of SMILES strings) and outputs their learned representations. These molecular representations can be applied to various downstream tasks, such as molecular property prediction.

For related projects by the main ML researcher and developer, visit: [https://github.com/liugangcode/InfoAlign](https://github.com/liugangcode/InfoAlign).

## Installation

Install the package via pip:

```
pip install infoalign
```

## Usage

### Command Line Interface (CLI)
```
infoalign_pred --input {path_to_input_smiles.csv} 
               --output {path_to_output.npy} 
               --output-to-input-column  # This adds the representation to the input CSV as an additional column
```

### Python API
```
from infoalign.representer import InfoAlignRepresenter

model = InfoAlignRepresenter(model_path='infoalign_model/pretrain.pt')

# For a single SMILES string
one_rep = model.predict('CCC')

# For a list of SMILES strings
two_reps = model.predict(['CCC', 'CCC'])
```

## Citation

If you find this repository helpful, please cite our paper:

```
@article{liu2024learning,
  title={Learning Molecular Representation in a Cell},
  author={Liu, Gang and Seal, Srijit and Arevalo, John and Liang, Zhenwen and Carpenter, Anne E and Jiang, Meng and Singh, Shantanu},
  journal={arXiv preprint arXiv:2406.12056},
  year={2024}
}
```

## Acknowledgement

This project template was adapted from: [https://github.com/lwaekfjlk/python-project-template](https://github.com/lwaekfjlk/python-project-template). We thank the authors for their open-source contribution.
