# The Package for InfoAlign: Learning Molecular Representation in a Cell

**InfoAlign** is a package for learning molecular representations from bottleneck information, derived from molecular structures, cell morphology, and gene expressions. For more detailed information, please refer to our [paper](https://arxiv.org/abs/2406.12056v3).

This package uses a pretrained model based on the method described in the [paper](https://arxiv.org/abs/2406.12056v3). It takes molecules as input (e.g., a single SMILES string or a list of SMILES strings) and outputs their learned representations. These molecular representations can be applied to various downstream tasks, such as molecular property prediction.

For related projects by the main ML researcher and developer, visit: [https://github.com/liugangcode/InfoAlign](https://github.com/liugangcode/InfoAlign).

## Installation


```
conda create --name infoalign python=3.11.7
```

Install the package via pip:

```
pip install infoalign
```

## Usage

### Command Line Interface (CLI)

```bash
infoalign_predict --input {path_to_input_smiles.csv} \
                  --output {path_to_output.npy} \
                  --output-to-input-column  # Adds the representation as an additional column in the input CSV
```

The input CSV file should look like this:

```csv
SMILES
CNC[C@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@@H]1C
CNC[C@@H]1OCc2cnnn2CCCC(=O)N([C@H](C)CO)C[C@H]1C
C[C@H]1CN([C@@H](C)CO)C(=O)CCCn2cc(nn2)CO[C@@H]1CN(C)C(=O)CCC(F)(F)F
```


### Python API
```
from infoalign.representer import InfoAlignRepresenter

# Load the pretrained model
model = InfoAlignRepresenter(model_path='infoalign_model/pretrain.pt')

# Predict representation for a single SMILES string (returns a NumPy array)
one_rep = model.predict('CCC')

# Predict representations for a list of SMILES strings
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
