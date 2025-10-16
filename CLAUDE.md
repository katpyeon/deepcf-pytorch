# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of DeepCF (AAAI 2019), a deep collaborative filtering framework for recommendation systems. The codebase implements three model variants:

1. **DMF (Deep Matrix Factorization)** - CFNet-rl: Uses element-wise product of user/item latent vectors
2. **MLP (Multi-Layer Perceptron)** - CFNet-ml: Uses concatenation of user/item embeddings + MLP
3. **CFNet** - Fusion model: Combines DMF + MLP for best performance

Original implementation: https://github.com/familyld/DeepCF

## Key Commands

### Setup
```bash
# Install dependencies
pip install -r requirement.txt
```

### Training Models
Models are trained via Jupyter notebooks, not CLI scripts:
- `cfnet_rl/dmf_train.ipynb` - Train DMF standalone
- `cfnet_ml/mlp_train.ipynb` - Train MLP standalone
- `cfnet/cfnet_train.ipynb` - Train CFNet (fusion)
- `cornac/cornac_eval.ipynb` - Evaluate all models with baselines

### Data Sampling (for quick testing)
```bash
# Generate sample datasets with N users
jupyter notebook data_sampling.ipynb
```
Creates `datasets/ml-1m-sample{N}.{train,test,negative}` files.

## Architecture

### Module Organization

- **common/** - Shared utilities across all models
  - `data_utils.py` - Data loading (`load_deepcf_data`), format conversion (`deepcf_to_uir`), matrix utilities (`get_train_matrix`)
  - `train_utils.py` - Negative sampling (`get_train_instances`), PyTorch Dataset wrapper
  - `eval_utils.py` - Model evaluation with leave-one-out protocol

- **cfnet_rl/** - DMF model (representation learning)
  - `dmf_model.py` - Core PyTorch model with dual towers
  - `cornac_dmf_wrapper.py` - Cornac framework integration

- **cfnet_ml/** - MLP model (metric learning)
  - `mlp_model.py` - Core PyTorch model with concatenation
  - `cornac_mlp_wrapper.py` - Cornac framework integration

- **cfnet/** - CFNet fusion model
  - `cfnet_model.py` - Combines DMF + MLP, supports pretrained weights

- **cornac/** - Evaluation framework
  - Compares models against baselines (NeuMF, ItemPop)
  - Outputs standardized metrics (Precision@K, Recall@K, NDCG@K, MAP)

### Model Architecture Details

**DMF (cfnet_rl/dmf_model.py)**
- Input: Multi-hot encoding of user-item interactions
- User Tower: `num_items → userlayers[0] → ... → userlayers[-1]`
- Item Tower: `num_users → itemlayers[0] → ... → itemlayers[-1]`
- Fusion: Element-wise product → Linear → Sigmoid
- Default layers: user=[512, 64], item=[1024, 64]

**MLP (cfnet_ml/mlp_model.py)**
- Input: Multi-hot encoding of user-item interactions
- User Embedding: `num_items → layers[0]//2`
- Item Embedding: `num_users → layers[0]//2`
- Fusion: Concatenate([user, item]) → MLP layers (ReLU) → Linear → Sigmoid
- Default layers: [512, 256, 128, 64]

**CFNet (cfnet/cfnet_model.py)**
- Parallel execution of DMF and MLP
- Concatenates DMF vector + MLP vector
- Final prediction layer combines both representations
- Supports two training modes:
  1. **Pretrain (recommended)**: Load weights from pre-trained DMF and MLP models
  2. **From scratch**: Random initialization and joint training

### Data Format

DeepCF uses three tab-separated files per dataset:
- `{dataset}.train.rating`: `userID\titemID\trating\ttimestamp`
- `{dataset}.test.rating`: Same format as train
- `{dataset}.test.negative`: `(userID,itemID)\tnegItem1\tnegItem2\t...` (99 negatives per test item for leave-one-out evaluation)

All user/item IDs are integers. Ratings are binarized (1.0 for positive interactions).

### Key Implementation Details

1. **Multi-hot Encoding**: Models use interaction history as input features
   - `user_matrix[u]` = binary vector of items user u interacted with
   - `item_matrix[i]` = binary vector of users who interacted with item i
   - Stored as PyTorch buffers (non-trainable)

2. **Training Data**: Uses negative sampling
   - For each positive (user, item) pair, sample `NUM_NEG` negative items
   - Negatives are randomly sampled from items the user hasn't interacted with
   - Default: `NUM_NEG = 4`

3. **Evaluation**: Leave-one-out protocol
   - Each test rating is evaluated against 99 random negatives
   - Metrics: HR@10 (Hit Ratio), NDCG@10 (Normalized Discounted Cumulative Gain)

4. **Weight Initialization**: Lecun normal initialization
   - `std = sqrt(1.0 / fan_in)`
   - Biases initialized to zero

5. **CFNet Pretrain Loading**:
   - DMF and MLP prediction layer weights are concatenated
   - Final prediction bias = 0.5 * (dmf_bias + mlp_bias)
   - Prediction weights scaled by 0.5 to balance contributions

## Common Development Workflows

### Adding a New Model
1. Create model class in new directory (e.g., `new_model/`)
2. Inherit from `nn.Module`, implement `__init__` and `forward`
3. Use `common/data_utils.py` for data loading
4. Use `common/train_utils.py` for training data generation
5. Create Cornac wrapper for evaluation (optional)

### Training with Custom Hyperparameters
Open the relevant notebook (`*_train.ipynb`), modify the configuration cell:
```python
# Example from dmf_train.ipynb Cell 4
DATASET = 'ml-1m-sample100'  # or 'ml-1m' for full dataset
USERLAYERS = [512, 64]
ITEMLAYERS = [1024, 64]
EPOCHS = 20
BATCH_SIZE = 256
NUM_NEG = 4
LEARNING_RATE = 0.0001
```

### Using Pretrained Models with CFNet
1. Train DMF: Run `cfnet_rl/dmf_train.ipynb`, saves to `pretrain/CFNet-rl.pth`
2. Train MLP: Run `cfnet_ml/mlp_train.ipynb`, saves to `pretrain/CFNet-ml.pth`
3. Train CFNet: Run `cfnet/cfnet_train.ipynb` with pretrain paths set

### Evaluating Against Baselines
Use `cornac/cornac_eval.ipynb` to compare multiple models:
- Automatically loads Cornac dataset from DeepCF format
- Runs all specified models (DMF, MLP, CFNet, NeuMF, ItemPop, etc.)
- Outputs formatted comparison table with standard metrics

## Data Files

- `datasets/` - Contains DeepCF format datasets
  - Full: `ml-1m.*` (MovieLens 1M, not included by default)
  - Samples: `ml-1m-sample{20,100,500,1000}.*` (for testing)
- `pretrain/` - Saved model checkpoints (.pth files)
- `docs/` - Documentation files

## Dependencies

Core dependencies (from requirement.txt):
- `torch`, `torchvision`, `torchaudio` - PyTorch framework
- `numpy`, `pandas`, `scikit-learn` - Data processing
- `matplotlib` - Visualization
- `cornac` - Recommender system framework for evaluation
