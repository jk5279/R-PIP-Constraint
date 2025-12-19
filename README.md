# R-PIP-constraint

PyTorch implementation of the **Proactive Infeasibility Prevention (PIP)** framework for the Traveling Salesman Problem with Time Windows (TSPTW), built on [POMO](https://github.com/yd-kwon/POMO).

## Overview

This repository contains a streamlined implementation of the PIP framework focused on TSPTW. PIP is an effective approach that integrates Lagrangian multipliers to enhance constraint awareness and introduces preventative infeasibility masking to proactively guide solution construction.

**Note:** This is a simplified version of the original [PIP-constraint](https://github.com/jieyibi/PIP-constraint) repository, containing only:
- TSPTW problem implementation
- POMO and POMO+PIP training and evaluation code
- TSPTW datasets and pretrained models

## Paper

For more details, please see the original paper: [Learning to Handle Complex Constraints for Vehicle Routing Problems](https://arxiv.org/pdf/2410.21066), accepted at NeurIPS 2024.

If you find this work useful, please cite:

```bibtex
@inproceedings{
    bi2024learning,
    title={Learning to Handle Complex Constraints for Vehicle Routing Problems},
    author={Bi, Jieyi and Ma, Yining and Zhou, Jianan and Song, Wen and Cao, 
    Zhiguang and Wu, Yaoxin and Zhang, Jie},
    booktitle = {Advances in Neural Information Processing Systems},
    year={2024}
}
```

## Installation

```bash
git clone https://github.com/jk5279/R-PIP-Constraint.git
cd R-PIP-constraint
conda create -n pip python=3.12
conda activate pip
pip3 install torch torchvision torchaudio  # Install PyTorch based on your CUDA version
pip install matplotlib tqdm pytz scikit-learn tensorflow tensorboard_logger pandas wandb
```

## Usage

### Generate Data

Generate TSPTW instances for evaluation:

```bash
cd POMO+PIP
# Default: --problem_size=50 --problem="TSPTW" --hardness="hard"
python generate_data.py --problem=TSPTW --problem_size=50 --hardness=hard
```

### Training

Train POMO* or POMO* + PIP models:

```bash
cd POMO+PIP

# Train POMO* baseline
python train.py --problem=TSPTW --hardness=hard --problem_size=50 --pomo_size=50

# Train POMO* + PIP
python train.py --problem=TSPTW --hardness=hard --problem_size=50 --pomo_size=50 --generate_PI_mask

# Resume training (optional)
python train.py --problem=TSPTW --hardness=hard --problem_size=50 --checkpoint=path/to/checkpoint.pt --resume_path=path/to/logs
```

### Evaluation

Evaluate trained models:

```bash
cd POMO+PIP

# Evaluate POMO* on provided dataset
python test.py --problem=TSPTW --hardness=hard --problem_size=50 --checkpoint=pretrained/TSPTW/tsptw50_hard/POMO_star/epoch-10000.pt

# Evaluate POMO* + PIP on provided dataset
python test.py --problem=TSPTW --hardness=hard --problem_size=50 --checkpoint=pretrained/TSPTW/tsptw50_hard/POMO_star_PIP/epoch-10000.pt --generate_PI_mask

# Evaluate on custom dataset
python test.py --test_set_path=path/to/test_data.pkl --checkpoint=path/to/model.pt --generate_PI_mask
# Optional: add --test_set_opt_sol_path=path/to/optimal_solutions.pkl to calculate optimality gap
```

**Note:** Adjust `--aug_batch_size` or `--test_batch_size` based on your GPU memory constraints.

## Repository Structure

```
R-PIP-constraint/
├── POMO+PIP/          # Main implementation directory
│   ├── envs/          # TSPTW environment
│   ├── models/        # Neural network models
│   ├── pretrained/    # Pretrained model checkpoints
│   ├── train.py       # Training script
│   ├── test.py        # Evaluation script
│   └── generate_data.py  # Data generation script
├── data/              # TSPTW datasets
│   └── TSPTW/
└── README.md
```

## Key Features

- **POMO Integration**: Built on top of the POMO (Policy Optimization with Multiple Optima) framework
- **PIP Framework**: Implements proactive infeasibility prevention through Lagrangian multipliers and masking
- **TSPTW Focus**: Specialized implementation for Traveling Salesman Problem with Time Windows
- **Pretrained Models**: Includes pretrained models for various problem sizes and hardness levels

## Acknowledgments

- [POMO](https://github.com/yd-kwon/POMO) - Base framework
- [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) - Multi-view Mixture of Experts architecture

## License

MIT License - see [LICENSE](LICENSE) file for details.
