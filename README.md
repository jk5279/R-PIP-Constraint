# R-PIP-constraint

PyTorch implementation of the **Proactive Infeasibility Prevention (PIP)** framework for the Stochastic Traveling Salesman Problem with Time Windows (STSPTW), built on [POMO](https://github.com/yd-kwon/POMO).

## Overview

This repository contains a streamlined implementation of the PIP framework focused on STSPTW. PIP is an effective approach that integrates Lagrangian multipliers to enhance constraint awareness and introduces preventative infeasibility masking to proactively guide solution construction.

**Note:** This is a simplified version of the original [PIP-constraint](https://github.com/jieyibi/PIP-constraint) repository, containing only:
- STSPTW problem implementation
- POMO and POMO+PIP training and evaluation code
- STSPTW datasets and pretrained models

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

Generate STSPTW instances for evaluation. The generation method uses a unified α/β-based approach for all hardness levels:

```bash
cd POMO+PIP
# Default: --problem_size=50 --problem="STSPTW" --hardness="hard"
python generate_data.py --problem=STSPTW --problem_size=50 --hardness=hard
```

**Hardness Levels:**
- `easy`: α = 0.5, β = 0.75 (Wide time windows, loose constraints)
- `medium`: α = 0.3, β = 0.48 (Moderate constraints)
- `hard`: α = 0.1, β = 0.2 (Narrow time windows, tight constraints)

The α and β parameters control the time window duration range as a fraction of the time factor, where time windows are sampled uniformly from [α × time_factor, β × time_factor]. All hardness levels use the same generation method, making the system robust to distance function modifications. STSPTW adds stochastic noise U(0, √2) to distance calculations at each step, where noise is sampled independently for each distance measurement.

### Training

Train POMO* or POMO* + PIP models:

```bash
cd POMO+PIP

# Train POMO* baseline
python train.py --problem=STSPTW --hardness=hard --problem_size=50 --pomo_size=50

# Train POMO* + PIP
python train.py --problem=STSPTW --hardness=hard --problem_size=50 --pomo_size=50 --generate_PI_mask

# Train POMO* + PIP with custom buffer term (default: sqrt(2))
python train.py --problem=STSPTW --hardness=hard --problem_size=50 --pomo_size=50 --generate_PI_mask --pip_buffer=1.5

# Resume training (optional)
python train.py --problem=STSPTW --hardness=hard --problem_size=50 --checkpoint=path/to/checkpoint.pt --resume_path=path/to/logs
```

### Evaluation

Evaluate trained models:

```bash
cd POMO+PIP

# Evaluate POMO* on provided dataset
python test.py --problem=STSPTW --hardness=hard --problem_size=50 --checkpoint=pretrained/STSPTW/stsptw50_hard/POMO_star/epoch-10000.pt

# Evaluate POMO* + PIP on provided dataset
python test.py --problem=STSPTW --hardness=hard --problem_size=50 --checkpoint=pretrained/STSPTW/stsptw50_hard/POMO_star_PIP/epoch-10000.pt --generate_PI_mask

# Evaluate POMO* + PIP with custom buffer term (default: sqrt(2))
python test.py --problem=STSPTW --hardness=hard --problem_size=50 --checkpoint=pretrained/STSPTW/stsptw50_hard/POMO_star_PIP/epoch-10000.pt --generate_PI_mask --pip_buffer=1.5

# Evaluate on custom dataset
python test.py --test_set_path=path/to/test_data.pkl --checkpoint=path/to/model.pt --generate_PI_mask
# Optional: add --test_set_opt_sol_path=path/to/optimal_solutions.pkl to calculate optimality gap
```

**Note:** Adjust `--aug_batch_size` or `--test_batch_size` based on your GPU memory constraints.

## Repository Structure

```
R-PIP-constraint/
├── POMO+PIP/          # Main implementation directory
│   ├── envs/          # STSPTW environment
│   ├── models/        # Neural network models
│   ├── pretrained/    # Pretrained model checkpoints
│   ├── train.py       # Training script
│   ├── test.py        # Evaluation script
│   └── generate_data.py  # Data generation script
├── data/              # STSPTW datasets
│   └── STSPTW/
└── README.md
```

## Key Features

- **POMO Integration**: Built on top of the POMO (Policy Optimization with Multiple Optima) framework
- **PIP Framework**: Implements proactive infeasibility prevention through Lagrangian multipliers and masking
- **STSPTW Focus**: Specialized implementation for Stochastic Traveling Salesman Problem with Time Windows, where distance measurements include additive noise U(0, √2) sampled independently at each step
- **PIP Buffer Term**: Configurable buffer term (default: √2) added to distance calculations in PIP lookahead filtering to account for stochastic noise uncertainty. Use `--pip_buffer` to adjust the buffer value.
- **Unified Problem Generation**: All hardness levels use the same α/β-based generation method, making the system robust to distance function modifications
- **Pretrained Models**: Includes pretrained models for various problem sizes and hardness levels

## Acknowledgments

- [POMO](https://github.com/yd-kwon/POMO) - Base framework
- [Routing-MVMoE](https://github.com/RoyalSkye/Routing-MVMoE) - Multi-view Mixture of Experts architecture

## License

MIT License - see [LICENSE](LICENSE) file for details.
