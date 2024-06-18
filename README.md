

#  
# SteBen: A Benchmark for Neural Combinatorial Optimization on the Steiner Tree Problem (STP)

![License](https://img.shields.io/badge/license-MIT-blue.svg)

## Overview

SteBen is a comprehensive benchmark dataset and framework for evaluating neural combinatorial optimization (NCO) methods on the Steiner Tree Problem (STP). This repository contains the source code for all baseline methods evaluated in our study, as well as scripts for generating the dataset and running experiments.

## Features

- Large-scale, high-quality datasets for training and testing NCO methods.
- Implementations of state-of-the-art NCO algorithms, including autoregressive and non-autoregressive, supervised and reinforcement learning paradigms.
- A gym-like environment for reinforcement learning experiments.
- Comprehensive evaluation metrics and scripts for reproducing our results.

## Dataset

The training and test datasets can be downloaded from the following link:
[Download Dataset](https://drive.google.com/drive/folders/1j_vuK-Mhv0mGoAXgF8FNVn1onONX-34T?usp=drive_link)

The datasets are stored in pickle files with networkx graph formats. For larger node sizes, the data is split across multiple pickle files.

## Getting Started
This repository contains baselines methods and dataset generator.
```
.
├── baselines # baseline methods
│   ├── am
│   ├── cherrypick
│   ├── common
│   └── heuristic
└── stpgen # dataset generator
    ├── datasets
    ├── envs
    └── solvers
```

### Prerequisites
- Python 3.7 or later
- Required Python packages (listed in `requirements.txt` or `readme.md`)
- SCIP-Jack (https://scipjack.zib.de, https://github.com/dRehfeldt/SCIPJack-Steiner-tree-solver)  *Build required!*

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yjpark1/steben.git
    cd steben
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Usage
Please refer each baseline folder.
- AM: `baselines/am/readme.md`
- Cherrypick: `baselines/cherrypick/readme.md`
- Pointer Network: `baselines/pointernet/readme.md`
- DIFUSCO: `baselines/difusco/readme.md`
- DIMES: `baselines/dimes/readme.md`



## License
This project is licensed under the MIT License - see the LICENSE file for details.