

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
### Directories
This repository contains baselines methods and dataset generator.
```
.
├── baselines # baseline methods
│   ├── am
│   ├── cherrypick
│   ├── common
│   ├── difusco
│   ├── dimes
│   ├── dimes_tsp
│   ├── heuristic
│   └── pointer_network
│
└── stpgen # dataset generator
    ├── datasets
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

### Using SCIP-Jack Solver
If you want to use SCIP-Jack solver, you need to specify the `path_scipjack` for `SCIPJackRunner` in `steben/stpgen/solvers/scipjack.py`.

```python
class SCIPJackRunner:
    """run SCIPJack command in terminal 
    """
    def __init__(self, path_setting=None, timelimit=None) -> None:
        self.path_setting = 'stpsolver/settingsfile.set' if path_setting is None else path_setting
        self.timelimit = timelimit
        self._set_scipjack_path()
    
    def _set_scipjack_path(self):
        os_ = platform.platform().lower()
        if 'mac' in os_:
            path_scipjack = 'stpsolver/build_mac_arm/scip/bin/applications/scipstp'
        elif 'linux' in os_:
            path_scipjack = 'stpsolver/build_linux/scip/bin/applications/scipstp'
        else:
            raise ValueError(f"Unsupported OS {os_}")
        self.path_scipjack = path_scipjack
```

### Usage
#### Datasets generation
```bash
python generate_train_data.py
python generate_test_data.py
```

#### Baselines
Please refer each baseline folder.
- AM: `baselines/am/readme.md`
- Cherrypick: `baselines/cherrypick/readme.md`
- Pointer Network: `baselines/pointernet/readme.md`
- DIFUSCO: `baselines/difusco/readme.md`
- DIMES: `baselines/dimes/readme.md`



## License
This project is licensed under the MIT License - see the LICENSE file for details.