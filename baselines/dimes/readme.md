# A DIfferentiable MEta Solver (Dimes) Baseline

This directory contains the implementation of the DIMES algorithm.

## Usage

### Training

To train the model, use the **run_dimes_train.py** at the root directory of the main repository:
You could modify training schedule in **run_dimes_train.py**. 
Each training schedule is a dictionary with arguments. 
Other arguments not listed in the dictionary will be set to default value. 

```
python baselines/dimes/run_dimes_train.py
```

### Evaluation

To evaluate the trained model, use the **run_dimes_test.py**. 
You should specify **dataset_root**.
Each test case should include "run_name" and "eval_size", the "run_name" is the path for the trained network, and the eval_size is the graph size for evaluation. 
You could change the **NUM_WORKERS** for parallel evaluation.


```
python baselines/dimes/run_dimes_test.py
```

### Checkpoints

The checkpoints are located in the **dimes/checkpoints** folder. 
It contains the args.json which is used at the training time. 
You could reproduce the reported values using **run_dimes_test.py** with these checkpoints. 
