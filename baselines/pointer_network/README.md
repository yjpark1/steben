# Pointer Network for Steiner Tree Problem

This project implements a Pointer Network for solving STP using PyTorch.

## Model Architecture

The Pointer Network model consists of an encoder and a decoder with attention mechanism. It takes graph data as input and predicts the solution to the graph problem.

## Usage

### Run

To run the model, run the following command:

```
python main.py --tag TEST --lr 0.0001 --n_epochs 300 --emb_size 512 --batch_size 2048 --seed 1234 --size 20 --g er --total_data_size 1000000 --phase gaussian
```

Modify the hyperparameters and settings according to your requirements.

### Training

To train the model, run the following command:

```
train(model, args.n_epochs, device, load_id, args.g, args.size, args.batch_size, args.lr, workers=4, tag=args.tag, total_data_size=args.total_data_size, phase=phase)
```

Modify the hyperparameters and settings according to your requirements.

### Evaluation

To evaluate the trained model, run the following command:

```
python test.py --load_id ER-10 --batch_size 2048 --size 20 --g er
```

Replace `load_id` with the trained model's checkpoint directory.

The script will automatically load the best checkpoint based on the `load_id`.

## Configuration

The `run.py` file serves as the entry point of the program. It parses the command-line arguments, sets up the device (CPU or GPU), and calls the appropriate functions for training and evaluation.

## Checkpoints

During training, the model checkpoints are saved in the `checkpoint` directory. Each checkpoint is identified by a timestamp in the format `YYYYMMDD_HHMMSS`. The best checkpoint based on the validation performance is also saved.