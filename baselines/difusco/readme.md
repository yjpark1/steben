# DIFUSCO for STP
reference from https://github.com/Edward-Sun/DIFUSCO


## Environment

```bash
conda env create -f environment.yaml
conda activate diffusion
```

## generate dataset
make data to txt form
```bash
python dataloader.py --data_pth {synthetic_STP/ER_n10/} --output_dir {train_stp10.txt} --type {train/test}
```

## Train
```bash
python difusco/train.py --do_train --wandb_logger_name {sample} --training_split {data/train.txt} --validation_split {data/valid.txt} --test_split {data/test.txt} --test_cost {data/test_cost.txt}
```
for sampling
```bash
python difusco/train.py {same as above} --inference_diffusion_steps 10 --sequential_sampling 32
```

## Inference
unzip checkpoints
```bash
unzip checkpoints/ER/ER_10.zip
```
test with checkpoints path
```bash
python difusco/train.py --do_test --wandb_logger_name {sample_inference} --ckpt_path {checkpoints/ER/ER_10.ckpt}
```
