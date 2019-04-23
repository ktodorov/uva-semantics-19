# Learning sentence representations from natural language inference data
This is repository for the individual project for Statistical Methods for Natural Language Semantics, University of Amsterdam, 2019

# Usage

## Training
    
You can train your model by running the `train.py` file

#### List of parameters

| Parameter     | type          | default value  | description |
| ------------- |:-------------:| --------------:|-------------|
| `--save_every_steps` | int | 1000 | Number of steps after which the model will be saved|
| `--log_every_steps` | int | 50 | Number of steps after which the current results will be printed|
| `--max_samples` | int | `None` | Number of samples to get from the datasets. If None, all samples will be used|
| `--snapshot_location` | str | `None` | Snapshot location from where to load the model|
| `--batch_size` | int | 64 | Batch size to use for the dataset |
| `--max_epochs` | int | 100 | Amount of max epochs of the training|
| `--learning_rate` | float | 0.1 | Learning rate |
| `--encoding_model` | str | `mean` | Model type for encoding sentences. Choose from `mean`, `uni-lstm`, `bi-lstm` and `bi-lstm-max-pool`|
| `--weight_decay` | float | 0.01 | "Weight decay for the optimizer")|


#### Example

```
python train.py --encoding_model=mean --weight_decay=0.1 --max_samples=10 --log_every_steps=100 --max_epochs=10000 --batch_size=64
Arguments:
save_every_steps : 1000
log_every_steps : 100
max_samples : 10
snapshot_location : None
batch_size : 64
max_epochs : 10000
learning_rate : 0.1
encoding_model : mean
weight_decay : 0.1
-----------------------------------------
Loading data...
Loading model...
Starting training...
  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Train/Micro-Accuracy  Train/Macro-Accuracy  Dev/Micro-Accuracy  Dev/Macro-Accuracy
     2     0         0     1/1         100% 1.104041 --------                  30.0000               30.0000 ------------------- -------------------
     2     0         1 ----------- ---------  ------ 1.109723                  30.0000               30.0000             30.0000             30.0000
     2     1         2 ----------- ---------  ------ 1.116314                  40.0000               40.0000             30.0000             30.0000
     2     2         3 ----------- ---------  ------ 1.118268                  40.0000               40.0000             30.0000             30.0000
     2     3         4 ----------- ---------  ------ 1.118569                  40.0000               40.0000             30.0000             30.0000
     ...
```

## Evaluation

#### List of parameters

| Parameter     | type          | default value  | description |
| ------------- |:-------------:| --------------:|-------------|
| `--model_path` | str | bi-lstm-max-pool model path | Path to the model which should be used for evaluation|
| `--eval_mode` | str | `snli` | Evaluation mode. Choose between `snli` and `senteval` |

#### Example
You can evaluate your model by running the `eval.py` file
```
python .\eval.py --eval_mode=senteval --model_path=MODEL_PATH
Loading model...Loaded
Starting evaluation...
{'MR': {'devacc': 77.87, 'acc': 76.63, 'ndev': 10685, 'ntest': 10685}, 'CR': {'devacc': 80.17, 'acc': 78.47, 'ndev': 3775, 'ntest': 3775}, 'SUBJ': {'devacc': 91.16, 'acc': 91.14, 'ndev': 10021, 'ntest': 10021}, 'MPQA': {'devacc': 87.51, 'acc': 87.57, 'ndev': 10606, 'ntest': 10606}, 'TREC': {'devacc': 73.59, 'acc': 82.6, 'ndev': 5452, 'ntest': 500}, 'SST2': {'devacc': 79.59, 'acc': 79.68, 'ndev': 872, 'ntest': 1821}}
```

```
python .\eval.py --eval_mode=snli --model_path=MODEL_PATH
Loading model...Loaded
Starting evaluation...
Loading data...
test macro accuracy: 37.67248376623377
test micro accuracy: 37.68322475570032
```
## Infer

#### List of parameters

| Parameter     | type          | default value  | description |
| ------------- |:-------------:| --------------:|-------------|
| `--model_path` | str | bi-lstm-max-pool model path | Path to the model which should be used for evaluation|

#### Example
You can test the inference of your model by running the `infer.py` file
After the model and the data are loaded, you will be asked to enter a premise and a hypothesis. After doing this, you the inference result will be calculated and presented

    python .\infer.py --eval_mode=senteval --model_path=MODEL_PATH
    Loading model...Loaded
    Loading data...Loaded
    Enter premise:
    the boy is doing a test
    Enter hypothesis:
    the boy is doing a test well
    The premise entails the hypothesis