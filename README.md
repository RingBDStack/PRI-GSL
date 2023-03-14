# PRI-GSL
This repo is the code implement of ["Self-organization Preserved Graph Structure Learning with Principle of Relevant Information"](AAAI 2023)



# Requirements
* `networkx==2.8.4`
* `numpy==1.22.4`
* `scikit-learn==1.1.1`
* `scipy==1.8.1`
* ` torch==1.11.0`
* `torch-cluster==1.6.0`
* `torch-geometric==2.0.4`
* `torch-scatter==2.0.9`
* `torch-sparse==0.6.13`
* `torch-spline-conv==1.2.1`
# Datasets

Cora, Citeseer are provided by [IDGL](https://github.com/hugochan/IDGL)
The other datasets (i.e., Photo,  Chameleon, Squirrel, Actor) are provided by [pyg](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

# Get Started
## Overview
```
main.py # getting start

core/models/ # core model files
core/models/graphwave.py # graph wave functions
core/models/PRILoss.py # Principle of Relevant Information functions
core/models/graph_clf.py # model configurations 

layers/ # basic gnn model and train steps
utils/ # auxiliary tools
```
## Run the code
```
git clone ...
cd PRI-GSL
python main.py --dataset_name <dataset_name>
```
For instance,
```
python main.py --dataset_name cora
```

## Repoduced the results
We provide the hyper-parameters settings in the 'config' folder, you can type the command to test the performance.
```
python main.py --dataset_name <dataset_name> --use_config 1
--config_path <you_path_to_config> 
```
For instance,
```
python main.py --dataset_name citeseer --use_config 1 --config_path ./config/ 
```  
```
>>> Train Epoch: [0 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 0]: 0h 00m 17s <> <>
Validation Epoch 0 -- Loss: 1.79150 | NLOSS = -1.79150 | ACC = 0.20000
Saved model to ***/Cond/IDGL/out/citeseer/idgl/whbb77mq
!!! Updated: 
NLOSS = -1.79150
ACC = 0.20000


>>> Train Epoch: [500 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 500]: 0h 33m 01s <> <>
Validation Epoch 500 -- Loss: 2.49713 | NLOSS = -2.49713 | ACC = 0.52778
Saved model to ***/Cond/IDGL/out/citeseer/idgl/whbb77mq
!!! Updated: 
NLOSS = -2.49713
ACC = 0.52778


>>> Train Epoch: [1000 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 1000]: 0h 37m 46s <> <>
Validation Epoch 1000 -- Loss: 2.28878 | NLOSS = -2.28878 | ACC = 0.58889

>>> Train Epoch: [1500 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 1500]: 0h 43m 10s <> <>
Validation Epoch 1500 -- Loss: 2.22111 | NLOSS = -2.22111 | ACC = 0.61667

>>> Train Epoch: [2000 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 2000]: 0h 42m 43s <> <>
Validation Epoch 2000 -- Loss: 2.19373 | NLOSS = -2.19373 | ACC = 0.63333

>>> Train Epoch: [2500 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 2500]: 0h 48m 23s <> <>
Validation Epoch 2500 -- Loss: 2.18154 | NLOSS = -2.18154 | ACC = 0.62222

>>> Train Epoch: [3000 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 3000]: 0h 43m 19s <> <>
Validation Epoch 3000 -- Loss: 2.16413 | NLOSS = -2.16413 | ACC = 0.64444

>>> Train Epoch: [3500 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 3500]: 0h 41m 44s <> <>
Validation Epoch 3500 -- Loss: 2.16961 | NLOSS = -2.16961 | ACC = 0.63889

>>> Train Epoch: [4000 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 4000]: 0h 42m 22s <> <>
Validation Epoch 4000 -- Loss: 2.17059 | NLOSS = -2.17059 | ACC = 0.65556

>>> Train Epoch: [4500 / 5000]
<> <> Timer [Train] <> <> Interval [Validation Epoch 4500]: 0h 41m 48s <> <>
Validation Epoch 4500 -- Loss: 2.18263 | NLOSS = -2.18263 | ACC = 0.66667
Finished Training: ***/Cond/IDGL/out/citeseer/idgl/whbb77mq
Training time: 24305.59

<<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> 
Best epoch = 3865; 
NLOSS = -2.15257
ACC = 0.65556

 <<<<<<<<<<<<<<<< MODEL SUMMARY >>>>>>>>>>>>>>>> 
[test] | test_exs = 3027 | step: [1 / 1] | NLOSS = -2.07726 | ACC = 0.68484
Finished Testing: ***/Cond/IDGL/out/citeseer/idgl/whbb77mq
Testing time: 0.97
```
For some datasets, we further freeze the params, you can check  the dir 'saved_params', the usage is:
```
python main.py --dataset_name <dataset_name> --dirname <you_path_to_params>
```
For instance,
```
python main.py --dataset_name citeseer --dirname ./saved_params/citeseer
```
```
Restoring best model
[ Loading saved model ./saved_params/citeseer/params.saved ]
[ Multi-perspective weighted_cosine GraphLearner: 1 ]
[ Graph Learner metric type: weighted_cosine ]
[ Multi-perspective weighted_cosine GraphLearner: 1 ]
[ Graph Learner metric type: weighted_cosine ]
[ Graph Learner ]
<> <> <> Starting Timer [Test] <> <> <>
epoch 1: updateing feature.
[test] | test_exs = 3027 | step: [1 / 1] | NLOSS = -2.07726 | ACC = 0.68484
Final score on the testing set: 0.68484
```
## If you find this repo helpful
Please cite our paper as:
```
@inproceedings{sun2023-PRI,
  author    = {Qingyun Sun and
               Jianxin Li and
               Beining Yang and
               Xingcheng Fu and
               Hao Peng and
               Philip S. Yu},
  title     = {Self-organization Preserved Graph Structure Learning with Principle
               of Relevant Information},
  booktitle = { Conference on Artificial Intelligence, {AAAI}
               2023},
  year      = {2023}
}
```