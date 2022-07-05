# Distributed Arcface Training in Pytorch

## Requirements

```
RAM > 10G
GPU
```
  
## How to Training

To train a model, run `train.py` with the path to the configs.  
The example commands below show how to run
distributed training.

```shell
python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py configs/wf_mbf
```

config.num_classes = 10572
config.num_image = 501196

## Download Datasets or Prepare Datasets  
- [WebFace0.5M](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view) (10572 IDs, 501196 images)

## Guide training on Kaggle
### 1. install network
```
!conda install -y gdown
!pip install -U --no-cache-dir gdown --pre
```
### 2. Clone backbone train
```
!git clone https://github.com/DuyNguDao/ArcFace_Pytorch.git
```
### 3. Download dataset and unzip
Note: change id gdrive of dataset
```
!gdown --id 1yaLoTdjybeLtXLsODjsJCqPRGvlk2a9k
!unzip /kaggle/working/faces_webface_112x112.zip
```
### 4. Change config
```
%load ./configs/wf4m_mbf.py
Change:
+ config.rec = "path of dataset" 
+ config.num_classes = xxx
+ config.image = xxx
+ config.epoch = xxx
+ config.batch_size = xxx
+ config.val_targets = ['lfw', "agedb_30"]: datatest (lfw, fcp-cp, agedb_30)
writefile:
%%writefile ./configs/wf4m_mbf.py
```
### 5. Install environments
```
!pip install -r requirement.txt
```
### 6. Training on GPU
```
!python -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 train.py configs/wf4m_mbf
```
### 7. Zip and download
```
!zip -r result.zip ./work_dirs
```


