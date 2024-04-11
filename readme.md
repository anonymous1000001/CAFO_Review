

# CAFO: Feature-Centric Explanation on Time Series Classification
![Overview](/figures/overview.png)


## Introduction
CAFO (Channel Attention and Feature Orthogonalization) is a novel framework for feature-centric explanation in multivariate time series (MTS) classification. We provide the official implementation for CAFO.

## Contents
- `channelattention`: Contains the `depthwise` channel attention module, along other channel attention modules used in our paper.
- `conf`: Configuration files based on `hydra` library. 
- `dataset`: PyTorch datasets.
- `featureorderstudy`: Constructs feature ranks, summarize rank results based on run files.
- `importance_visualizer`: Scripts for visualizing feature importance (CWRI, GI).
- `lightning_models`: `PyTorch Lightning` based trainers.
- `loss`: The QR Regularizer loss
- `model`: The model architectures used for CAFO.
- `scripts`: Utility scripts for running experiments, preprocessing, etc.
- `notebook`: Contains all the visualization files needed to reproduce our work.

## Installation
CAFO was tested on the below dependency. You can simply run the following command. CAFO was tested on two `CUDA 11.6` and `CUDA 10.1`
```
pip install -r requirements.txt
```

The main dependency are the following 
```
torch==1.8.1
pytorch-lightning==1.6.5 #Trainer module
pyts==0.12.0 # for RP, GAF encoding
```


## How to Run?
There are two main functions `main_cafo` and `main_ce`. `main_cafo` is the main function to run CAFO module with QR decomposition based loss. The main module runs the `lightning_model_cls_cafo_qr.py` trainer. The below is an example of running the `gilon_activity` task, with `mlpmixer` model. You can set which validation fold to use with `task.validation_cv_num`. `gamma` controls the regularization parameter.


### Commands for main_cafo.py
``` sh
python  main_cafo.py task=gilon_activity model=mlpmixer 
        task.validation_cv_num=0\
        gpu_id=1 lightning_model=lightning_model_cls_cafo_qr exp_num=9999 \
        channelattention.expansion_filter_num=3 loss.gamma=0.5 loss=orthoregularizer\
        dataset.batch_size=512 add_random_channel_idx=false seed=42 
```


### Commands for main_ce.py
``` sh
python main_ce.py task=gilon_activity model=mlpmixer task.validation_cv_num=0\
            gpu_id=1 lightning_model=lightning_model_cls_baseline exp_num=9998 \
            channelattention=depthwise channelattention.expansion_filter_num=3\
            dataset.batch_size=512 add_random_channel_idx=false seed=42&
```

Running the above commands will automatically save the output files to `outputs/$data_name/$exp_num/` directory.

