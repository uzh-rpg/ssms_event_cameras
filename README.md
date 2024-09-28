# [CVPR'24 Spotlight] State Space Models for Event Cameras
<p align="center">
 <a href="https://www.youtube.com/watch?v=WRZZJn6Me9M">
  <img src="https://github.com/uzh-rpg/ssms_event_cameras/blob/master/scripts/zubic_cvpr2024_youtube.png" alt="youtube_video"/>
 </a>
</p>

This is the official PyTorch implementation of the CVPR 2024 paper [State Space Models for Event Cameras](https://arxiv.org/abs/2402.15584).

### 🖼️ Check Out Our Poster! 🖼️ [here](https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/Zubic_CVPR24_poster.pdf)

## :white_check_mark: Updates
* **` June. 14th, 2024`**: Everything is updated! Poster released! Check it above.
* **` June. 6st, 2024`**: Video released! To watch our video, simply click on the YouTube play button above.
* **` June. 1st, 2024`**: Our CVPR conference paper has also been accepted as a Spotlight presentation at "The 3rd Workshop on Transformers for Vision (T4V)."
* **` April. 19th, 2024`**: The code along with the best checkpoints is released! The poster and video will be released shortly before CVPR 2024.

## Citation
If you find this work and/or code useful, please cite our paper:

```bibtex
@InProceedings{Zubic_2024_CVPR,
    author    = {Zubic, Nikola and Gehrig, Mathias and Scaramuzza, Davide},
    title     = {State Space Models for Event Cameras},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {5819-5828}
}
```

## SSM-ViT
- S5 model used in our SSM-ViT pipeline can be seen [here](https://github.com/uzh-rpg/ssms_event_cameras/tree/master/RVT/models/layers/s5).
- In particular, S5 is used instead of RNN in a 4-stage hierarchical ViT backbone, and its forward function is exposed [here](https://github.com/uzh-rpg/ssms_event_cameras/blob/master/RVT/models/detection/recurrent_backbone/maxvit_rnn.py#L245). What is nice about this approach is that we do not need a 'for' loop over sequence dimension, but instead we employ a parallel scanning algorithm. This model assumes that a hidden state is being carried over.
- For a model that is standalone, and can be used for any sequence modeling problem, one does not use by default this formulation where we carry on the hidden state. The implementation is the same as the original JAX implementation and can be downloaded in zip format from [ssms_event_cameras/RVT/models/s5.zip](https://github.com/uzh-rpg/ssms_event_cameras/raw/master/RVT/models/s5.zip).

## Installation
### Conda
We highly recommend using [Mambaforge](https://github.com/conda-forge/miniforge#mambaforge) to reduce the installation time.
```Bash
conda create -y -n events_signals python=3.11
conda activate events_signals
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install lightning wandb pandas plotly opencv-python tabulate pycocotools bbox-visualizer StrEnum hydra-core einops torchdata tqdm numba h5py hdf5plugin lovely-tensors tensorboardX pykeops scikit-learn          
```

## Required Data
To evaluate or train the S5-ViT model, you will need to download the required preprocessed datasets:

<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">1 Mpx</th>
<th valign="bottom">Gen1</th>
<tr><td align="left">pre-processed dataset</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen4.tar">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/RVT/datasets/preprocessed/gen1.tar">download</a></td>
</tr>
<tr><td align="left">crc32</td>
<td align="center"><tt>c5ec7c38</tt></td>
<td align="center"><tt>5acab6f3</tt></td>
</tr>
</tbody></table>

You may also pre-process the dataset yourself by following the [instructions](https://github.com/NikolaZubic/ssms_event_cameras/blob/master/RVT/scripts/genx/README.md).

## Pre-trained Checkpoints
### 1 Mpx
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">S5-ViT-Base</th>
<th valign="bottom">S5-ViT-Small</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/gen4_base.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/gen4_small.ckpt">download</a></td>
</tr>
</tbody></table>

### Gen1
<table><tbody>
<th valign="bottom"></th>
<th valign="bottom">S5-ViT-Base</th>
<th valign="bottom">S5-ViT-Small</th>
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/gen1_base.ckpt">download</a></td>
<td align="center"><a href="https://download.ifi.uzh.ch/rpg/CVPR24_Zubic/gen1_small.ckpt">download</a></td>
</tr>
</tbody></table>

## Evaluation
- Evaluation scripts with concrete parameters that we trained our models can be seen [here](https://github.com/uzh-rpg/ssms_event_cameras/tree/master/scripts).
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set `CKPT_PATH` to the path of the *correct* checkpoint matching the choice of the model and dataset
- Set
  - `MDL_CFG=base` or
  - `MDL_CFG=small`
      
  to load either the base or small model configuration.
- Set `GPU_ID` to the PCI BUS ID of the GPU that you want to use. e.g. `GPU_ID=0`.
  Only a single GPU is supported for evaluation
### 1 Mpx
```Bash
python RVT/validation.py dataset=gen4 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=1 hardware.gpus=${GPU_ID} +experiment/gen4="${MDL_CFG}.yaml" \
batch_size.eval=12 model.postprocess.confidence_threshold=0.001
```
### Gen1
```Bash
python RVT/validation.py dataset=gen1 dataset.path=${DATA_DIR} checkpoint=${CKPT_PATH} \
use_test_set=1 hardware.gpus=${GPU_ID} +experiment/gen1="${MDL_CFG}.yaml" \
batch_size.eval=8 model.postprocess.confidence_threshold=0.001
```
We set the same batch size for the evaluation and training: 12 for the 1 Mpx dataset, and 8 for the Gen1 dataset.

## Evaluation results
Evaluation should give the same results as shown below:
- 47.7 and 47.8 mAP on Gen1 and 1 Mpx datasets for the base model, and
- 46.6 and 46.5 mAP on Gen1 and 1 Mpx datasets for the small model.
<p align="center">
  <img src="https://github.com/uzh-rpg/ssms_event_cameras/blob/master/scripts/checkpoints.png">
</p>

## Training
- Set `DATA_DIR` as the path to either the 1 Mpx or Gen1 dataset directory
- Set
    - `MDL_CFG=base` or
    - `MDL_CFG=small`
  
  to load either the base or the small configuration.
- Set `GPU_IDS` to the PCI BUS IDs of the GPUs that you want to use. e.g. `GPU_IDS=[0,1]` for using GPU 0 and 1.
  **Using a list of IDS will enable single-node multi-GPU training.**
  Pay attention to the batch size which is defined per GPU.
- Set `BATCH_SIZE_PER_GPU` such that the effective batch size is matching the parameters below.
  The **effective batch size** is (batch size per GPU)*(number of GPUs).
- If you would like to change the effective batch size, we found the following learning rate scaling to work well for 
all models on both datasets:
  
  `lr = 2e-4 * sqrt(effective_batch_size/8)`.
- The training code uses [W&B](https://wandb.ai/) for logging during the training.
Hence, we assume that you have a W&B account. 
  - The training script below will create a new project called `ssms_event_cameras`. Adapt the project name and group name if necessary.
 
### 1 Mpx
- The effective batch size for the 1 Mpx training is 12.
- For training the model on 1 Mpx dataset, we need 2x A100 80 GB GPUs and we use 12 workers per GPU for training and 4 workers per GPU for evaluation:
```Bash
GPU_IDS=[0,1]
BATCH_SIZE_PER_GPU=6
TRAIN_WORKERS_PER_GPU=12
EVAL_WORKERS_PER_GPU=4
python RVT/train.py model=rnndet dataset=gen4 dataset.path=${DATA_DIR} wandb.project_name=ssms_event_cameras \
wandb.group_name=1mpx +experiment/gen4="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```
If you for example want to execute the training on 4 GPUs simply adapt `GPU_IDS` and `BATCH_SIZE_PER_GPU` accordingly:
```Bash
GPU_IDS=[0,1,2,3]
BATCH_SIZE_PER_GPU=3
```
### Gen1
- The effective batch size for the Gen1 training is 8.
- For training the model on the Gen1 dataset, we need 1x A100 80 GPU using 24 workers for training and 8 workers for evaluation:
```Bash
GPU_IDS=0
BATCH_SIZE_PER_GPU=8
TRAIN_WORKERS_PER_GPU=24
EVAL_WORKERS_PER_GPU=8
python RVT/train.py model=rnndet dataset=gen1 dataset.path=${DATA_DIR} wandb.project_name=ssms_event_cameras \
wandb.group_name=gen1 +experiment/gen1="${MDL_CFG}.yaml" hardware.gpus=${GPU_IDS} \
batch_size.train=${BATCH_SIZE_PER_GPU} batch_size.eval=${BATCH_SIZE_PER_GPU} \
hardware.num_workers.train=${TRAIN_WORKERS_PER_GPU} hardware.num_workers.eval=${EVAL_WORKERS_PER_GPU}
```

## Code Acknowledgments
This project has used code from the following projects:
- [RVT](https://github.com/uzh-rpg/RVT) - Recurrent Vision Transformers for Object Detection with Event Cameras in PyTorch
- [S4](https://github.com/state-spaces/s4) - Structured State Spaces for Sequence Modeling, in particular S4 and S4D models in PyTorch
- [S5](https://github.com/lindermanlab/S5) - Simplified State Space Layers for Sequence Modeling in JAX
- [S5 PyTorch](https://github.com/i404788/s5-pytorch) - S5 model in PyTorch
