# Semantic-Flow
[ICLR 2023] This is the official implementation of our paper "Semantic Flow: Learning Semantic Fields of Dynamic Scenes from Monocular Videos".

![image](https://github.com/tianfr/Semantic-Flow/assets/44290909/97fb59da-2987-4546-8c49-39480f3c0431)

*Codes will be released. Stay tuned!*

## Environment Setup
The code is tested with
* Ubuntu 16.04
* Anaconda 3
* Python 3.8.12
* CUDA 11.1
* A100 or 3090 GPUs


To get started, please create the conda environment `semflow` by running
```
conda create --name mononerf python=3.8
conda activate semflow

pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install imageio==2.19.2 pyhocon==0.3.60  pyparsing==2.4.7 configargparse==1.5.3 tensorboard==2.13.0 ipdb==0.13.13 imgviz==1.7.2 imageio--ffmpeg==0.4.8 
pip install mmcv-full==1.7.1
```
Then install [MMAction2](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) v0.24.1 manually.

```
git clone https://github.com/open-mmlab/mmaction2.git
cd mmaction2
git checkout v0.24.1
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without re-installation.
```

Install other dependencies.
```
pip install tqdm Pillow==9.1.1
```
Finally, clone the Semantic Flow project:
```
git clone https://github.com/tianfr/Semantic-Flow.git
cd Semantic-Flow
```


## Dataset

Our Semantic Nvidia Dataset could be accessed from this [link](https://drive.google.com/file/d/1lXQZaDASjY44CJo-gc6f5cwvsdzDmpMy/view?usp=sharing).

After downloading the data run the following command to unzip the data

```
unzip data_semantic.zip
```
## Backbone Checkpoints
Download the [SlowOnly](https://arxiv.org/abs/1812.03982) pretrained model from [MMAction2](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html) website.
```
mkdir checkpoints
wget -P checkpoints/ https://download.openmmlab.com/mmaction/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb_20200912-3f9ce182.pth
```

## Training
### Environment Initialization
```
export PYTHONPATH=.
```
All the training procedure is conducted on **GPU 0** by default.
### Multiple scenes
You can train a model from scratch by running:
```
chmod +x run_sh/full_label_w_config.sh
run_sh/full_label_w_config.sh 0 semflow_conf/exp/multiscenes/Balloon1_Balloon2_w_consistency_lrdecay8k.conf
```

### Generalization
Use the checkpoints by multiple scene training and test the model on the novel scenes:
```
chmod +x run_sh/scene_adaptation.sh
run_sh/scene_adaptation.sh 0 2000 path_to_multiscene_ckpt  semflow_conf/exp/generation/Umbrella.conf
```

### Tracking
Test the model on tracking task:
```
chmod +x run_sh/tracking_w_config.sh
run_sh/tracking_w_config.sh 0 semflow_conf/exp/tracking/Balloon2_75_track.conf
```

### Completion
Test the generalization ability on completion task:
```
chmod +x run_sh/completion_w_config.sh
run_sh/completion_w_config.sh 0 Semantic-Flow/semflow_conf/exp/completion/Balloon2.conf
```
