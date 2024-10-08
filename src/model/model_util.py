from mmaction import __version__
from mmaction.apis import init_random_seed, train_model
from mmaction.datasets import build_dataset
from mmaction.models import build_model
from mmaction.utils import (collect_env, get_root_logger,
                            register_module_hooks, setup_multi_processes)
from mmcv import Config

from .encoder import (ImageEncoder, Resnet18,
                      SpatialTemporalEncoder)
from .resnetfc import ResnetFC
from .resnetfc_static import ResnetFC_static

def turn_off_pretrained(cfg):
    # recursively find all pretrained in the model config,
    # and set them None to avoid redundant pretrain steps for testing
    if 'pretrained' in cfg:
        cfg.pretrained = None

    # recursively turn off pretrained value
    for sub_cfg in cfg.values():
        if isinstance(sub_cfg, dict):
            turn_off_pretrained(sub_cfg)

def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get_string('type', 'mlp')  # mlp | resnet
    if mlp_type == 'resnet':
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == 'resnet_static':
        net = ResnetFC_static.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == 'empty' and allow_empty:
        net = None
    else:
        raise NotImplementedError('Unsupported MLP type')
    return net


def make_encoder(conf, **kwargs):
    enc_type = conf.get_string('type', 'slowonly')  # spatial | global
    print(f'Use {enc_type} for backbone.')
    if enc_type == 'spatial':
        net = SpatialEncoder.from_conf(conf, **kwargs)
    elif enc_type == 'global':
        net = ImageEncoder.from_conf(conf, **kwargs)
    elif enc_type == 'slowonly':
        # # model = dict(
        # #     type='Recognizer3D',
        # #     backbone=dict(
        # #         type='ResNet3dSlowOnly',
        # #         depth=50,
        # #         pretrained='torchvision://resnet50',
        # #         lateral=False,
        # #         conv1_kernel=(1, 7, 7),
        # #         conv1_stride_t=1,
        # #         pool1_stride_t=1,
        # #         inflate=(0, 0, 1, 1),
        # #         norm_eval=False),
        # #     cls_head=dict(
        # #         type='I3DHead',
        # #         in_channels=2048,
        # #         num_classes=400,
        # #         spatial_type='avg',
        # #         dropout_ratio=0.5),
        # #         # model training and testing settings
        # #         train_cfg=None,
        # #         test_cfg=dict(average_clips='prob'))
        # config_path = "mmaction_configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py"
        # config_options = {}
        # cfg = Config.fromfile(config_path)
        # cfg.merge_from_dict(config_options)
        # turn_off_pretrained(cfg.model)
        # net = build_model(
        # cfg.model,
        # train_cfg=cfg.get('train_cfg'),
        # test_cfg=cfg.get('test_cfg'))
        net = SpatialTemporalEncoder(
            num_layers=conf['num_layers'],
            config_path=conf['config_path'],
            pretrained=conf['pretrained'],
            pretrained_path=conf['pretrained_path'],

        )
    elif enc_type == 'resnet18':
        net = Resnet18()

    else:
        raise NotImplementedError('Unsupported encoder type')
    return net
