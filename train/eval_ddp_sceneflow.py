# Training to a set of multiple objects (e.g. ShapeNet or DTU)
# tensorboard logs available in logs/<expname>

import os
import os.path as osp
import sys

import logging
import random
import time

import colorlog
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
import trainlib
from dotmap import DotMap
from render_utils import *
from run_nerf_helpers import *
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from src import util
from src.model import loss

import torch.distributed as dist


print_freq = 100

def set_random_seed(seed=1029, deterministic=True):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    log_colors_config = {
        'DEBUG': 'white',  # cyan white
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }

    file_formatter = logging.Formatter(
        '[%(asctime)s][%(filename)s][line:%(lineno)d] %(message)s', '%m-%d %H:%M:%S'
    )
    # console_formatter = colorlog.ColoredFormatter(
    #     fmt='%(log_color)s[%(asctime)s] %(filename)s line:%(lineno)d [%(levelname)s] : %(message)s',
    #     datefmt="%m-%d %H:%M:%S",
    #     log_colors=log_colors_config
    # )
    console_formatter = logging.Formatter(
        fmt='\033[1;36m[%(asctime)s]\033[0m \033[1;33m%(filename)s\033[0m \033[1;35mline:%(lineno)d\033[0m \033[0;32m[%(levelname)s]\033[0m : %(message)s',
        datefmt='%m-%d %H:%M:%S',
        # log_colors=log_colors_config
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(console_formatter)
    logger.addHandler(sh)

    return logger


def extra_args(parser):
    parser.add_argument(
        '--batch_size', '-B', type=int, default=10, help="Object batch size ('SB')"
    )
    parser.add_argument(
        '--nviews',
        '-V',
        type=str,
        default='1',
        help="Number of source views (multiview); put multiple (space delim) to pick randomly per batch ('NV')",
    )
    parser.add_argument(
        '--freeze_enc',
        action='store_true',
        default=True,
        help='Freeze encoder weights and only train MLP',
    )

    parser.add_argument(
        '--no_bbox_step',
        type=int,
        default=100000,
        help='Step to stop using bbox sampling',
    )
    parser.add_argument(
        '--fixed_test',
        action='store_true',
        default=None,
        help='Freeze encoder weights and only train MLP',
    )

  
args, conf = util.args.parse_args(extra_args)
device = util.get_cuda(args.gpu_id[0])
args.eval_only = True

#!--------------------------------------

# slowonly_feature_fusion = FeatureFusion()

#!
# setup_multi_processes()

# args.lrate = 5e-5
args.add_features = conf['info.add_features']
args.basedir = conf['info.basedir']
args.expname = conf['info.expname']
# args.expname = 'train_2enc_sparse_loss_lambda0.01_dyn_blending_depth0.1_wo_depthdecay_Fnorm_EPSdepth_Balloon1_H270_DyNeRF_pretrain_PixelnerfSemantic_256channel_useviewdir_SDresnet_woNDC'


# args.blending_thickness = 0.1
# args.sparse_loss_lambda = 0.01
#! Specify file lists. None denotes using all datas
# args.dataset_file_lists = ["data/Balloon2", "data/Balloon1", "data/Truck"]
# args.dataset_file_lists = ["data/Balloon2", "data/Balloon1"]
args.dataset_file_lists = conf['info.dataset_file_lists']
# args.dataset_file_lists = ['data/Playground']
# N_scenes = len(args.dataset_file_lists)
#! Cancel the pretrained resnet model
args.ft_path_S = conf['info.ft_path_S']
args.ft_S = conf['info.ft_S']
# args.ft_path_S = None
# args.use_viewdirsDyn = False
args.no_ndc = conf['info.no_ndc']
#! Random seed
args.random_seed = conf['info.random_seed']
#! Enable encoder grads
args.freeze_enc = conf['info.freeze_enc']
args.i_testset = args.i_testset // 2
args.i_video = args.i_video // 2
args.N_rand = conf['info.N_rand']
args.blending_thickness = conf['info.blending_thickness']
args.chunk = conf['info.chunk']
#!--------------------------------------

nviews = list(map(int, args.nviews.split()))

class PixelNeRFTrainer(trainlib.Trainer):
    def __init__(self, args, conf):
        super().__init__(args, conf, device=None)
        # self.renderer_state_path = "%s/%s/_renderer" % (
        #     self.args.checkpoints_path,
        #     self.args.name,
        # )

        # self.lambda_coarse = conf.get_float('loss.lambda_coarse')
        # self.lambda_fine = conf.get_float('loss.lambda_fine', 1.0)
        # self.logger.info(
        #     'lambda coarse {} and fine {}'.format(self.lambda_coarse, self.lambda_fine)
        # )
        self.rgb_coarse_crit = loss.get_rgb_loss(conf['loss.rgb'], True)
        fine_loss_conf = conf['loss.rgb']
        if 'rgb_fine' in conf['loss']:
            self.logger.info('using fine loss')
            fine_loss_conf = conf['loss.rgb_fine']
        self.rgb_fine_crit = loss.get_rgb_loss(fine_loss_conf, False)

        # if self.args.resume:
        #     if os.path.exists(self.renderer_state_path):
        #         renderer.load_state_dict(
        #             torch.load(self.renderer_state_path, map_location=device)
        #         )

        # self.z_near = dset.z_near
        # self.z_far = dset.z_far

        self.use_bbox = self.args.no_bbox_step > 0


    # def post_batch(self, epoch, batch):
    #     renderer.sched_step(self.args.batch_size)

    # def extra_save_state(self):
    #     torch.save(renderer.state_dict(), self.renderer_state_path)

    def train_step(self, data, global_step):
        # import ipdb; ipdb.set_trace()

        encoder_imgs = data['encoder_imgs'].permute(0, 4, 1, 2, 3)
        # encoder_features = self.net.encoder.backbone(encoder_imgs)
        # encoder_features = slowonly_feature_fusion(encoder_features).squeeze().detach()
        # self.render_kwargs_train['network_fn_d'].module.encoder_features = encoder_features
        dataname = data['dataname']
        images = data['images']
        invdepths = data['invdepths']
        masks = data['masks']
        poses = data['poses']
        bds = data['bds']
        render_poses = data['render_poses']
        render_focals = data['render_focals']
        grids = data['grids']

        hwf = poses[:, 0, :3, -1]
        poses = poses[:, :, :3, :4]
        num_img = float(poses.shape[1])
        assert len(poses) == len(images)
        self.logger.info('Loaded llff'+str(images.shape)+
            str(render_poses.shape)+str(hwf)+str(self.args.datadir))

        # Use all views to train
        i_train = np.array([i for i in np.arange(int(images.shape[1]))])

        self.logger.info('DEFINING BOUNDS')
        #! Change the pos
        if self.args.no_ndc:
            # raise NotImplementedError
            near = bds.min() * .9
            far = bds.max() * 1.
        else:
            near = 0.
            far = 1.
            self.logger.info(f'NEAR FAR {near} {far}')

        H, W, focal = torch.split(hwf, [1,1,1], -1)
        H, W = H.int(), W.int()
        hwf = torch.cat([H, W, focal], dim=-1)


        global_step = self.start

        bds_dict = {
            'near': near,
            'far': far,
            'num_img': num_img,
        }
        self.render_kwargs_train.update(bds_dict)
        self.render_kwargs_test.update(bds_dict)

        #! Use semantic feature
        self.render_kwargs_train['use_feature'] = True
        self.render_kwargs_test['use_feature'] = True

        N_rand = self.args.N_rand #1024

        # Move training data to GPU
        images = torch.Tensor(images).cuda()
        invdepths = torch.Tensor(invdepths).cuda()
        masks = 1.0 - torch.Tensor(masks).cuda()
        poses = torch.Tensor(poses).cuda()
        grids = torch.Tensor(grids).cuda()

        if self.rank == 0:
            self.logger.info('Begin')
            self.logger.info('EVAL views are'+ str(i_train))


        decay_iteration = max(25, num_img)

        # Pre-train StaticNeRF
        self.render_kwargs_train.update({'pretrain': False})
        self.render_kwargs_test.update({'pretrain': False})
        global_step = self.start
        self.net.pretrain = False

        grad_vars = list(self.net.encoder.parameters()) + list(self.net.mlp_dynamic.parameters())
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=self.args.lrate, betas=(0.9, 0.999))
        set_requires_grad(self.net.encoder2d, False)
        set_requires_grad(self.net.mlp_static, False)
        # set_requires_grad(self.net.encoder2d, False)

            # Fix the StaticNeRF and only train the DynamicNeRF
            # grad_vars_d = []
            # for name, para in self.render_kwargs_train['network_fn_d'].named_parameters():
            #     # if "mlp_static" in name:
            #     #     continue
            #     grad_vars_d.append(para)
            # self.optimizer = torch.optim.Adam(params=grad_vars_d, lr=self.args.lrate, betas=(0.9, 0.999))
            # grad_vars_d = list(self.render_kwargs_train['network_fn_d'].parameters())
            # self.optimizer = torch.optim.Adam(params=grad_vars_d, lr=self.args.lrate, betas=(0.9, 0.999))
        if self.conf['model']['mlp_dynamic']['origin_pipeline'] == True or self.args.freeze_enc:
            with torch.no_grad():
                feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")

        with logging_redirect_tqdm():
            batch_masks = []
            batch_rays = []
            batch_poses = []
            batch_invdepths = []
            batch_grids = []
            target_rgbs = []
            rays_o = []
            rays_d = []
            t = []
            img_ies = []

            # self.net.encode(encoder_imgs, poses, focal, images2d=images)
            if self.conf['model']['mlp_dynamic']['origin_pipeline'] == False and not self.args.freeze_enc:
                with torch.no_grad():
                    feature_dict = self.net.train_step(encoder_imgs, poses, focal, images2d=images, mode="encode")

            for b in range(images.shape[0]):
                # self.render_kwargs_train["network_fn_d"].module.select_batch = True
                # self.render_kwargs_test["network_fn_d"].module.batch = b
                # No raybatching as we need to take random rays from one image at a time
                img_i = np.random.choice(i_train)
                curr_t = img_i / num_img * 2. - 1.0 # time of the current frame
                target = images[b, img_i]
                pose = poses[b, img_i, :3, :4]
                mask = masks[b, img_i] # Static region mask
                invdepth = invdepths[b, img_i]
                grid = grids[b, img_i]


                curr_rays_o, curr_rays_d = get_rays(H[b].item(), W[b].item(), focal[b].item(), pose) # (H, W, 3), (H, W, 3)
                coords_d = torch.stack((torch.where(mask < 0.5)), -1)
                coords_s = torch.stack((torch.where(mask >= 0.5)), -1)
                coords = torch.stack((torch.where(mask > -1)), -1)
                # test coord transfer.
                my_coords = torch.stack((torch.where(mask[:32, :32])), -1)

                # Evenly sample dynamic region and static region
                select_inds_d = np.random.choice(coords_d.shape[0], size=[min(len(coords_d), N_rand//2)], replace=False)
                select_inds_s = np.random.choice(coords_s.shape[0], size=[N_rand//2], replace=False)
                select_coords = torch.cat([coords_s[select_inds_s],
                                        coords_d[select_inds_d]], 0)

                def select_batch(value, select_coords=select_coords):
                    return value[select_coords[:, 0], select_coords[:, 1]]
                # def select_batch(value, select_coords=my_coords):
                    # return value[select_coords[:, 0], select_coords[:, 1]]
                curr_rays_o = select_batch(curr_rays_o) # (N_rand, 3)
                curr_rays_d = select_batch(curr_rays_d) # (N_rand, 3)
                curr_target_rgb = select_batch(target)
                curr_batch_grid = select_batch(grid) # (N_rand, 8)
                curr_batch_mask = select_batch(mask[..., None])
                curr_batch_invdepth = select_batch(invdepth)
                curr_batch_rays = torch.stack([curr_rays_o, curr_rays_d], 0)

                rays_o.append(curr_rays_o)
                rays_d.append(curr_rays_d)
                target_rgbs.append(curr_target_rgb)
                batch_masks.append(curr_batch_mask)
                batch_rays.append(curr_batch_rays)
                batch_poses.append(pose)
                batch_grids.append(curr_batch_grid)
                batch_invdepths.append(curr_batch_invdepth)
                t.append(curr_t)
                img_ies.append(img_i)

            rays_o = torch.stack(rays_o, dim=0)
            rays_d = torch.stack(rays_d, dim=0)
            target_rgbs = torch.stack(target_rgbs, dim=0)
            batch_masks = torch.stack(batch_masks, dim=0)
            batch_rays = torch.stack(batch_rays, dim=0)
            batch_poses = torch.stack(batch_poses, dim=0)
            batch_grids = torch.stack(batch_grids, dim=0)
            batch_invdepths = torch.stack(batch_invdepths)
            t = torch.Tensor(t).cuda()


            # i = 12345

            # if i % self.args.i_video == 0 and self.rank == 0 and i > 0:
            #     print("rank ", self.rank, " is in the i_video")

            #     time2renders = []
            #     pose2renders = []
            #     for b in range(images.shape[0]):

            #         # Change time and change view at the same time.
            #         time2render = np.concatenate((np.repeat((i_train / float(num_img) * 2. - 1.0), 4),
            #                                     np.repeat((i_train / float(num_img) * 2. - 1.0)[::-1][1:-1], 4)))
            #         if len(time2render) > len(render_poses[b]):
            #             pose2render = np.tile(render_poses[b], (int(np.ceil(len(time2render) / len(render_poses[b]))), 1, 1))
            #             pose2render = pose2render[:len(time2render)]
            #             pose2render = torch.Tensor(pose2render).cuda()
            #         else:
            #             time2render = np.tile(time2render, int(np.ceil(len(render_poses[b]) / len(time2render))))
            #             time2render = time2render[:len(render_poses[b])]
            #             pose2render = torch.Tensor(render_poses[b]).cuda()

            #         time2renders.append(time2render)
            #         pose2renders.append(pose2render)

            #     time2renders = np.stack(time2renders, axis=0)
            #     pose2renders = torch.stack(pose2renders, dim=0)

            #     result_type = 'novelviewtime'

            #     testsavedir = os.path.join(
            #         self.basedir, self.expname, result_type + '_{:06d}'.format(i))
            #     os.makedirs(testsavedir, exist_ok=True)
            #     with torch.no_grad():
            #         ret = render_path(pose2renders, time2renders,
            #                         hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
            #                         feature_dict=feature_dict, near=near)
            #     moviebase = os.path.join(
            #         testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
            #     save_res(moviebase, ret, images.shape[0])

            evaldir = conf['info.evaldir']
            if True:
                print("rank ", self.rank, " is in the i_testset_all")
                # # Change view and time.
                # pose2render = poses[...]
                # time2render = i_train / float(num_img) * 2. - 1.0
                # time2render = np.tile(time2render, [poses.shape[0], 1])
                # result_type = 'testset'

                # testsavedir = os.path.join(
                #     self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                # os.makedirs(testsavedir, exist_ok=True)
                # with torch.no_grad():
                #     ret = render_path(pose2render, time2render,
                #                     hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                #                     flows_gt_f=grids[:, :, :, :, 2:4], flows_gt_b=grids[:, :, :, :, 5:7],
                #                     feature_dict=feature_dict, near=near)
                # moviebase = os.path.join(
                #     testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                # save_res(moviebase, ret, images.shape[0])

                # # Fix view (first view) and change time.
                # pose2render = poses[:, 0:1, ...].expand([poses.shape[0], int(num_img), 3, 4])
                # time2render = i_train / float(num_img) * 2. - 1.0
                # time2render = np.tile(time2render, [poses.shape[0], 1])
                # result_type = 'testset_view000'

                # testsavedir = os.path.join(
                #     self.basedir, self.expname, result_type + '_{:06d}'.format(i))
                # os.makedirs(testsavedir, exist_ok=True)
                # with torch.no_grad():
                #     ret = render_path(pose2render, time2render,
                #                     hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                #                     feature_dict=feature_dict, near=near)
                # moviebase = os.path.join(
                #     testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, i))
                # save_res(moviebase, ret, images.shape[0])

                # Fix time (the first timestamp) and change view.
                for each in range(len(i_train)):
                    pose2render = poses[...]
                    time2render = np.tile(i_train[each], [int(num_img)]) / float(num_img) * 2. - 1.0
                    time2render = np.tile(time2render, [poses.shape[0], 1])
                    result_type = '{:08d}'.format(each + 1)

                    testsavedir = os.path.join(
                        self.basedir, self.expname, evaldir, 'step_{:06d}'.format(global_step), result_type)
                    os.makedirs(testsavedir, exist_ok=True)
                    with torch.no_grad():
                        ret = render_path(pose2render, time2render,
                                        hwf, self.args.chunk, self.render_kwargs_test, savedir=testsavedir,
                                        feature_dict=feature_dict, near=near, eval_only=args.eval_only)
                    moviebase = os.path.join(
                        testsavedir, '{}_{}_{:06d}_'.format(self.expname, result_type, global_step))
                    save_res(moviebase, ret, images.shape[0])


        return



# torch.set_default_tensor_type('torch.cuda.FloatTensor')
trainer = PixelNeRFTrainer(args, conf)
trainer.start_train()
