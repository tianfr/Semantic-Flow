import os
import os.path as osp
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tqdm
import warnings
import json

import time
from run_nerf_helpers import *
from src import util
from src.data import get_split_dataset
from src.model import make_model

import torch.distributed as dist
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash
from mmaction.apis import init_random_seed
#  set_random_seed, train_detector
from mmaction.utils import (collect_env,  get_root_logger, setup_multi_processes, build_ddp)
                        #  replace_cfg_vals, 
                        #  update_data_root)
class Trainer:
    def __init__(self, args, conf, device=None):
        self.args = args
        N_scenes = len(args.dataset_file_lists)

        assert args.no_ndc != conf['model']['use_ndc']

        if conf['model']['mlp_static']['origin_pipeline']  or conf['model']['mlp_dynamic']['origin_pipeline']:
            assert True
            # assert N_scenes == 1

        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            dist_params = dict(
                backend = args.backend,
                # port = args.master_port,
                port = 22446
            )
            if args.launcher == 'pytorch':
                del dist_params["port"]
            # re-set gpu_ids with distributed training mode
            init_dist(args.launcher, **dist_params)
            _, world_size = get_dist_info()
            print("world size: ", world_size)
            args.gpu_ids = range(world_size)
            args.N_iters = args.N_iters // max(1, min(world_size // N_scenes, 2))
            args.i_testset = args.i_testset // max(1, min(world_size // N_scenes, 2))
            args.i_video = args.i_video // max(1, min(world_size // N_scenes, 2))
            args.N_rand = args.N_rand // N_scenes
            args.expname = str(world_size) + "gpus" + args.expname

            if "lrate" in conf["info"]:
                args.lrate = conf.get_bool("info.lrate")
            if "lrate_decay" in conf["info"]:
                args.lrate_decay = conf.get_int("info.lrate_decay")

            if conf.get_bool("info.fast_render", False):
                # assert args.fast_render_iter != 0
                args.expname = str(args.fast_render_iter) +"steps_" + args.expname
                if args.fast_render_basename is not None:
                    args.expname = str(args.fast_render_basename) +"_" + args.expname
                args.N_iters = max(args.fast_render_iter, 1)
                args.N_static_iters = max(args.fast_render_iter, 1)
                args.i_video = max(args.fast_render_iter - 1, 1)
                args.i_testset = max(args.fast_render_iter - 1, 1)
                args.i_testset_all = max(args.fast_render_iter - 1, 1)
            else:
                assert args.fast_render_iter == 0

        # Create log dir and copy the config file
        basedir = args.basedir
        expname = args.expname
        os.makedirs(osp.join(basedir, expname), exist_ok=True)

        timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime())
        log_file = osp.join(args.basedir,args.expname,timestamp + '.log')
        # logger = get_logger(logger_file)
        self.logger = get_root_logger(log_file=log_file)
        # NOTE: There is a bug on get_root_loger of mmaction2, but personalized util.get_logger cannot write logs into a file. 
        # self.logger = util.get_logger()
        # os.environ["MASTER_ADDR"] = args.master_addr
        # os.environ["MASTER_PORT"] = args.master_port
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        self.logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info
        # meta['config'] = args.pretty_text
        # log some basic info
        self.logger.info(f'Distributed training: {distributed}')
        # self.logger.info(f'Config:\n{args.pretty_text}')

        args.device = util.get_device()
        # set random seeds
        seed = init_random_seed(args.random_seed, device=args.device)
        seed = seed + dist.get_rank()
        self.logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')

        util.set_random_seed(seed, deterministic=args.deterministic)
        args.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(args.config)

        # 
        if conf.get_bool('model.use_semantic_labels', False):
            conf['model.mlp_static']['use_semantic_labels'] = conf['model.use_semantic_labels']
            conf['model.mlp_static']['semantic_class_num'] = conf['model.semantic_class_num']
            conf['model.mlp_dynamic']['use_semantic_labels'] = conf['model.use_semantic_labels']
            conf['model.mlp_dynamic']['semantic_class_num'] = conf['model.semantic_class_num']
            
        # Build SemanticFlow network with SlowOnly encoder.
        self.net = make_model(conf['model'],  stop_encoder_grad=args.freeze_enc, use_static_resnet=True)
        train_dataset, val_dset, _ = get_split_dataset(
            args.dataset_format, 
            args.datadir, 
            file_lists=args.dataset_file_lists, 
            use_occlusion=conf.get_bool("info.use_occlusion", False),
            use_semantic_labels=conf.get_bool("model.use_semantic_labels", False),
            add_flow_noise=conf.get_float("info.add_flow_noise", 0),
            use_flow_net_flow = conf.get_bool("info.use_flow_net_flow", False),
        )
        print(
            'dset z_near {}, z_far {}, lindisp {}'.format(train_dataset.z_near, train_dataset.z_far, train_dataset.lindisp)
        )

        # For DDP
        self.rank = dist.get_rank()
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_resnet_nerf(args, self.net, self.rank)
        
        render_kwargs_train['use_semantic_labels'] = conf.get_bool('model.use_semantic_labels', False)
        render_kwargs_train['semantic_class_num'] = conf.get_int('model.semantic_class_num', 0)
        render_kwargs_test['use_semantic_labels'] = conf.get_bool('model.use_semantic_labels', False)
        render_kwargs_test['semantic_class_num'] = conf.get_int('model.semantic_class_num',0)
        
        render_kwargs_train['net'] = self.net
        render_kwargs_test['net'] = self.net


        self.train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        # self.test_data_loader = torch.utils.data.DataLoader(
        #     test_dataset,
        #     batch_size=min(args.batch_size, 16),
        #     shuffle=True,
        #     num_workers=4,
        #     pin_memory=True,
        # )
        self.render_kwargs_train = render_kwargs_train
        self.render_kwargs_test = render_kwargs_test
        self.start = start
        self.grad_vars = grad_vars
        self.optimizer = optimizer

        self.expname = expname
        self.basedir = basedir
        self.timestamp = timestamp
        # self.save_interval = conf.get_int("save_interval")
        # self.print_interval = conf.get_int("print_interval")
        # self.vis_interval = conf.get_int("vis_interval")
        # self.eval_interval = conf.get_int("eval_interval")
        # self.num_epoch_repeats = conf.get_int("num_epoch_repeats", 1)
        # self.num_epochs = args.epochs
        # self.accu_grad = conf.get_int("accu_grad", 1)
        self.conf = conf


        # Summary writers
        self.writer = SummaryWriter(os.path.join(self.basedir, 'summaries', self.expname, self.timestamp))



        self.fixed_test = hasattr(args, "fixed_test") and args.fixed_test

        if not args.eval_only and self.rank == 0:
            f = os.path.join(self.basedir, self.expname, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(args)):
                    attr = getattr(args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
            if args.config is not None:
                f = os.path.join(self.basedir, self.expname, 'dynamic_config.txt')
                with open(f, 'w') as file:
                    file.write(open(args.config, 'r').read())
            f = os.path.join(self.basedir, self.expname, 'semflow_config.json')
            with open(f, 'w') as file:
                conf_dict = json.dumps(conf, sort_keys=False, indent=4, separators=(',', ': '))
                file.write(conf_dict)

        # # Currently only Adam supported
        # self.optim = torch.optim.Adam(net.parameters(), lr=args.lr)
        # if args.gamma != 1.0:
        #     self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        #         optimizer=self.optim, gamma=args.gamma
        #     )
        # else:
        #     self.lr_scheduler = None

        # # Load weights
        # self.managed_weight_saving = hasattr(net, "load_weights")
        # if self.managed_weight_saving:
        #     net.load_weights(self.args)
        # self.iter_state_path = "%s/%s/_iter" % (
        #     self.args.checkpoints_path,
        #     self.args.name,
        # )
        # self.optim_state_path = "%s/%s/_optim" % (
        #     self.args.checkpoints_path,
        #     self.args.name,
        # )
        # self.lrsched_state_path = "%s/%s/_lrsched" % (
        #     self.args.checkpoints_path,
        #     self.args.name,
        # )
        # self.default_net_state_path = "%s/%s/net" % (
        #     self.args.checkpoints_path,
        #     self.args.name,
        # )
        self.start_iter_id = 0
        # if args.resume:
        #     if osp.exists(self.optim_state_path):
        #         try:
        #             self.optim.load_state_dict(
        #                 torch.load(self.optim_state_path, map_location=device)
        #             )
        #         except:
        #             warnings.warn(
        #                 "Failed to load optimizer state at", self.optim_state_path
        #             )
        #     if self.lr_scheduler is not None and osp.exists(
        #         self.lrsched_state_path
        #     ):
        #         self.lr_scheduler.load_state_dict(
        #             torch.load(self.lrsched_state_path, map_location=device)
        #         )
        #     if osp.exists(self.iter_state_path):
        #         self.start_iter_id = torch.load(
        #             self.iter_state_path, map_location=device
        #         )["iter"]
        #     if not self.managed_weight_saving and osp.exists(
        #         self.default_net_state_path
        #     ):
        #         net.load_state_dict(
        #             torch.load(self.default_net_state_path, map_location=device)
        #         )

        # self.visual_path = osp.join(self.args.visual_path, self.args.name)

    def post_batch(self, epoch, batch):
        """
        Ran after each batch
        """
        pass

    def extra_save_state(self):
        """
        Ran at each save step for saving extra state
        """
        pass

    def train_step(self, data, global_step):
        """
        Training step
        """
        raise NotImplementedError()

    def eval_step(self, data, global_step):
        """
        Evaluation step
        """
        raise NotImplementedError()

    def vis_step(self, data, global_step):
        """
        Visualization step
        """
        return None, None

    def load_checkpoints(self):
        start = 0
        basedir = args.basedir
        expname = args.expname

        if args.ft_path is not None and args.ft_path != 'None':
            ckpts = [args.ft_path]
        else: #  here
            ckpts = [osp.join(basedir, expname, f) for f in sorted(os.listdir(osp.join(basedir, expname))) if 'tar' in f]
        print('Found ckpts', ckpts)
        ###Have some problems here so comment it.
        if len(ckpts) > 2 and not args.no_reload:
            ckpt_path = ckpts[-2]
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            # grad_vars_d = list(render_kwargs_train['network_fn_d'].parameters())
            grad_vars_d = []
            for name, para in model_d.named_parameters():
                if 'mlp_static' in name and not args.ft_S:
                    continue
                grad_vars_d.append(para)
            optimizer = torch.optim.Adam(params=grad_vars_d, lr=args.lrate, betas=(0.9, 0.999))
            start = ckpt['global_step'] + 1
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            model_d.load_state_dict(ckpt['network_fn_d_state_dict'])
            model_s.load_state_dict(ckpt['network_fn_s_state_dict'])
            print('Resetting step to', start)
            args.pretrain = False

            if model_fine is not None:
                raise NotImplementedError



    def start_train(self):
        step_id = self.start_iter_id
        for data in self.train_data_loader:
            self.train_step(data, global_step=step_id)
            return 
        
        
        
        
        
        def fmt_loss_str(losses):
            return "loss " + (" ".join(k + ":" + str(losses[k]) for k in losses))

        def data_loop(dl):
            """
            Loop an iterable infinitely
            """
            while True:
                for x in iter(dl):
                    yield x

        test_data_iter = data_loop(self.test_data_loader)

        step_id = self.start_iter_id

        progress = tqdm.tqdm(bar_format="[{rate_fmt}] ")
        for epoch in range(self.num_epochs):
            self.writer.add_scalar(
                "lr", self.optim.param_groups[0]["lr"], global_step=step_id
            )

            batch = 0
            for _ in range(self.num_epoch_repeats):
                start = time.time()
                for data in self.train_data_loader:
                    end = time.time()-start
                    losses = self.train_step(data, global_step=step_id)
                    loss_str = fmt_loss_str(losses)
                    if batch % self.print_interval == 0:
                        print(
                            "E",
                            epoch,
                            "B",
                            batch,
                            loss_str,
                            " lr",
                            self.optim.param_groups[0]["lr"],
                            " data time",
                            end
                        )

                    if batch % self.eval_interval == 0:
                        test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            test_losses = self.eval_step(test_data, global_step=step_id)
                        self.net.train()
                        test_loss_str = fmt_loss_str(test_losses)
                        self.writer.add_scalars("train", losses, global_step=step_id)
                        self.writer.add_scalars(
                            "test", test_losses, global_step=step_id
                        )
                        print("*** Eval:", "E", epoch, "B", batch, test_loss_str, " lr")

                    if batch % self.save_interval == 0 and (epoch > 0 or batch > 0):
                        print("saving")
                        if self.managed_weight_saving:
                            self.net.save_weights(self.args)
                        else:
                            torch.save(
                                self.net.state_dict(), self.default_net_state_path
                            )
                        torch.save(self.optim.state_dict(), self.optim_state_path)
                        if self.lr_scheduler is not None:
                            torch.save(
                                self.lr_scheduler.state_dict(), self.lrsched_state_path
                            )
                        torch.save({"iter": step_id + 1}, self.iter_state_path)
                        self.extra_save_state()

                    if batch % self.vis_interval == 0:
                        print("generating visualization")
                        if self.fixed_test:
                            test_data = next(iter(self.test_data_loader))
                        else:
                            test_data = next(test_data_iter)
                        self.net.eval()
                        with torch.no_grad():
                            vis, vis_vals = self.vis_step(
                                test_data, global_step=step_id
                            )
                        if vis_vals is not None:
                            self.writer.add_scalars(
                                "vis", vis_vals, global_step=step_id
                            )
                        self.net.train()
                        if vis is not None:
                            import imageio

                            vis_u8 = (vis * 255).astype(np.uint8)
                            imageio.imwrite(
                                osp.join(
                                    self.visual_path,
                                    "{:04}_{:04}_vis.png".format(epoch, batch),
                                ),
                                vis_u8,
                            )

                    if (
                        batch == self.num_total_batches - 1
                        or batch % self.accu_grad == self.accu_grad - 1
                    ):
                        self.optim.step()
                        self.optim.zero_grad()

                    self.post_batch(epoch, batch)
                    step_id += 1
                    batch += 1
                    progress.update(1)
                    start = time.time()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
