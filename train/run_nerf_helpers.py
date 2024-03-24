import os

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.utils import build_ddp
from imgviz import label_colormap, depth2rgb

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Misc utils
def img2mse(x, y, M=None):
    if M == None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def img2mae(x, y, M=None):
    if M == None:
        return torch.mean(torch.abs(x - y))
    else:
        return torch.sum(torch.abs(x - y) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L1(x, M=None):
    if M == None:
        return torch.mean(torch.abs(x))
    else:
        return torch.sum(torch.abs(x) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def L2(x, M=None):
    if M == None:
        return torch.mean(x ** 2)
    else:
        return torch.sum((x ** 2) * M) / (torch.sum(M) + 1e-8) / x.shape[-1]


def entropy(x):
    return -torch.sum(x * torch.log(x + 1e-19)) / (x.shape[0] * x.shape[1])


def mse2psnr(x): return -10. * torch.log(x) / torch.log(torch.Tensor([10.])).cuda()


def to8b(x): return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def CosineSimilarity(feature1, feature2):
    feature1 = feature1.reshape(-1, feature1.shape[-1])#将特征转换为N*(C*W*H)，即两维
    feature2 = feature2.reshape(-1, feature2.shape[-1])
    feature1 = F.normalize(feature1)  #F.normalize只能处理两维的数据，L2归一化
    feature2 = F.normalize(feature2)
    return 1.0 - (feature1 * feature2).sum(-1).mean()

def logits2label(x):
    return torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)


def crossentropy_loss(logit, label, ignore_index=-1):
    CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=ignore_index) # ignore index is followed SemanticNeRF which needs to revises 
    return CrossEntropyLoss(logit, label)

def logits2uncertainty(x):
    return torch.sum(-F.log_softmax(x, dim=-1)*F.softmax(x, dim=-1), dim=-1, keepdim=True)

class Embedder:

    def __init__(self, **kwargs):

        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):

        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2.**torch.linspace(0., max_freq, steps=N_freqs)
            #[  1.,   2.,   4.,   8.,  16.,  32.,  64., 128., 256., 512.]
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq : p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        #[sin(x), sin(y), sin(z), sin(t), cos(x), cos(y), cos(z), cos(t), sin(2x), ...]
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, i=0, input_dims=3):

    if i == -1:
        return nn.Identity(), 3

    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims, # 4
        'max_freq_log2': multires-1, # 9
        'num_freqs': multires, # 10
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim


# Dynamic NeRF model architecture
class NeRF_d(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirsDyn=True, add_features=False, use_fcnorm=False):
        """
        """
        super(NeRF_d, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirsDyn = use_viewdirsDyn
        self.use_fcnorm = use_fcnorm

        self.add_features = add_features
        if self.add_features:
            self.feature_embedding = nn.Linear(2048, 128)
            ##### adding 128 dims to input_ch for creating mlps
            input_ch += 128
        else:
            self.feature_embedding = None

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if self.use_viewdirsDyn:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.encoder_features = None

        self.sf_linear = nn.Linear(W, 6)
        self.weight_linear = nn.Linear(W, 1)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        if self.add_features:
            encoder_features = self.feature_embedding(self.encoder_features)
            input_pts = torch.cat([input_pts, encoder_features.expand([input_pts.size(0),-1])], -1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_fcnorm:
            h = F.normalize(h, dim=-1)

        # Scene flow should be unbounded. However, in NDC space the coordinate is
        # bounded in [-1, 1].
        sf = torch.tanh(self.sf_linear(h))
        blending = torch.sigmoid(self.weight_linear(h))

        if self.use_viewdirsDyn:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, sf, blending], dim=-1)


# Static NeRF model architecture
class NeRF_s(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=True):
        """
        """
        super(NeRF_s, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        self.pts_linears = nn.ModuleList(
            [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        if self.use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, 3)
        else:
            self.output_linear = nn.Linear(W, output_ch)

        self.weight_linear = nn.Linear(W, 1)

    def forward(self, x):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        blending = torch.sigmoid(self.weight_linear(h))
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return torch.cat([outputs, blending], -1)


def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn

    def ret(inputs, ori_input=None, use_network='dynamic', **kwargs):
        slice_chunk = chunk // inputs.shape[0]
        if ori_input is not None:
            static_output, dynamic_output = [], []
            for i in range(0, inputs.shape[1], slice_chunk):
                rs = fn.train_step(inputs[:, i: i+slice_chunk], ori_input[:, i: i+slice_chunk], use_network, mode="forward", **kwargs)
                if "static" in rs.keys():
                    static_output.append(rs["static"])
                else:
                    static_output.append(None)
                if "dynamic" in rs.keys():
                    dynamic_output.append(rs["dynamic"])
                else:
                    dynamic_output.append(None)
            if static_output[0] is not None:
                static_output = torch.cat(static_output, 1)
            else:
                static_output = None
            if dynamic_output[0] is not None:
                dynamic_output = torch.cat(dynamic_output, 1)
            else:
                dynamic_output = None
            return static_output, dynamic_output
            # return torch.cat([fn(inputs[:, i:i+slice_chunk], ori_input[:, i:i+slice_chunk]) for i in range(0, inputs.shape[1], slice_chunk)], 1)
        return torch.cat([fn(inputs[:, i:i+slice_chunk]) for i in range(0, inputs.shape[1], slice_chunk)], 1)
    return ret


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64, use_feature=False, **kwargs):
    """Prepares inputs and applies network 'fn'.
    """

    ori_input = inputs.reshape([inputs.shape[0], -1, inputs.shape[-1]])
    inputs_flat = torch.reshape(inputs, [inputs.shape[0], -1, inputs.shape[-1]])

    embedded = embed_fn(inputs_flat)
    kwargs['embed_fn'] = embed_fn
    if viewdirs is not None:
        input_dirs = viewdirs[:, :, None].expand(inputs[:, :, :, :3].shape)
        input_dirs_flat = torch.reshape(input_dirs, [input_dirs.shape[0], -1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)
    if use_feature:
        static_outputs_flat, dynamic_outputs_flat = batchify(fn, netchunk)(embedded, ori_input, **kwargs)
    else:
        outputs_flat = batchify(fn, netchunk)(embedded)
        outputs = torch.reshape(outputs_flat, list(
            inputs.shape[:-1]) + [outputs_flat.shape[-1]])
        return outputs
    if static_outputs_flat is not None:
        static_outputs = torch.reshape(static_outputs_flat, list(
            inputs.shape[:-1]) + [static_outputs_flat.shape[-1]])
    else:
        static_outputs = None
    if dynamic_outputs_flat is not None:
        dynamic_outputs = torch.reshape(dynamic_outputs_flat, list(
            inputs.shape[:-1]) + [dynamic_outputs_flat.shape[-1]])
    else:
        dynamic_outputs = None
    return static_outputs, dynamic_outputs


# def create_nerf(args,):
#     """Instantiate NeRF's MLP model.
#     """

#     embed_fn_d, input_ch_d = get_embedder(args.multires, args.i_embed, 4) # embed (x, y) to higher dim.
#     # 10 * 2 * 4 + 4 = 84
#     # L * (sin, cos) * (x, y, z, t) + (x, y, z, t)

#     input_ch_views = 0
#     embeddirs_fn = None
#     if args.use_viewdirs:
#         embeddirs_fn, input_ch_views = get_embedder(
#             args.multires_views, args.i_embed, 3)
#         # 4 * 2 * 3 + 3 = 27
#         # L * (sin, cos) * (3 Cartesian viewing direction unit vector from [theta, phi]) + (3 Cartesian viewing direction unit vector from [theta, phi])
#     output_ch = 5 if args.N_importance > 0 else 4
#     skips = [4]
#     model_d = NeRF_d(D=args.netdepth, W=args.netwidth,
#                      input_ch=input_ch_d, output_ch=output_ch, skips=skips,
#                      input_ch_views=input_ch_views,
#                      use_viewdirsDyn=args.use_viewdirsDyn, add_features=args.add_features).to(device)

#     device_ids = list(range(torch.cuda.device_count()))
#     # model_d = torch.nn.DataParallel(model_d, device_ids=device_ids)
#     grad_vars = list(model_d.parameters())

#     embed_fn_s, input_ch_s = get_embedder(args.multires, args.i_embed, 3)
#     # 10 * 2 * 3 + 3 = 63
#     # L * (sin, cos) * (x, y, z) + (x, y, z)

#     model_s = NeRF_s(D=args.netdepth, W=args.netwidth,
#                      input_ch=input_ch_s, output_ch=output_ch, skips=skips,
#                      input_ch_views=input_ch_views,
#                      use_viewdirs=args.use_viewdirs).to(device)

#     # model_s = torch.nn.DataParallel(model_s, device_ids=device_ids)
#     grad_vars += list(model_s.parameters())

#     model_fine = None
#     if args.N_importance > 0:
#         raise NotImplementedError

#     def network_query_fn_d(inputs, viewdirs, network_fn): return run_network(
#         inputs, viewdirs, network_fn,
#         embed_fn=embed_fn_d,
#         embeddirs_fn=embeddirs_fn,
#         netchunk=args.netchunk)

#     def network_query_fn_s(inputs, viewdirs, network_fn): return run_network(
#         inputs, viewdirs, network_fn,
#         embed_fn=embed_fn_s,
#         embeddirs_fn=embeddirs_fn,
#         netchunk=args.netchunk)

#     render_kwargs_train = {
#         'network_query_fn_d': network_query_fn_d,
#         'network_query_fn_s': network_query_fn_s,
#         'network_fn_d': model_d,
#         'network_fn_s': model_s,
#         'perturb': args.perturb,
#         'N_importance': args.N_importance,
#         'N_samples': args.N_samples,
#         'use_viewdirs': args.use_viewdirs,
#         'raw_noise_std': args.raw_noise_std,
#         'inference': False,
#         'DyNeRF_blending': args.DyNeRF_blending,
#     }

#     # NDC only good for LLFF-style forward facing data
#     if args.dataset_type != 'llff' or args.no_ndc:
#         print('Not ndc!')
#         render_kwargs_train['ndc'] = False
#         render_kwargs_train['lindisp'] = args.lindisp
#     else: #here
#         render_kwargs_train['ndc'] = True

#     render_kwargs_test = {
#         k: render_kwargs_train[k] for k in render_kwargs_train}
#     render_kwargs_test['perturb'] = False
#     render_kwargs_test['raw_noise_std'] = 0.
#     render_kwargs_test['inference'] = True

#     # Create optimizer
#     optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

#     start = 0
#     basedir = args.basedir
#     expname = args.expname

#     if args.ft_path is not None and args.ft_path != 'None':
#         ckpts = [args.ft_path]
#     else: #  here
#         ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if 'tar' in f]
#     print('Found ckpts', ckpts)
#     ###Have some problems here so comment it.
#     if len(ckpts) > 2 and not args.no_reload:
#         ckpt_path = ckpts[-2]
#         print('Reloading from', ckpt_path)
#         ckpt = torch.load(ckpt_path)
#         grad_vars_d = list(render_kwargs_train['network_fn_d'].parameters())
#         optimizer = torch.optim.Adam(params=grad_vars_d, lr=args.lrate, betas=(0.9, 0.999))
#         start = ckpt['global_step'] + 1
#         optimizer.load_state_dict(ckpt['optimizer_state_dict'])
#         model_d.load_state_dict(ckpt['network_fn_d_state_dict'])
#         model_s.load_state_dict(ckpt['network_fn_s_state_dict'])
#         print('Resetting step to', start)
#         args.pretrain = False

#         if model_fine is not None:
#             raise NotImplementedError

#     return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def create_resnet_nerf(args, net, rank=0, eval_ckpt=None):
    """Instantiate NeRF's MLP model.
    """

    embed_fn_d, input_ch_d = get_embedder(args.multires, args.i_embed, 4) # embed (x, y) to higher dim.
    # 10 * 2 * 4 + 4 = 84
    # L * (sin, cos) * (x, y, z, t) + (x, y, z, t)

    input_ch_views = 0
    embeddirs_fn = None
    if args.use_viewdirs:
        embeddirs_fn, input_ch_views = get_embedder(
            args.multires_views, args.i_embed, 3)
        # 4 * 2 * 3 + 3 = 27
        # L * (sin, cos) * (3 Cartesian viewing direction unit vector from [theta, phi]) + (3 Cartesian viewing direction unit vector from [theta, phi])
    output_ch = 5 if args.N_importance > 0 else 4
    skips = [4]

    model_d = net
    if model_d.mlp_dynamic.ode_flow:
        model_d.mlp_dynamic.vector_field.embed_fn_d, input_ch_flow = get_embedder(args.multires, args.i_embed, 4)
    # model_d.mlp_dynamic.vector_field.embed_fn_d = embed_fn_d

    # grad_vars = list(model_d.encoder.parameters()) + list(model_d.mlp_dynamic.parameters())

    # embed_fn_s, input_ch_s = get_embedder(args.multires, args.i_embed, 3)
    # 10 * 2 * 3 + 3 = 63
    # L * (sin, cos) * (x, y, z) + (x, y, z)


    model_fine = None
    if args.N_importance > 0:
        raise NotImplementedError

    # Create optimizer
    # optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    
    start = 0
    basedir = args.basedir
    expname = args.expname

    if args.ft_path is not None and args.ft_path != 'None':
        ckpts = [args.ft_path]
    else: #  here
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('tar' in f and "Pretrained_S" not in f)]
    print('Found ckpts', ckpts)
    # Have some problems here so comment it.
    if len(ckpts) >= 1 and not args.no_reload:
        ckpt_path = ckpts[-1]
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
        # grad_vars_d = list(render_kwargs_train['network_fn_d'].parameters())
        grad_vars_d = []
        if args.fast_render_iter == 0:
            args.pretrain = False
            start = ckpt['global_step']
        if rank == 0:
            missing_keys, unexpected_keys, = model_d.load_state_dict(ckpt['network_fn_d_state_dict'])
            print("Missing Keys: ", missing_keys)
            print("Unexpected Keys: ", missing_keys)
        print('Resetting step to', start)

        if model_fine is not None:
            raise NotImplementedError
    if args.pretrain and args.fast_render_iter == 0:
        assert args.DyNeRF_blending == True
        ckpts = [os.path.join(basedir, expname, f) for f in sorted(os.listdir(os.path.join(basedir, expname))) if ('tar' in f and "Pretrained_S" in f)]
        if args.ft_path_S is None or args.ft_path_S == 'None' and len(ckpts) == 1:
            args.ft_path_S = ckpts[-1]
            assert "Pretrained_S" in args.ft_path_S
        if args.ft_path_S is not None and args.ft_path_S != 'None' and os.path.exists(args.ft_path_S):
            # Load Pre-trained StaticNeRF
            ckpt_path = args.ft_path_S
            print("*" * 50)
            print('Reloading StaticNeRF from' + str(ckpt_path))
            print("*" * 50)
            ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
            static_nerf_dict = {}
            for key in ckpt['network_fn_d_state_dict'].keys():
                if 'mlp_static' in key or "encoder2d" in key:
                    static_nerf_dict[key] = ckpt['network_fn_d_state_dict'][key]
            # import ipdb; ipdb.set_trace()
            if rank == 0:
                # pass
                missing_keys, unexpected_keys = model_d.load_state_dict(static_nerf_dict, strict=False)
                assert len(unexpected_keys) == 0
            args.pretrain = False        
        # else:
        #     grad_vars = list(model_d.encoder2d.parameters()) + list(model_d.mlp_static.parameters())

    model_d = build_ddp(
        model_d,
        args.device,
        default_args=dict(
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False,
            find_unused_parameters=True))

    grad_vars = list(model_d.module.encoder.parameters()) + list(model_d.module.mlp_dynamic.parameters())
    if args.pretrain:
        grad_vars = list(model_d.module.encoder2d.parameters()) + list(model_d.module.mlp_static.parameters())

    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))
    if len(ckpts) >= 2 and not args.no_reload:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    def network_query_fn_d(inputs, viewdirs, network_fn, **kwargs): return run_network(
        inputs, viewdirs, network_fn,
        embed_fn=embed_fn_d,
        embeddirs_fn=embeddirs_fn,
        netchunk=args.netchunk,
        **kwargs)


    render_kwargs_train = {
        'network_query_fn_d': network_query_fn_d,
        'network_fn_d': model_d,
        'perturb': args.perturb,
        'N_importance': args.N_importance,
        'N_samples': args.N_samples,
        'use_viewdirs': args.use_viewdirs,
        'raw_noise_std': args.raw_noise_std,
        'inference': False,
        'DyNeRF_blending': args.DyNeRF_blending,
    }

    # NDC only good for LLFF-style forward facing data
    if args.dataset_type != 'llff' or args.no_ndc:
        print('Not ndc!')
        render_kwargs_train['ndc'] = False
        render_kwargs_train['lindisp'] = args.lindisp
    else: #here
        render_kwargs_train['ndc'] = True

    render_kwargs_test = {
        k: render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    render_kwargs_test['inference'] = True

    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

# Ray helpers
def get_rays(H, W, focal, c2w): # 270, 480,
    """Get ray origins, directions from a pinhole camera."""
    i, j = torch.meshgrid(torch.linspace(0, W-1, W,), torch.linspace(0, H-1, H)) # pytorch's meshgrid has indexing='ij'
    i = i.t().cuda()
    j = j.t().cuda()
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1) #270, 480, 3
    # Rotate ray directions from camera frame to the world frame
    rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1) # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.
    Space such that the canvas is a cube with sides [-1, 1] in each axis.
    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.
    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    H, W = H.cuda(), W.cuda()
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # For all the rays from H * W image
    if len(rays_o.shape) == 4:
        focal = focal[..., None]
        H, W = H[..., None], W[..., None]

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
    (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def get_grid(H, W, num_img, flows_f, flow_masks_f, flows_b, flow_masks_b):

    # |--------------------|  |--------------------|
    # |       j            |  |       v            |
    # |   i   *            |  |   u   *            |
    # |                    |  |                    |
    # |--------------------|  |--------------------|

    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')

    grid = np.empty((0, H, W, 8), np.float32)
    for idx in range(num_img):
        grid = np.concatenate((grid, np.stack([i,
                                               j,
                                               flows_f[idx, :, :, 0],
                                               flows_f[idx, :, :, 1],
                                               flow_masks_f[idx, :, :],
                                               flows_b[idx, :, :, 0],
                                               flows_b[idx, :, :, 1],
                                               flow_masks_b[idx, :, :]], -1)[None, ...]))
    return grid


def NDC2world(pts, H, W, f):

    # NDC coordinate to world coordinate
    while len(H.shape) != len(pts.shape):
        H, W, f = H.unsqueeze(-1), W.unsqueeze(-1), f.unsqueeze(-1)
    pts_z = 2 / (torch.clamp(pts[..., 2:], min=-1., max=1-1e-3) - 1)
    pts_x = - pts[..., 0:1] * pts_z * W / 2 / f
    pts_y = - pts[..., 1:2] * pts_z * H / 2 / f
    pts_world = torch.cat([pts_x, pts_y, pts_z], -1)

    return pts_world


def render_3d_point(H, W, f, pose, weights, pts):
    """Render 3D position along each ray and project it to the image plane.
    """

    c2w = pose
    w2c = c2w[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)

    # NDC coordinate to world coordinate
    pts_map_world = NDC2world(pts_map_NDC, H, W, f)

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[:, 3]
    # Rotate
    pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                         - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                         -1)

    return pts_plane

def render_3d_point_wo_ndc(H, W, f, pose, weights, pts):
    """Render 3D position along each ray and project it to the image plane.
    """

    c2w = pose
    w2c = c2w[:3, :3].transpose(0, 1) # same as np.linalg.inv(c2w[:3, :3])

    # Rendered 3D position in NDC coordinate
    pts_map_NDC = torch.sum(weights[..., None] * pts, -2)

    # NDC coordinate to world coordinate
    # pts_map_world = NDC2world(pts_map_NDC, H, W, f)
    pts_map_world = pts_map_NDC

    # World coordinate to camera coordinate
    # Translate
    pts_map_world = pts_map_world - c2w[:, 3]
    # Rotate
    pts_map_cam = torch.sum(pts_map_world[..., None, :] * w2c[:3, :3], -1)

    # Camera coordinate to 2D image coordinate
    pts_plane = torch.cat([pts_map_cam[..., 0:1] / (- pts_map_cam[..., 2:]) * f + W * .5,
                         - pts_map_cam[..., 1:2] / (- pts_map_cam[..., 2:]) * f + H * .5],
                         -1)

    return pts_plane


def induce_flow_wo_ndc(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor = render_3d_point_wo_ndc(H, W, focal,
                                      pose_neighbor,
                                      weights,
                                      pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow

def induce_flow(H, W, focal, pose_neighbor, weights, pts_3d_neighbor, pts_2d):

    # Render 3D position along each ray and project it to the neighbor frame's image plane.
    pts_2d_neighbor = render_3d_point(H, W, focal,
                                      pose_neighbor,
                                      weights,
                                      pts_3d_neighbor)
    induced_flow = pts_2d_neighbor - pts_2d

    return induced_flow


def compute_depth_loss(dyn_depth, gt_depth):

    t_d = torch.median(dyn_depth)
    s_d = torch.mean(torch.abs(dyn_depth - t_d)) + 1e-9 # For numerical statement.
    # s_d = torch.mean(torch.abs(dyn_depth - t_d))
    dyn_depth_norm = (dyn_depth - t_d) / s_d

    t_gt = torch.median(gt_depth)
    s_gt = torch.mean(torch.abs(gt_depth - t_gt))
    gt_depth_norm = (gt_depth - t_gt) / s_gt

    return torch.mean((dyn_depth_norm - gt_depth_norm) ** 2)

def compute_depth_blending_loss(gt_depth, z_vals, blending_map, near=0.99, far=1.75, min_depth=None, max_depth=None, M=None, thickness=0.06):

    if min_depth == None:
        min_depth = gt_depth.min()
    if max_depth == None:
        max_depth = gt_depth.max()

    gt_depth -= min_depth
    gt_depth /= max_depth - min_depth
    gt_depth *= far - near
    gt_depth += near

    gt_blending_map = z_vals
    gt_depth = gt_depth[..., None].expand(gt_blending_map.shape)
    gt_blending_map[torch.abs(gt_blending_map - gt_depth) > thickness] = 0.

    return img2mse(gt_blending_map, blending_map, M)

    # if M == None:
    #     return torch.mean((gt_blending_map - blending_map) ** 2)
    # return torch.mean((gt_blending_map - blending_map) ** 2 * M) / (torch.sum(M) + 1e-9) / gt_blending_map.shape[-1]

def compute_static_depth_blending_loss(gt_depth, z_vals, blending, near=0.99, far=1.75, min_depth=None, max_depth=None, M=None, thickness=0.06):

    '''
    Force the static part blendings in dynamic scene to zero.
    '''
    if min_depth == None:
        min_depth = gt_depth.min()
    if max_depth == None:
        max_depth = gt_depth.max()

    gt_depth -= min_depth
    gt_depth /= max_depth - min_depth
    gt_depth *= far - near
    gt_depth += near

    gt_blending_map = torch.ones(z_vals.shape, device=device)
    gt_depth = gt_depth[..., None].expand(gt_blending_map.shape)
    gt_blending_map[torch.abs(z_vals - gt_depth) > (far - near) * thickness] = 0.

    M = M * (1 - gt_blending_map)

    return torch.sum((blending ** 2) * M) / (torch.sum(M) + 1e-8)
    # return img2mse(torch.zeros(z_vals.shape, device=device), blending, M)

def normalize_depth(depth, near=0.):
    depth -= near
    return torch.clamp(depth / percentile(depth, 97), 0., 1.)


def percentile(t, q):
    """
    Return the ``q``-th percentile of the flattened input tensor's data.

    CAUTION:
     * Needs PyTorch >= 1.1.0, as ``torch.kthvalue()`` is used.
     * Values are not interpolated, which corresponds to
       ``numpy.percentile(..., interpolation="nearest")``.

    :param t: Input tensor.
    :param q: Percentile to compute, which must be between 0 and 100 inclusive.
    :return: Resulting value (scalar).
    """

    k = 1 + round(.01 * float(q) * (t.numel() - 1))
    result = t.view(-1).kthvalue(k).values.item()
    return result


def save_res(moviebase, ret, bs=1, fps=None):

    if fps == None:
        if len(ret['rgbs'][0]) < 25:
            fps = 4
        else:
            fps = 24
    for b in range(bs):
        for k in ret:
            if 'rgbs' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b]), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(ret[k]), format='gif', fps=fps)
            elif 'depths' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b]), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(ret[k]), format='gif', fps=fps)
            elif 'disps' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b] / np.max(ret[k][b])), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(ret[k] / np.max(ret[k])), format='gif', fps=fps)
            elif 'sceneflow_' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(norm_sf(ret[k][b])), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(norm_sf(ret[k])), format='gif', fps=fps)
            elif 'flows' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                ret[k][b], fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  ret[k], format='gif', fps=fps)
            elif 'dynamicness' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b]), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(ret[k]), format='gif', fps=fps)
            elif 'disocclusions' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b][..., 0]), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(ret[k][..., 0]), format='gif', fps=fps)
            elif 'blending' in k:
                blending = ret[k][b][..., None]
                blending = np.moveaxis(blending, [0, 1, 2, 3], [1, 2, 0, 3])
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(blending), fps=fps, quality=8, macro_block_size=1)
                # imageio.mimsave(moviebase + k + '.gif',
                #                  to8b(blending), format='gif', fps=fps)
            elif 'weights' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                to8b(ret[k][b]), fps=fps, quality=8, macro_block_size=1)
            
            elif 'vis_sem_labels' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                ret[k][b].astype(np.uint8), fps=fps, quality=8, macro_block_size=1)
            
            elif 'vis_sem_uncertainties' in k:
                imageio.mimwrite(moviebase + k + '_{:02d}.mp4'.format(b),
                                ret[k][b].astype(np.uint8), fps=fps, quality=8, macro_block_size=1) 
            else:
                raise NotImplementedError


def norm_sf_channel(sf_ch):

    # Make sure zero scene flow is not shifted
    sf_ch[sf_ch >= 0] = sf_ch[sf_ch >= 0] / sf_ch.max() / 2
    sf_ch[sf_ch < 0] = sf_ch[sf_ch < 0] / np.abs(sf_ch.min()) / 2
    sf_ch = sf_ch + 0.5
    return sf_ch


def norm_sf(sf):

    sf = np.concatenate((norm_sf_channel(sf[..., 0:1]),
                         norm_sf_channel(sf[..., 1:2]),
                         norm_sf_channel(sf[..., 2:3])), -1)
    sf = np.moveaxis(sf, [0, 1, 2, 3], [1, 2, 0, 3])
    return sf


# Spatial smoothness (adapted from NSFF)
def compute_sf_smooth_s_loss(pts1, pts2, H, W, f):

    N_samples = pts1.shape[2]

    # NDC coordinate to world coordinate
    pts1_world = NDC2world(pts1[..., :int(N_samples * 0.95), :], H, W, f)
    pts2_world = NDC2world(pts2[..., :int(N_samples * 0.95), :], H, W, f)

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])

def compute_sf_smooth_s_loss_wo_ndc(pts1, pts2, H, W, f):

    N_samples = pts1.shape[2]

    # NDC coordinate to world coordinate
    pts1_world = pts1[..., :int(N_samples * 0.95), :]
    pts2_world = pts2[..., :int(N_samples * 0.95), :]

    # scene flow in world coordinate
    scene_flow_world = pts1_world - pts2_world

    return L1(scene_flow_world[..., :-1, :] - scene_flow_world[..., 1:, :])

# Temporal smoothness
def compute_sf_smooth_loss(pts, pts_f, pts_b, H, W, f):

    N_samples = pts.shape[2]

    pts_world   = NDC2world(pts[..., :int(N_samples * 0.9), :],   H, W, f)
    pts_f_world = NDC2world(pts_f[..., :int(N_samples * 0.9), :], H, W, f)
    pts_b_world = NDC2world(pts_b[..., :int(N_samples * 0.9), :], H, W, f)

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L1(sceneflow_f + sceneflow_b)

def compute_sf_smooth_loss_wo_ndc(pts, pts_f, pts_b, H, W, f):

    N_samples = pts.shape[2]
    # WHY????????????????? 让光路上前百分之90的点计算场景流？
    pts_world   = pts[..., :int(N_samples * 0.9), :]
    pts_f_world = pts_f[..., :int(N_samples * 0.9), :]
    pts_b_world = pts_b[..., :int(N_samples * 0.9), :]

    # scene flow in world coordinate
    sceneflow_f = pts_f_world - pts_world
    sceneflow_b = pts_b_world - pts_world

    # For a 3D point, its forward and backward sceneflow should be opposite.
    return L2(sceneflow_f + sceneflow_b)


def compute_semantic_loss(pred_sem_label, gt_sem_label, ignore_label=-1):
    # logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)

    N_batch, N_ray, N_class = pred_sem_label.shape
    pred_sem_label, gt_sem_label = pred_sem_label.reshape(N_batch * N_ray, N_class), gt_sem_label.reshape(N_batch * N_ray)

    # pred_sem_label = logits_2_label(pred_sem_map)

    return crossentropy_loss(pred_sem_label, gt_sem_label, ignore_index=ignore_label)


def compute_semantic_uncertainty(pred_sem_label):
    sem_uncertainty = logits2uncertainty(pred_sem_label)
    sem_uncertainty = sem_uncertainty.cpu().numpy().reshape((sem_uncertainty.shape[:-1]))
    vis_sem_uncertainty = depth2rgb(sem_uncertainty)
    sem_uncertainty = to8b(sem_uncertainty)
    return sem_uncertainty, vis_sem_uncertainty