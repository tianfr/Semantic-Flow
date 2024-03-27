import torch
#  import torch_scatter
import torch.autograd.profiler as profiler
import torch.nn.functional as F
from torch import nn

from src import util


def worldpts2NDCpts(pts, H=270., W=480., f=418.9622):
    """Convert world points to NDC points.

    Args:
        pts (array of shape [..., 3]): World coordinates.
        H (float, optional): Height in pixels. Defaults to 270..
        W (float, optional): Width in pixels. Defaults to 480..
        f (float, optional): Focal length of pinhole camera. Defaults to 418.9622.

    Returns:
        ndc: NDC coordinates of shape [..., 3].
    """
    #! near 1.0 far inf.
    ax = -2 * f / W
    ay = -2 * f / H
    az = 1
    bz = 2
    worldx, worldy, worldz = torch.split(pts, [1, 1, 1], dim=-1)
    ndcx = ax * worldx / worldz
    ndcy = ay * worldy / worldz
    ndcz = az + bz / worldz

    return torch.cat([ndcx, ndcy, ndcz], dim=-1)

class MLP_static(nn.Module):
    def __init__(
        self,
        d_in,
        d_out=4,
        n_blocks=8,
        d_latent=2048,
        d_hidden=128,
        beta=0.0,
        combine_layer=1000,
        combine_type='average',
        use_spade=False,
        # D=8,
        # W=256,
        input_ch=63,
        input_ch_views=27,
        output_ch=4,
        skips=[4],
        use_viewdirs=True,
        add_features=False,
        pixelnerf_mode=False,
        use_sf_nonlinear = False,
        debug=False,
    ):
        """
        :param d_in input size
        :param d_out output size
        :param n_blocks number of Resnet blocks
        :param d_latent latent size, added in each resnet block (0 = disable)
        :param d_hidden hiddent dimension throughout network
        :param beta softplus beta, 100 is reasonable; if <=0 uses ReLU activations instead
        """
        super().__init__()
        # if use_viewdirs:
        #     d_in = input_ch
        # else:
        #     d_in = input_ch + input_ch_views
        d_in = input_ch
        d_out = output_ch

        assert pixelnerf_mode == False

        self.n_blocks = n_blocks
        self.d_latent = d_latent
        self.d_in = d_in
        self.d_out = d_out
        self.d_hidden = d_hidden

        self.combine_layer = combine_layer
        self.combine_type = combine_type
        self.use_spade = use_spade

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.add_features = add_features
        self.use_sf_nonlinear = use_sf_nonlinear
        self.pixelnerf_mode = pixelnerf_mode
        # if self.pixelnerf_mode:
        #     self.add_features = False
        #     print("PixelNeRF mode.")
        # import ipdb; ipdb.set_trace()
        if self.add_features and not self.pixelnerf_mode:
            if d_latent == 0:
                raise ValueError('d_latent is ZERO!')
            self.feature_embedding = nn.Linear(d_latent, 128)
            self.d_in += 128
        else:
            self.feature_embedding = None

        self.blocks = nn.ModuleList(
            [nn.Linear(d_hidden, d_hidden) if i not in self.skips else nn.Linear(self.d_in+d_hidden, d_hidden) for i in range(n_blocks-1)]
        )



        if d_in > 0:
            self.lin_in = nn.Linear(self.d_in, d_hidden)
            # nn.init.constant_(self.lin_in.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_in.weight, a=0, mode="fan_in")


        if self.use_viewdirs:
            self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + d_hidden, d_hidden//2)])
            self.feature_linear = nn.Linear(d_hidden, d_hidden)
            self.alpha_linear = nn.Linear(d_hidden, 1)
            self.rgb_linear = nn.Linear(d_hidden//2, 3)
            # nn.init.constant_(self.views_linears[0].bias, 0.0)
            # nn.init.kaiming_normal_(self.views_linears[0].weight, a=0, mode="fan_in")
            # nn.init.constant_(self.feature_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.feature_linear.weight, a=0, mode="fan_in")
            # nn.init.constant_(self.alpha_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.alpha_linear.weight, a=0, mode="fan_in")
            # nn.init.constant_(self.rgb_linear.bias, 0.0)
            # nn.init.kaiming_normal_(self.rgb_linear.weight, a=0, mode="fan_in")

        else:
            # self.output_linear = nn.Linear(d_hidden, output_ch)
            self.lin_out = nn.Linear(d_hidden, d_out)
            # nn.init.constant_(self.lin_out.bias, 0.0)
            # nn.init.kaiming_normal_(self.lin_out.weight, a=0, mode="fan_in")

        self.encoder_features = None

        self.weight_linear = nn.Linear(d_hidden, 1)

        # if d_latent != 0 and self.pixelnerf_mode:
        #     n_lin_z = min(combine_layer, n_blocks)
        #     self.lin_z = nn.ModuleList(
        #         [nn.Linear(d_latent, d_hidden) for i in range(n_lin_z)]
        #     )
        #     for i in range(n_lin_z):
        #         nn.init.constant_(self.lin_z[i].bias, 0.0)
        #         nn.init.kaiming_normal_(self.lin_z[i].weight, a=0, mode="fan_in")

        #     if self.use_spade:
        #         self.scale_z = nn.ModuleList(
        #             [nn.Linear(d_latent, d_hidden) for _ in range(n_lin_z)]
        #         )
        #         for i in range(n_lin_z):
        #             nn.init.constant_(self.scale_z[i].bias, 0.0)
        #             nn.init.kaiming_normal_(self.scale_z[i].weight, a=0, mode="fan_in")

        if beta > 0:
            self.activation = nn.Softplus(beta=beta)
        else:
            self.activation = nn.ReLU()

        self.debug = debug
        if self.debug:
            self.clock = 0
            self.print_clock = 100

    def forward(self, x, combine_inner_dims=(1,), combine_index=None, dim_size=None):
        """
        :param zx (..., d_latent + d_in)
        :param combine_inner_dims Combining dimensions for use with multiview inputs.
        Tensor will be reshaped to (-1, combine_inner_dims, ...) and reduced using combine_type
        on dim 1, at combine_layer
        """
        if self.d_latent != 0:
            x, latent = torch.split(x, [self.input_ch + self.input_ch_views, self.d_latent], dim=-1)
        # if self.use_viewdirs:
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        x = input_pts
        if self.add_features:
            encoder_features = self.feature_embedding(latent)
            x = torch.cat([x, encoder_features], -1)
        with profiler.record_function('resnetfc_infer'):
            if self.d_latent > 0 and self.pixelnerf_mode:
                z = latent
                x = x
            else:
                x = x
            # x = F.normalize(x, dim=-1)
            if self.d_in > 0:
                x = self.lin_in(x)
                x = F.relu(x)
            else:
                x = torch.zeros(self.d_hidden, device=x.device)

            for blkid in range(self.n_blocks - 1):
                if blkid in self.skips:
                    if self.add_features:
                        x = torch.cat([encoder_features, x], -1)
                    x = torch.cat([input_pts, x], -1)

                x = self.blocks[blkid](x)
                x = F.relu(x)

            # print("prev: ", torch.abs(x).mean().item(), torch.abs(x).std().item())
            # x = F.normalize(x)
            # print("after: ",torch.abs(x).mean().item(), torch.abs(x).std().item())
            # import ipdb; ipdb.set_trace()
            # sf = torch.tanh(self.sf_linear(F.normalize(x, p=1)))
            if self.debug:
                normx = F.normalize(x)
                dis = torch.sum(torch.abs(x), dim=1)
                self.clock += 1
                if self.clock % 100 == 0:
                    print('step: ', self.clock, dis.mean().item(), dis.std().item())
            # import ipdb; ipdb.set_trace()
            # sf = torch.tanh(self.sf_linear(F.normalize(x)))
            # sf = torch.tanh(self.sf_linear(x))
            # convert world coord to ndc coord.
            # sf = sf.reshape([sf.shape[0], -1, 3])
            # sf = worldpts2NDCpts(sf).reshape([sf.shape[0], -1])

            # if self.debug and self.clock % 100 == 0:
            #     print("sf: ", torch.abs(sf).sum(dim=1).mean().item())
            blending = torch.sigmoid(self.weight_linear(x))
            # print("blending: ", torch.abs(blending).mean().item())
            if self.use_viewdirs:
                # raise NotImplementedError
                alpha = self.alpha_linear(x)

                feature = self.feature_linear(x)
                x = torch.cat([feature, input_views], -1)

                for i, l in enumerate(self.views_linears):
                    x = self.views_linears[i](x)
                    x = F.relu(x)

                rgb = self.rgb_linear(x)
                out = torch.cat([rgb, alpha], -1)
            else:
                out = self.lin_out(x)

            return torch.cat([out, blending], dim=-1)

    @classmethod
    def from_conf(cls, conf, d_in, **kwargs):
        # PyHocon construction
        return cls(
            d_in,
            n_blocks=conf.get_int('n_blocks', 8),
            d_hidden=conf.get_int('d_hidden', 128),
            beta=conf.get_float('beta', 0.0),
            combine_layer=conf.get_int('combine_layer', 1000),
            combine_type=conf.get_string('combine_type', 'average'),  # average | max
            use_viewdirs=conf.get_bool('use_viewdirs', True),
            use_spade=conf.get_bool('use_spade', False),
            pixelnerf_mode=conf.get_bool('pixelnerf_mode', False),
            **kwargs
        )

if __name__ == '__main__':
    resnetfc = Nerf_s(
        d_in=84,
        n_blocks=5,
        d_hidden=512,
        beta=0.0,
        combine_layer=3,
        combine_type='average',  # average | max
        use_spade=False,
        d_latent=2048,
        d_out=4,
        )
    bs = 16
    x = torch.randn([bs, 84+2048])
    import ipdb; ipdb.set_trace()
    y = resnetfc(x)
