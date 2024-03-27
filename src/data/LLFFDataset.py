import time

start = time.time()
import glob
import io
import os
# from petrel_client.client import Client
import time
from os.path import splitext

import cv2
import imageio
import ipdb
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib.pyplot import grid

# from util import get_image_to_tensor_balanced, get_mask_to_tensor


if __name__ == '__main__':
    print(time.time() - start)

import copy

from mmaction.datasets.pipelines import Compose
from mmcv import Config

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
DEFAULT_PIPELINE = [
    # dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    #! If use scale=(img_w, img_h), which is contradictary to the img.shape=[img_h, img_w]
    # dict(type='Resize', scale=(480, 270), keep_ratio=False), #!Bug fixed
    # dict(type='Flip', flip_ratio=0.5),
    # dict(type='ColorJitter', brightness=0.5, contrast=0.0, saturation=0.0, hue=0.0),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='FormatShape', input_format='NTHWC'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs'])
]

COLOR_JITTOR_PIPELINE = [
    # dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    #! If use scale=(img_w, img_h), which is contradictary to the img.shape=[img_h, img_w]
    # dict(type='Resize', scale=(480, 270), keep_ratio=False), #!Bug fixed
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='FormatShape', input_format='NTHWC'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs'])
]
ONLY_COLOR_JITTOR_PIPELINE = [
    # dict(type='SampleFrames', clip_len=8, frame_interval=8, num_clips=1),
    # dict(type='RawFrameDecode'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='RandomResizedCrop'),
    #! If use scale=(img_w, img_h), which is contradictary to the img.shape=[img_h, img_w]
    # dict(type='Resize', scale=(480, 270), keep_ratio=False), #!Bug fixed
    # dict(type='Flip', flip_ratio=0.5),
    dict(type='ColorJitter', brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='FormatShape', input_format='NTHWC'),
    # dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    # dict(type='ToTensor', keys=['imgs'])
]



def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses

def poses_avg(poses):

    hwf = poses[0, :3, -1:]

    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)

    return c2w

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def normalize(x):
    return x / np.linalg.norm(x)

def resize_flow(flow, H_new, W_new):
    H_old, W_old = flow.shape[0:2]
    flow_resized = cv2.resize(flow, (W_new, H_new), interpolation=cv2.INTER_LINEAR)
    flow_resized[:, :, 0] *= H_new / H_old
    flow_resized[:, :, 1] *= W_new / W_old
    return flow_resized

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True, use_semantic_labels=False, flow_net=False):
    # print('factor ', factor)
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    bds = poses_arr[:, -2:].transpose([1,0])

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape

    sfx = ''

    if factor is not None:
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        if width % 2 == 1:
            width -= 1
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        if height % 2 == 1:
            height -= 1
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return

    sh = imageio.imread(imgfiles[0]).shape #(270, 480, 3)
    num_img = len(imgfiles)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    poses[2, 4, :] = poses[2, 4, :] * 1./factor

    if not load_imgs:
        return poses, bds

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
    encoder_imgs = [imread(f)[..., :3] for f in imgfiles]
    imgs = np.stack(imgs, -1) # (270, 480, 3, 12)
    encoder_imgs = np.stack(encoder_imgs, -1) # (270, 480, 3, 12)

    assert imgs.shape[0] == sh[0]
    assert imgs.shape[1] == sh[1]

    disp_dir = os.path.join(basedir, 'disp')

    dispfiles = [os.path.join(disp_dir, f) \
                for f in sorted(os.listdir(disp_dir)) if f.endswith('npy')]

    disp = [cv2.resize(np.load(f),
                    (sh[1], sh[0]),
                    interpolation=cv2.INTER_NEAREST) for f in dispfiles]
    disp = np.stack(disp, -1) # (270, 480, 12)

    mask_dir = os.path.join(basedir, 'motion_masks')
    maskfiles = [os.path.join(mask_dir, f) \
                for f in sorted(os.listdir(mask_dir)) if f.endswith('png')]

    masks = [cv2.resize(imread(f)/255., (sh[1], sh[0]),
                        interpolation=cv2.INTER_NEAREST) for f in maskfiles]
    masks = np.stack(masks, -1)

    masks = np.float32(masks > 1e-3) # (270, 380, 12)
    # Warning("!!!!!!!!!!!!!!!!!!!!!!")
    # Warning("Here we use background mask !!!!!!!!!!!!!!!!!!!!!!")
    # Warning("!!!!!!!!!!!!!!!!!!!!!!")
    # import ipdb; ipdb.set_trace()
    # masks = np.float32(masks < 1e-3) # (270, 380, 12)
    if flow_net:
        flow_dir = os.path.join(basedir, 'flownet_flow')
    else:
        flow_dir = os.path.join(basedir, 'flow')
    flows_f = []
    flow_masks_f = []
    flows_b = []
    flow_masks_b = []
    for i in range(num_img):
        if i == num_img - 1:
            fwd_flow, fwd_mask = np.zeros((sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
        else:
            if flow_net:
                fwd_flow_path = os.path.join(flow_dir, '%03d_fwd.npy'%i)
                fwd_data = np.load(fwd_flow_path)
                fwd_flow = fwd_data
                fwd_mask = np.zeros(fwd_flow.shape[:-1])
            else:
                fwd_flow_path = os.path.join(flow_dir, '%03d_fwd.npz'%i)
                fwd_data = np.load(fwd_flow_path)
                fwd_flow, fwd_mask = fwd_data['flow'], fwd_data['mask']
            fwd_flow = resize_flow(fwd_flow, sh[0], sh[1])
            fwd_mask = np.float32(fwd_mask)
            fwd_mask = cv2.resize(fwd_mask, (sh[1], sh[0]),
                                interpolation=cv2.INTER_NEAREST)
        flows_f.append(fwd_flow)
        flow_masks_f.append(fwd_mask)

        if i == 0:
            bwd_flow, bwd_mask = np.zeros((sh[0], sh[1], 2)), np.zeros((sh[0], sh[1]))
        else:
            
            if flow_net:
                bwd_flow_path = os.path.join(flow_dir, '%03d_bwd.npy'%i)
                bwd_data = np.load(bwd_flow_path)
                bwd_flow = bwd_data
                bwd_mask = np.zeros(bwd_flow.shape[:-1])
            else:
                bwd_flow_path = os.path.join(flow_dir, '%03d_bwd.npz'%i)
                bwd_data = np.load(bwd_flow_path)
                bwd_flow, bwd_mask = bwd_data['flow'], bwd_data['mask']
            bwd_flow = resize_flow(bwd_flow, sh[0], sh[1])
            bwd_mask = np.float32(bwd_mask)
            bwd_mask = cv2.resize(bwd_mask, (sh[1], sh[0]),
                                interpolation=cv2.INTER_NEAREST)
        flows_b.append(bwd_flow)
        flow_masks_b.append(bwd_mask)

    flows_f = np.stack(flows_f, -1)
    flow_masks_f = np.stack(flow_masks_f, -1)
    flows_b = np.stack(flows_b, -1)
    flow_masks_b = np.stack(flow_masks_b, -1)

    if use_semantic_labels:
        semantic_labels = {}
        semantic_dir = os.path.join(basedir, "SegmentationClass_train")
        semantic_label_files = [os.path.join(semantic_dir, f) for f in sorted(os.listdir(semantic_dir)) \
                if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
        semantic_label_imgs = [cv2.resize(imread(f)[..., :3], (sh[1], sh[0]),
                        interpolation=cv2.INTER_NEAREST) for f in semantic_label_files]
        semantic_label_imgs = np.stack(semantic_label_imgs, -1)
        semantic_label_logits = np.zeros(semantic_label_imgs.shape[:2] + (semantic_label_imgs.shape[-1],))

        masks = np.zeros(semantic_label_imgs.shape[:2] + (semantic_label_imgs.shape[-1],))
        box2semantic_labels = np.zeros(semantic_label_imgs.shape[:2] + (semantic_label_imgs.shape[-1],))
        label_map_path = os.path.join(os.path.dirname(basedir), "labelmap.txt")
        with open(label_map_path, 'r') as f:
            label_map_content = f.readlines()[1:]

        foreground_label_path = os.path.join(os.path.dirname(basedir), "foreground_object.txt")
        with open(foreground_label_path, 'r') as f:
            foreground_labels = f.read().strip().split("\n")

        label_maps = {}
        label_maps_array = []
        foreground_boxes = {}
        for idx, label in enumerate(label_map_content):
            label_name, label_color = label.strip().strip(":").split(":")
            label_color = np.array([int(x) for x in label_color.split(",")])
            label_maps[label_name] = dict(
                color = label_color,
                idx = idx,
            )
            label_maps_array.append(label_color)

            _label_pos = ((semantic_label_imgs[:, :, 0] == label_color[0]) & (semantic_label_imgs[:, :, 1] == label_color[1]) & (semantic_label_imgs[:, :, 2] == label_color[2]))
            # _color_logic = (semantic_label_imgs[:,:,:,:] == label_color[None, None, :, None])
            # _color_logic = _color_logic[:,:, 0] & _color_logic[:, :, 1] & _color_logic[:, :, 2]
            semantic_label_logits[_label_pos] = idx

            if label_name in foreground_labels:
                
                #! Exclude the static person in Balloon1
                if basedir.split("/")[-1] == "Balloon1" and label_name == "person":
                    print("The person in Balloon1 is static!")
                    continue
                masks[_label_pos] = 1

                # Draw BBox
                if (_label_pos == False).all():
                    continue
                curr_label_imgs = np.zeros(semantic_label_imgs.shape[:2] + (semantic_label_imgs.shape[-1],)).astype(np.uint8)
                # curr_label_imgs[_label_pos[:,:, None, :].repeat(3,axis=2)] = 1
                curr_label_imgs[_label_pos] = 1

                boxes = []
                for i in range(curr_label_imgs.shape[-1]):
                    curr_label_img = curr_label_imgs[...,i]
                    curr_semantic_label_img = semantic_label_imgs[...,i].copy()
                
                    contours, hierarchy = cv2.findContours(curr_label_img[...,None], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                    # Merge All contours
                    cnt = np.concatenate(contours,axis=0)

                    bounding_box = cv2.boundingRect(cnt)
                    boxes.append(bounding_box)
                    [x , y, w, h] = bounding_box

                    center_x = x + w//2
                    center_y = y + h//2
                    box2semantic_labels[center_y-5:center_y+5, center_x-5:center_x+5, i] = idx

                    cv2.rectangle(curr_semantic_label_img, (x, y), (x + w, y + h), (255, 255, 0), 3)
                    curr_semantic_label_img[center_y-5:center_y+5, center_x-5:center_x+5] = (255, 255, 0)
 
                    # bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]
                    # if len(bounding_boxes) != 1:
                    #     import ipdb; ipdb.set_trace()

                    # for bbox in bounding_boxes:
                    #     [x , y, w, h] = bbox
                    #     cv2.rectangle(curr_semantic_label_img, (x, y), (x + w, y + h), (255, 255, 0), 3)
                    
                    # cv2.imwrite(f'img_bbox_test{i}.png', curr_semantic_label_img, )

                foreground_boxes[label_name] = boxes

        semantic_labels["label_imgs"] = semantic_label_imgs
        semantic_labels["label_maps"] = label_maps
        semantic_labels["label_logits"] = semantic_label_logits
        semantic_labels["label_maps_array"] = np.array(label_maps_array)
        semantic_labels['box2semantic_labels'] = box2semantic_labels
        # import ipdb; ipdb.set_trace()

        masks = np.float32(masks > 1e-3)
    else:
        semantic_labels = {}

    # print(imgs.shape)
    # print(disp.shape)
    # print(masks.shape)
    # print(flows_f.shape)
    # print(flow_masks_f.shape)
    # (270, 480, 3, 12)
    # (270, 480, 12)
    # (270, 480, 12)
    # (270, 480, 2, 12)
    # (270, 480, 12)

    assert(imgs.shape[0] == disp.shape[0])
    assert(imgs.shape[0] == masks.shape[0])
    assert(imgs.shape[0] == flows_f.shape[0])
    assert(imgs.shape[0] == flow_masks_f.shape[0])

    assert(imgs.shape[1] == disp.shape[1])
    assert(imgs.shape[1] == masks.shape[1])

    return poses, bds, imgs, encoder_imgs, disp, masks, flows_f, flow_masks_f, flows_b, flow_masks_b, semantic_labels

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from shutil import copy
    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')


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


class LLFFDataset(torch.utils.data.Dataset):
    """
    Dataset from LLFF.
    """

    def __init__(
        self,
        path,
        pipeline=DEFAULT_PIPELINE,
        frame2dolly=-1,
        factor=8,
        spherify=False,
        num_novelviews=60,
        focal_decrease=200,
        z_trans_multiplier=5.,
        x_trans_multiplier=1.,
        y_trans_multiplier=0.33,
        no_ndc=False,
        stage='train',
        list_prefix='softras_',
        image_size=None,
        sub_format='shapenet',
        scale_focal=True,
        max_imgs=100000,
        z_near=0.0,
        z_far=1.0,
        skip_step=None,
        file_lists=None,
        use_occlusion=False,
        use_semantic_labels=False,
        add_flow_noise=0,
        use_flow_net_flow=False
    ):
        """
        :param path dataset root path, contains metadata.yml
        :param stage train | val | test
        :param list_prefix prefix for split lists: <list_prefix>[train, val, test].lst
        :param image_size result image size (resizes if different); None to keep original size
        :param sub_format shapenet | dtu dataset sub-type.
        :param scale_focal if true, assume focal length is specified for
        image of side length 2 instead of actual image size. This is used
        where image coordinates are placed in [-1, 1].
        """
        super().__init__()
        # import ipdb; ipdb.set_trace()
        # conf_path = '~/petreloss.conf'
        # oss_path = "s3://tianfengrui/dataset/rs_dtu_4/DTU/"
        # self.client = Client(conf_path)
        self.base_path = path
        # assert os.path.exists(self.base_path)

        if file_lists == None:
            self.file_lists = [x for x in glob.glob(os.path.join(path, '*')) if os.path.isdir(x)]
        elif len(file_lists) == 1 and file_lists[0] == "ALL":
            path = "./data"
            self.file_lists = [x for x in glob.glob(os.path.join(path, '*')) if os.path.isdir(x)]
        else:
            self.file_lists = file_lists
        # for Balloon2 scence testing
        # self.file_lists = ["data/Balloon2"]
        # for Balloon2 and truck scenes testing
        # self.file_lists = ["data/Balloon2", "data/Balloon1", "data/Truck"]

        # if stage == "train":
        #     file_lists = [os.path.join(x, list_prefix + "train.lst") for x in cats]
        # elif stage == "val":
        #     file_lists = [os.path.join(x, list_prefix + "val.lst") for x in cats]
        # elif stage == "test":
        #     file_lists = [os.path.join(x, list_prefix + "test.lst") for x in cats]

        self.frame2dolly = frame2dolly
        self.factor = factor
        self.spherify = spherify
        self.no_ndc = no_ndc
        self.num_novelviews =  num_novelviews
        self.focal_decrease = focal_decrease
        self.x_trans_multiplier = x_trans_multiplier
        self.y_trans_multiplier = y_trans_multiplier
        self.z_trans_multiplier =  z_trans_multiplier

        # all_objs = []
        # for file_list in file_lists:
        #     if not os.path.exists(file_list):
        #         continue
        #     base_dir = os.path.dirname(file_list)
        #     cat = os.path.basename(base_dir)
        #     with open(file_list, "r") as f:
        #         objs = [(cat, os.path.join(oss_path, x.strip())) for x in f.readlines()]
        #     all_objs.extend(objs)

        # self.all_objs = all_objs
        # self.stage = stage

        # self.image_to_tensor = get_image_to_tensor_balanced()
        # self.mask_to_tensor = get_mask_to_tensor()
        # print(
        #     "Loading DVR dataset",
        #     self.base_path,
        #     "stage",
        #     stage,
        #     len(self.all_objs),
        #     "objs",
        #     "type:",
        #     sub_format,
        # )

        self.image_size = image_size
        if sub_format == 'dtu':
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        else:
            self._coord_trans_world = torch.tensor(
                [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
            self._coord_trans_cam = torch.tensor(
                [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
                dtype=torch.float32,
            )
        self.sub_format = sub_format
        self.scale_focal = scale_focal
        self.max_imgs = max_imgs

        self.z_near = z_near
        self.z_far = z_far
        self.lindisp = False
        self.encoder_pipeline = Compose(pipeline)

        self.use_occlusion = use_occlusion
        if self.use_occlusion:
            print("********** use_occlusion **********")
        self.use_semantic_labels = use_semantic_labels

        self.use_color_jittor = False
        self.add_flow_noise = add_flow_noise
        if self.add_flow_noise > 0:
            print("********* add flow noise: ", self.add_flow_noise, " **********************")
        
        self.use_flow_net_flow = use_flow_net_flow
        if self.use_flow_net_flow:
            print("********* use_flow_net ***********")

    def __len__(self):
        return len(self.file_lists)

    def __getitem__(self, index):
        datadir = self.file_lists[index]
        if "color_jittor" in datadir:
            self.use_color_jittor = True
            self.encoder_pipeline = Compose(COLOR_JITTOR_PIPELINE)
            img_pipeline = Compose(ONLY_COLOR_JITTOR_PIPELINE)
            datadir = datadir[:-len("_color_jittor")]
        images, encoder_imgs, invdepths, masks, poses, bds, \
        render_poses, render_focals, grids, semantic_labels = self.load_llff_data(
                                                                datadir,
                                                                self.factor,
                                                                frame2dolly=self.frame2dolly,
                                                                recenter=True, bd_factor=.9,
                                                                spherify=self.spherify)

        # hwf = poses[0, :3, -1]
        # poses = poses[:, :3, :4]
        # num_img = float(poses.shape[0])
        # assert len(poses) == len(images)
        # # print('Loaded llff', images.shape,
        # #     render_poses.shape, hwf, args.datadir)

        # # Use all views to train
        # i_train = np.array([i for i in np.arange(int(images.shape[0]))])

        # # print('DEFINING BOUNDS')
        # if self.no_ndc:
        #     raise NotImplementedError
        #     near = np.ndarray.min(bds) * .9
        #     far = np.ndarray.max(bds) * 1.
        # else:
        #     near = 0.
        #     far = 1.
        # # print('NEAR FAR', near, far)
        if self.use_color_jittor:
            encoder_data_images = dict(
                imgs = encoder_imgs,
                modality = 'RGB'
            )
            encoder_data_images = img_pipeline(encoder_data_images)
            images = encoder_data_images['imgs']
            images = [x.astype(np.float32) /255. for x in encoder_data_images['imgs']]
            # cv2.imwrite('color_jittor_test0.png', encoder_data_images['imgs'][0])
        encoder_data = dict(
            imgs = encoder_imgs,
            modality = 'RGB'
        )
        encoder_data = self.encoder_pipeline(encoder_data)
        result = dict(
            dataname = os.path.basename(datadir),
            images = images,
            encoder_imgs = encoder_data['imgs'],
            invdepths = invdepths,
            masks = masks,
            poses = poses,
            bds = bds,
            render_poses = render_poses,
            render_focals = render_focals,
            grids = grids,
            semantic_labels = semantic_labels,
            # hwf = hwf,
            # num_img = num_img,
            # i_train = i_train
        )
        if self.use_color_jittor:
            self.use_color_jittor = False
            self.encoder_pipeline = Compose(DEFAULT_PIPELINE)

        return result
    def generate_path(self, c2w):
        hwf = c2w[:, 4:5]
        num_novelviews = self.num_novelviews
        max_disp = 48.0
        H, W, focal = hwf[:, 0]

        max_trans = max_disp / focal
        output_poses = []
        output_focals = []

        # Rendering teaser. Add translation.
        for i in range(num_novelviews):
            x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * self.x_trans_multiplier
            y_trans = max_trans * (np.cos(2.0 * np.pi * float(i) / float(num_novelviews)) - 1.) * self.y_trans_multiplier
            z_trans = 0.

            i_pose = np.concatenate([
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
            ],axis=0)

            i_pose = np.linalg.inv(i_pose)

            ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

            render_pose = np.dot(ref_pose, i_pose)
            output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
            output_focals.append(focal)

        # Rendering teaser. Add zooming.
        if self.frame2dolly != -1:
            for i in range(num_novelviews // 2 + 1):
                x_trans = 0.
                y_trans = 0.
                # z_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_novelviews)) * self.z_trans_multiplier
                z_trans = max_trans * self.z_trans_multiplier * i / float(num_novelviews // 2)
                i_pose = np.concatenate([
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
                ],axis=0)

                i_pose = np.linalg.inv(i_pose) #torch.tensor(np.linalg.inv(i_pose)).float()

                ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

                render_pose = np.dot(ref_pose, i_pose)
                output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
                output_focals.append(focal)
                print(z_trans / max_trans / self.z_trans_multiplier)

        # Rendering teaser. Add dolly zoom.
        if self.frame2dolly != -1:
            for i in range(num_novelviews // 2 + 1):
                x_trans = 0.
                y_trans = 0.
                z_trans = max_trans * self.z_trans_multiplier * i / float(num_novelviews // 2)
                i_pose = np.concatenate([
                    np.concatenate(
                        [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]], axis=1),
                    np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]
                ],axis=0)

                i_pose = np.linalg.inv(i_pose)

                ref_pose = np.concatenate([c2w[:3, :4], np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :]], axis=0)

                render_pose = np.dot(ref_pose, i_pose)
                output_poses.append(np.concatenate([render_pose[:3, :], hwf], 1))
                new_focal = focal - self.focal_decrease * z_trans / max_trans / self.z_trans_multiplier
                output_focals.append(new_focal)
                print(z_trans / max_trans / self.z_trans_multiplier, new_focal)
        return output_poses, output_focals

    def load_llff_data(self, basedir,
                    factor=2,
                    recenter=True, bd_factor=.75,
                    spherify=False, path_zflat=False,
                    frame2dolly=10):

        poses, bds, imgs, encoder_imgs, disp, masks, flows_f, flow_masks_f, flows_b, flow_masks_b, semantic_labels = \
            _load_data(basedir, factor=factor, use_semantic_labels=self.use_semantic_labels, flow_net=self.use_flow_net_flow) # factor=2 downsamples original imgs by 2x

        if self.use_occlusion:
            imgs[40:100, 200:300, :, 6] = 0
            encoder_imgs[40:100, 200:300, :, 6] = 0
            flows_f[40:100, 200:300, :, 6] = 0
            flows_b[40:100, 200:300, :, 6] = 0
            semantic_labels['label_imgs'][40:100, 200:300, :, 6] = 0
            semantic_labels['label_logits'][40:100, 200:300, 6] = 0
        # print('Loaded', basedir, bds.min(), bds.max())
        # import ipdb; ipdb.set_trace()
        if self.add_flow_noise > 0:
            for i in range(flows_f.shape[-1]):
                scale_f = flows_f[..., i].max() - flows_f[..., i].min()
                scale_b = flows_b[..., i].max() - flows_b[..., i].min()

                flows_f_noise = (np.random.rand(*flows_f.shape[:-1]) - 0.5) * (scale_f * self.add_flow_noise)
                flows_b_noise = (np.random.rand(*flows_b.shape[:-1]) - 0.5) * (scale_b * self.add_flow_noise)

                flows_f[..., i] += flows_f_noise
                flows_b[..., i] += flows_b_noise
                


        # Correct rotation matrix ordering and move variable dim to axis 0
        poses = np.concatenate([poses[:, 1:2, :],
                            -poses[:, 0:1, :],
                                poses[:, 2:, :]], 1)
        # move the frame (last) axis to the first axis
        poses = np.moveaxis(poses, -1, 0).astype(np.float32)
        images = np.moveaxis(imgs, -1, 0).astype(np.float32)
        encoder_imgs = np.moveaxis(encoder_imgs, -1, 0).astype(np.float32)
        bds = np.moveaxis(bds, -1, 0).astype(np.float32) # 12, 2
        disp = np.moveaxis(disp, -1, 0).astype(np.float32)
        masks = np.moveaxis(masks, -1, 0).astype(np.float32)
        flows_f = np.moveaxis(flows_f, -1, 0).astype(np.float32)
        flow_masks_f = np.moveaxis(flow_masks_f, -1, 0).astype(np.float32)
        flows_b = np.moveaxis(flows_b, -1, 0).astype(np.float32)
        flow_masks_b = np.moveaxis(flow_masks_b, -1, 0).astype(np.float32)
        if self.use_semantic_labels:
            semantic_labels['label_imgs'] = np.moveaxis(semantic_labels['label_imgs'], -1, 0).astype(np.float32)
            semantic_labels['label_logits'] = np.moveaxis(semantic_labels['label_logits'], -1, 0).astype(np.float32)
            semantic_labels['box2semantic_labels'] = np.moveaxis(semantic_labels['box2semantic_labels'], -1, 0).astype(np.float32)


        # Rescale if bd_factor is provided
        sc = 1. if bd_factor is None else 1./(np.percentile(bds[:, 0], 5) * bd_factor)

        poses[:, :3, 3] *= sc
        bds *= sc

        if recenter:
            poses = recenter_poses(poses)

        # Only for rendering
        if frame2dolly == -1:
            c2w = poses_avg(poses)
        else:
            c2w = poses[frame2dolly, :, :]

        H, W, _ = c2w[:, -1]

        # Generate poses for novel views
        render_poses, render_focals = self.generate_path(c2w)
        render_poses = np.array(render_poses).astype(np.float32)

        grids = get_grid(int(H), int(W), len(poses), flows_f, flow_masks_f, flows_b, flow_masks_b) # [N, H, W, 8]

        return images, encoder_imgs, disp, masks, poses, bds,\
            render_poses, render_focals, grids, semantic_labels

if __name__ == '__main__':
    '''
    Start debugging from the root path!
    '''
    config_path = 'mmaction_configs/recognition/slowonly/slowonly_imagenet_pretrained_r50_8x8x1_150e_kinetics400_rgb.py'
    config_options = {}
    cfg = Config.fromfile(config_path)
    cfg.merge_from_dict(config_options)
    llff_dataset = LLFFDataset(
        path='data',
        # pipeline=cfg.data.train.pipeline,
        stage='train',
        factor=2,
    )
    train_data = torch.utils.data.DataLoader(
        llff_dataset, batch_size=2, shuffle=False, num_workers=8, pin_memory=False
    )
    for (idx, data) in enumerate(train_data):
        ipdb.set_trace()
