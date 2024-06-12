import os
import cv2
import torch
import numpy as np
import glob
import imageio

sequences = ["Balloon2",]

def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def readimage_gt(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%02d" % pose
    data_dir = f"./data/{sequences[idx]}/multiview_GT/{time}/cam{pose}.jpg"
    img = cv2.imread(data_dir)
    img = cv2.resize(img,(img.shape[1]//2, img.shape[0]//2))[...,::-1]
    return img

def readimage(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%03d" % (pose-1)
    idx = "%02d" % idx
    # data_dir = f"{data_dir}/{time}/imgs/{pose}_{idx}.png"
    data_dir = f"{data_dir}/{time}/{pose}.png"
    img = cv2.imread(data_dir)
    if img is None:
        import ipdb; ipdb.set_trace()
    return img[...,::-1]

def calculate_metrics(data_dir, sequence_idx, methods, input_pose, dir_name):


    nFrame = 0

    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    # if 'Yoon' in methods:
    #     time_start = 1
    #     time_end = 11
    # else:
    time_start, time_end = 1, 13
    pose_start, pose_end = 1, 13

    times = np.repeat(np.arange(1,11), 1)
    times = np.concatenate([np.arange(1,11), np.arange(1,11)[::-1], np.arange(1,11), np.arange(1,11)[::-1], np.arange(1,11), np.arange(1,11)[::-1]])
    poses = np.concatenate([np.arange(input_pose, input_pose+1)])
    if len(times) > len(poses):
        poses = np.tile(poses, (int(np.ceil(len(times) / len(poses)))))
        poses = poses[:len(times)]
    # import ipdb; ipdb.set_trace()
    imgs = []
    gts = []
    # for time in range(time_start, time_end): # Fix view v0, change time
    #     for pose in range(pose_start, pose_end):
    for time, pose in zip(times, poses):
        nFrame += 1

        img_true = readimage_gt(data_dir, sequence_idx, time, pose)
        for method_idx, method in enumerate(methods):

            # if 'Yoon' in methods and sequence == 'Truck' and time == 10:
            #     break
            img = readimage(method, sequence_idx, time, pose)
            imgs.append(img)
            gts.append(img_true)

    gts = np.stack(gts, axis=0)
    imgs = np.stack(imgs, axis=0)
    imageio.mimwrite(os.path.join(dir_name, "%03d"% input_pose+".mp4"), 
        imgs, fps=8, quality=8, macro_block_size=1)
    # imageio.mimwrite(os.path.join(dir_name, "%03d"% input_pose+"_gt.mp4"), gts, fps=8, quality=8, macro_block_size=1)
    print("finish %03d" % input_pose)



if __name__ == '__main__':

    data_dir = '../results'
    multiview_dir = os.path.join(data_dir, "multiview","step*")
    steps = glob.glob(multiview_dir)
    steps.sort()
    steps = steps[3:4]
    dir_name = os.path.basename(data_dir) + "_" + os.path.basename(steps[0])
    dir_name = os.path.join(steps[0], "videos")
    os.makedirs(dir_name,exist_ok=True)

    poses = np.arange(1, 12)

    # sequences = ['Balloon1', 'Balloon2', 'Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']

    # sequences = ['Balloon2']
    # methods = ['NeRF', 'NeRF_t', 'Yoon', 'NR', 'NSFF', 'Ours']
    # methods = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours']
    # steps = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours', "DNeRF", 'Ours.depthblending0.1']

    PSNRs_total = np.zeros((len(steps)))
    SSIMs_total = np.zeros((len(steps)))
    LPIPSs_total = np.zeros((len(steps)))
    for idx in range(len(sequences)):
        print(sequences[idx])
        for pose in poses:
            calculate_metrics(data_dir, idx, steps, pose, dir_name)