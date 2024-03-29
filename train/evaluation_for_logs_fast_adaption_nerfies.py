import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity
import glob

SCENE = "tail-12"
# STEP = 50
sequences = [SCENE]

def im2tensor(img):
    return torch.Tensor(img.transpose(2, 0, 1) / 127.5 - 1.0)[None, ...]


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def readimage_gt(data_dir, idx, time, pose):
    time = "%03d" % time 
    # pose = "%02d" % pose
    data_dir = f"./data/{sequences[idx]}/images/{time}.png"
    img = cv2.imread(data_dir)
    img = cv2.resize(img,(img.shape[1]//2, img.shape[0]//2))
    return img

def readimage(data_dir, idx, time, pose):
    time = "%03d" % time 
    # pose = "%03d" % (pose-1)
    idx = "%02d" % idx
    # data_dir = f"{data_dir}/imgs/{time}_{idx}.png"
    data_dir = f"{data_dir}/{time}.png"
    # data_dir = f"{data_dir}/{time}/{pose}.png"
    img = cv2.imread(data_dir)
    if img is None:
        import ipdb; ipdb.set_trace()
    return img

def calculate_metrics(data_dir, sequence_idx, methods, lpips_loss):

    PSNRs = np.zeros((len(methods)))
    SSIMs = np.zeros((len(methods)))
    LPIPSs = np.zeros((len(methods)))

    nFrame = 0

    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    # if 'Yoon' in methods:
    #     time_start = 1
    #     time_end = 11
    # else:
    time_start, time_end = 0, 12
    pose_start, pose_end = 1, 2
    for time in range(time_start, time_end): # Fix view v0, change time
        for pose in range(pose_start, pose_end):
            nFrame += 1

            img_true = readimage_gt(data_dir, sequence_idx, time, pose)
            for method_idx, method in enumerate(methods):

                # if 'Yoon' in methods and sequence == 'Truck' and time == 10:
                #     break
                img = readimage(method, sequence_idx, time, pose)
                PSNR = cv2.PSNR(img_true, img)
                SSIM = structural_similarity(img_true, img, multichannel=True)
                LPIPS = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()
                if time == 4:
                    print(LPIPS)
                PSNRs[method_idx] += PSNR
                SSIMs[method_idx] += SSIM
                LPIPSs[method_idx] += LPIPS

    PSNRs = PSNRs / nFrame
    SSIMs = SSIMs / nFrame
    LPIPSs = LPIPSs / nFrame

    return PSNRs, SSIMs, LPIPSs

lpips_loss = lpips.LPIPS(net='alex') # best forward scores
def main(iter):


    # data_dir = '../results'
    # data_dir = f'/mnt/cache/tianfengrui/NeRF_series/SeDyODENeRF/logs/paper/generalization_wo_trajfeat/{iter}steps_1gpus{SCENE}'
    data_dir = f'/data/tianfr/NeRF_series/DynamicNeRF/logs_new/transfer_from_Balloon2/500steps_tfr_tail-12_from_Balloon2_H270_DyNeRF'
    multiview_dir = os.path.join(data_dir, "testset_0*")
    steps = glob.glob(multiview_dir)
    steps.sort()
    # steps = steps[:-1]

    # sequences = ['Balloon1', 'Balloon2', 'Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']

    # sequences = ['Balloon2']
    # methods = ['NeRF', 'NeRF_t', 'Yoon', 'NR', 'NSFF', 'Ours']
    # methods = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours']
    # steps = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours', "DNeRF", 'Ours.depthblending0.1']

    PSNRs_total = np.zeros((len(steps)))
    SSIMs_total = np.zeros((len(steps)))
    LPIPSs_total = np.zeros((len(steps)))
    for idx in range(len(sequences)):
        # print(sequences[idx])
        PSNRs, SSIMs, LPIPSs = calculate_metrics(data_dir, idx, steps, lpips_loss)
        for method_idx, method in enumerate(steps):
            print(method.split("/")[-1].ljust(15) + '%.2f'%(PSNRs[method_idx]) + ' / %.4f'%(SSIMs[method_idx]) + ' / %.3f'%(LPIPSs[method_idx]))
        PSNRs_total += PSNRs
        SSIMs_total += SSIMs
        LPIPSs_total += LPIPSs

if __name__ == "__main__":
    # iters = [50]+list(range(100,3200,100))
    iters = [0]
    print("*"*15)
    print(SCENE)
    for iter in iters:
        main(iter)
    # PSNRs_total = PSNRs_total / len(sequences)
    # SSIMs_total = SSIMs_total / len(sequences)
    # LPIPSs_total = LPIPSs_total / len(sequences)
    # print('Avg.')
    # for method_idx, method in enumerate(steps):
    #     print(method.split("/")[-1].ljust(15) + '%.2f'%(PSNRs_total[method_idx]) + ' / %.4f'%(SSIMs_total[method_idx]) + ' / %.3f'%(LPIPSs_total[method_idx]))
