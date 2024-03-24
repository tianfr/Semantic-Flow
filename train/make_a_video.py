import cv2
import numpy as np
import os
import os.path as osp
import ipdb, glob

def readimage_gt(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%02d" % pose
    data_dir = f"./data/{sequences[idx]}/multiview_GT/{time}/cam{pose}.jpg"
    img = cv2.imread(data_dir)
    img = cv2.resize(img,(img.shape[1]//2, img.shape[0]//2))
    return img



def readimage(data_dir):
    time = "%08d" % time 
    pose = "%03d" % (pose-1)
    idx = "%02d" % idx
    data_dir = f"{data_dir}/{time}/imgs/{pose}_{idx}.png"
    img = cv2.imread(data_dir)
    if img is None:
        import ipdb; ipdb.set_trace()
    return img





scene = "Balloon2"
logs_dir = "/mnt/cache/tianfengrui/NeRF_series/SeDyODENeRF/logs/paper/single_scene/1gpusdiscrete2_Balloon2_use_globallayer5_addition_temporalspatial1-1_maskflow0.01wodecay_dbl0.03/multiview/step_060000"
gt_dir = "/mnt/cache/tianfengrui/NeRF_series/SeDyODENeRF/data/Balloon2/multiview_GT"

def main():
    ipdb.set_trace()
    imgs = []
    for i in range(1, 13):
        curr_dir = osp.join(logs_dir, "%08d"% i, "imgs","*.png")
        imgs_dir = glob.glob(curr_dir).sort()




if __name__ == "__main__":
    main()