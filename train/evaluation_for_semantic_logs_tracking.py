import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity
import glob
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

sequences = ["Truck"]
# sequences = [ "Jumping","Skating",]
# sequences = ["Playground"]

semantic_data_dir = f'data_semantic/'
label_map_path = os.path.join(semantic_data_dir, 'labelmap.txt')
with open(label_map_path, 'r') as f:
    label_map_content = f.readlines()[1:]

foreground_label_path = os.path.join(semantic_data_dir, "foreground_object.txt")
with open(foreground_label_path, 'r') as f:
    foreground_labels = f.read().strip().split('\n')
ONLY_FOREGROUND_LABELS = False
background_label_ids = []

ONLY_UNSEEN_FRAMES = True

label_maps = {}
label_maps_array = []
for idx, label in enumerate(label_map_content):
    label_name, label_color = label.strip().strip(":").split(":")
    if label_name not in foreground_labels:
        background_label_ids.append(idx)
    label_color = np.array([int(x) for x in label_color.split(',')])
    label_maps[label_name] = dict(
        color = label_color,
        idx = idx,
    )
#NOTE Background index corretion
# label_maps['background']['idx'] = label_maps['grey wall']['idx']

def nanmean(data, **args):
    # This makes it ignore the first 'background' class
    return np.ma.masked_array(data, np.isnan(data)).mean(**args)
    # In np.ma.masked_array(data, np.isnan(data), elements of data == np.nan is invalid and will be ingorned during computation of np.mean()

def calculate_segmentation_metrics(true_labels, predicted_labels, number_classes, ignore_labels):
    # if (true_labels == ignore_labels).all():
    #     return [0]*4

    true_labels = true_labels.flatten()
    predicted_labels = predicted_labels.flatten()
    if type(ignore_labels) != list:
        ignore_labels = [ignore_labels]
    for ignore_label in ignore_labels:
        valid_pix_ids = predicted_labels==ignore_label
        predicted_labels[valid_pix_ids] = 0

        valid_pix_ids = true_labels==ignore_label
        true_labels[valid_pix_ids] = 0
    
    # conf_mat = confusion_matrix(true_labels, predicted_labels, labels=list(range(number_classes)))
    classes = unique_labels(true_labels, predicted_labels)
    conf_mat = confusion_matrix(true_labels, predicted_labels, labels=classes)
    norm_conf_mat = np.transpose(
        np.transpose(conf_mat) / conf_mat.astype(np.float64).sum(axis=1))

    missing_class_mask = np.isnan(norm_conf_mat.sum(1)) # missing class will have NaN at corresponding class
    exsiting_class_mask = ~ missing_class_mask

    class_average_accuracy = nanmean(np.diagonal(norm_conf_mat))
    total_accuracy = (np.sum(np.diagonal(conf_mat)) / np.sum(conf_mat))
    ious = np.zeros(len(classes))
    for class_id in range(len(classes)):
        ious[class_id] = (conf_mat[class_id, class_id] / (
                np.sum(conf_mat[class_id, :]) + np.sum(conf_mat[:, class_id]) -
                conf_mat[class_id, class_id]))
    miou = nanmean(ious)
    miou_valid_class = np.mean(ious[exsiting_class_mask])
    
    return miou, miou_valid_class, total_accuracy, class_average_accuracy, ious


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
    img = cv2.resize(img,(img.shape[1]//2, img.shape[0]//2))
    return img

def readimage(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%03d" % (pose-1)
    idx = "%02d" % idx
    data_dir = f"{data_dir}/{time}/imgs/{pose}_{idx}.png"
    img = cv2.imread(data_dir)
    if img is None:
        import ipdb; ipdb.set_trace()
    return img

def readsemantic_gt(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%02d" % pose
    data_dir = f"./data_semantic/{sequences[idx]}/multiview_GT_semantic/{time}/cam{pose}.png"
    # print(data_dir)
    img = cv2.imread(data_dir)[...,::-1]
    img = cv2.resize(img, (img.shape[1] //2, img.shape[0] //2), interpolation=cv2.INTER_NEAREST)
    semantic_true_logit = np.zeros(img.shape[:2])

    for label_name, label_value in label_maps.items():
        label_color = label_value['color']
        idx = label_value['idx']
        _label_pos = ((img[:, :, 0] == label_color[0]) & (img[:, :, 1] == label_color[1]) & (img[:, :, 2] == label_color[2]))
        semantic_true_logit[_label_pos] = idx
    return semantic_true_logit

def readsemantic(data_dir, idx, time, pose):
    time = "%08d" % time 
    pose = "%03d" % (pose-1)
    idx = "%02d" % idx
    data_dir = f"{data_dir}/{time}/SegmentationClass/{pose}.png"
    img = cv2.imread(data_dir)[...,::-1]
    # import ipdb; ipdb.set_trace()
    img = cv2.resize(img, (480, 270))
    semantic_true_logit = np.zeros(img.shape[:2])

    for label_name, label_value in label_maps.items():
        label_color = label_value['color']
        idx = label_value['idx']
        _label_pos = ((img[:, :, 0] == label_color[0]) & (img[:, :, 1] == label_color[1]) & (img[:, :, 2] == label_color[2]))
        semantic_true_logit[_label_pos] = idx
    return semantic_true_logit

def calculate_metrics(data_dir, sequence_idx, methods, lpips_loss):

    PSNRs = np.zeros((len(methods)))
    SSIMs = np.zeros((len(methods)))
    LPIPSs = np.zeros((len(methods)))
    mIOUs = np.zeros((len(methods)))
    ACCURACYs = np.zeros((len(methods)))
    CLASSACCURACYs = np.zeros((len(methods)))

    nFrame = 0
    # json_path = os.path.join(data_dir, 'pixelnerf_config.json')
    # with open(json_path, 'r') as f:
    #     json_data = json.load(f)

    # w_semantic_label_frames = json_data['info'].get('w_semantic_label_frames', [])
    w_semantic_label_frames = [1,2,3,4]
    print(w_semantic_label_frames)


    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    # if 'Yoon' in methods:
    #     time_start = 1
    #     time_end = 11
    # else:
    time_start, time_end = 1, 13
    pose_start, pose_end = 1, 13
    for time in range(time_start, time_end):

        if time in w_semantic_label_frames and ONLY_UNSEEN_FRAMES:
            continue
        for pose in range(pose_start, pose_end):
            nFrame += 1

            img_true = readimage_gt(data_dir, sequence_idx, time, pose)
            semantic_true = readsemantic_gt(data_dir, sequence_idx, time, pose)

            for method_idx, method in enumerate(methods):

                # if 'Yoon' in methods and sequence == 'Truck' and time == 10:
                #     break
                # img = readimage(method, sequence_idx, time, pose)
                semantic = readsemantic(method, sequence_idx, time, pose)
                # cv2.imwrite("our.png",img)
                # cv2.imwrite("gt.png", img_true)
                ignore_labels = 0
                if ONLY_FOREGROUND_LABELS:
                    ignore_labels = background_label_ids
                
                # import ipdb; ipdb.set_trace()
                # miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = calculate_segmentation_metrics(semantic_true, semantic, len(np.unique(semantic_true)), ignore_label=0)
                miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = calculate_segmentation_metrics(semantic_true, semantic, 30, ignore_labels=ignore_labels)
                
                ACCURACYs[method_idx] += total_accuracy
                CLASSACCURACYs[method_idx] += class_average_accuracy
                mIOUs[method_idx] += miou


                # PSNR = cv2.PSNR(img_true, img)
                # SSIM = structural_similarity(img_true, img, channel_axis=-1)
                # LPIPS = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()

                # PSNRs[method_idx] += PSNR
                # SSIMs[method_idx] += SSIM
                # LPIPSs[method_idx] += LPIPS

    PSNRs = PSNRs / nFrame
    SSIMs = SSIMs / nFrame
    LPIPSs = LPIPSs / nFrame
    ACCURACYs = ACCURACYs / nFrame
    CLASSACCURACYs = CLASSACCURACYs / nFrame
    mIOUs = mIOUs / nFrame

    return PSNRs, SSIMs, LPIPSs, ACCURACYs, CLASSACCURACYs, mIOUs


if __name__ == '__main__':

    lpips_loss = lpips.LPIPS(net='alex') # best forward scores
    # data_dir = '../results'
    data_dir = f'/data/tianfr/NeRF_series/Segment-and-Track-Anything/data_semantic_prediction_tracking_25/'+sequences[0]
    print(data_dir)
    print("ONLY_DYNAMIC: ", ONLY_FOREGROUND_LABELS, "ONLY_UNSEEN: ", ONLY_UNSEEN_FRAMES )
    multiview_dir = os.path.join(data_dir, "multiview_GT")
    steps = glob.glob(multiview_dir)
    steps.sort()
    # steps = steps[:-1]
    # steps = steps[-1:]
    print(steps)

    # sequences = ['Balloon1', 'Balloon2', 'Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']

    # sequences = ['Balloon2']
    # methods = ['NeRF', 'NeRF_t', 'Yoon', 'NR', 'NSFF', 'Ours']
    # methods = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours']
    # steps = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours', "DNeRF", 'Ours.depthblending0.1']

    PSNRs_total = np.zeros((len(steps)))
    SSIMs_total = np.zeros((len(steps)))
    LPIPSs_total = np.zeros((len(steps)))
    ACCURACYs_total = np.zeros((len(steps)))
    CLASSACCURACYs_total = np.zeros((len(steps)))
    mIOUs_total = np.zeros(len(steps))
    for idx in range(len(sequences)):
        print(sequences[idx])
        PSNRs, SSIMs, LPIPSs, ACCURACYs, CLASSACCURACYs, mIOUs = calculate_metrics(data_dir, idx, steps, lpips_loss)
        for method_idx, method in enumerate(steps):
            print(method.split("/")[-1].ljust(12) + '%.2f'%(PSNRs[method_idx]) + ' / %.4f'%(SSIMs[method_idx]) + ' / %.3f'%(LPIPSs[method_idx]) + ' / %.3f'%(ACCURACYs[method_idx]) + ' / %.3f'%(CLASSACCURACYs[method_idx]) + ' / %.3f'%(mIOUs[method_idx]))

        PSNRs_total += PSNRs
        SSIMs_total += SSIMs
        LPIPSs_total += LPIPSs
        ACCURACYs_total += ACCURACYs
        CLASSACCURACYs_total += CLASSACCURACYs
        mIOUs_total += mIOUs

    PSNRs_total = PSNRs_total / len(sequences)
    SSIMs_total = SSIMs_total / len(sequences)
    LPIPSs_total = LPIPSs_total / len(sequences)
    ACCURACYs_total = ACCURACYs_total / len(sequences)
    CLASSACCURACYs_total = CLASSACCURACYs_total / len(sequences)
    mIOUs_total = mIOUs_total / len(sequences)
    print('Avg.')
    for method_idx, method in enumerate(steps):
        print(method.split("/")[-1].ljust(12) + '%.2f'%(PSNRs_total[method_idx]) + ' / %.4f'%(SSIMs_total[method_idx]) + ' / %.3f'%(LPIPSs_total[method_idx]) + ' / %.3f'%(ACCURACYs_total[method_idx])  + ' / %.3f'%(CLASSACCURACYs_total[method_idx]) + ' / %.3f'%(mIOUs_total[method_idx]))
