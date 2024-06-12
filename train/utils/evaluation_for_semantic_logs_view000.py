import os
import cv2
import lpips
import torch
import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import json
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

semantic_data_dir = f'/data/tianfr/NeRF_series/SemanticDyNeRF/data_semantic/'
label_map_path = os.path.join(semantic_data_dir, 'labelmap.txt')
with open(label_map_path, 'r') as f:
    label_map_content = f.readlines()[1:]

foreground_label_path = os.path.join(semantic_data_dir, "foreground_object.txt")
with open(foreground_label_path, 'r') as f:
    foreground_labels = f.read().strip().split('\n')
ONLY_FOREGROUND_LABELS = True
background_label_ids = []

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


def readimage(data_dir, sequence, time, method):
    img = cv2.imread(os.path.join(data_dir, method, sequence, 'v000_t' + str(time).zfill(3) + '.png'))
    return img

def readimage_semantic(data_dir, time):
    img = cv2.imread(os.path.join(data_dir, str(time+1).zfill(8), 'cam01.png'))[...,::-1]
    img = cv2.resize(img, (img.shape[1] //2, img.shape[0] //2), interpolation=cv2.INTER_NEAREST)
    semantic_true_logit = np.zeros(img.shape[:2])

    for label_name, label_value in label_maps.items():
        label_color = label_value['color']
        idx = label_value['idx']
        _label_pos = ((img[:, :, 0] == label_color[0]) & (img[:, :, 1] == label_color[1]) & (img[:, :, 2] == label_color[2]))
        semantic_true_logit[_label_pos] = idx
    return semantic_true_logit

def read_log_image(data_dir, seq_id, time, method):
    img = cv2.imread(os.path.join(data_dir, str(time).zfill(3) + '_%02d.png' % seq_id))
    return img

def read_log_semantic_label(data_dir, seq_id, time, method):
    img = cv2.imread(os.path.join(data_dir, str(time).zfill(3) + '_%02d.png' % seq_id))[...,::-1]
    semantic_true_logit = np.zeros(img.shape[:2])

    for label_name, label_value in label_maps.items():
        label_color = label_value['color']
        idx = label_value['idx']
        _label_pos = ((img[:, :, 0] == label_color[0]) & (img[:, :, 1] == label_color[1]) & (img[:, :, 2] == label_color[2]))
        semantic_true_logit[_label_pos] = idx
    return semantic_true_logit

def calculate_metrics(data_dir, sequence, methods, lpips_loss, seq_id=0):

    PSNRs = np.zeros((len(methods)))
    SSIMs = np.zeros((len(methods)))
    LPIPSs = np.zeros((len(methods)))
    mIOUs = np.zeros((len(methods)))
    ACCURACYs = np.zeros((len(methods)))


    nFrame = 0
    log_base_dir = f"/data/tianfr/NeRF_series/SemanticDyNeRF/logs/semantic_flow/full_radiance_field/mononerf_head/1gpusBalloon1"
    json_path = os.path.join(log_base_dir, 'semflow_config.json')
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    # w_semantic_label_frames = json_data['info'].get('w_semantic_label_frames', [])

    # Yoon's results do not include v000_t000 and v000_t011. Omit these two
    # frames if evaluating Yoon's method.
    if 'Yoon' in methods:
        time_start = 1
        time_end = 11
    else:
        time_start = 0
        time_end = 12

    for time in range(time_start, time_end): # Fix view v0, change time

        # if time in w_semantic_label_frames:
        #     continue

        nFrame += 1
        img_true = readimage(data_dir, sequence, time, 'gt')

        semantic_data_dir_curr = os.path.join(semantic_data_dir, f'{sequence}/multiview_GT_semantic')
        semantic_true = readimage_semantic(semantic_data_dir_curr, time)

        for method_idx, method in enumerate(methods):

            # if 'Yoon' in methods and sequence == 'Truck' and time == 10:
            #     break
            # import ipdb; ipdb.set_trace()
            log_dir = os.path.join(log_base_dir, f"testset_view000_{method}")
            log_dir_img = os.path.join(log_dir, 'imgs')
            log_dir_semantic = os.path.join(log_dir, 'semantic_labels')
            # import ipdb; ipdb.set_trace()
            if not os.path.exists(log_dir): 
                # print("time: ", time)
                print(sequence, "don't have", method)
                break
            img = read_log_image(log_dir_img, seq_id, time, method)
            # import ipdb; ipdb.set_trace()
            semantic = read_log_semantic_label(log_dir_semantic, seq_id, time, method)
            # cv2.imwrite("our.png",img)
            # cv2.imwrite("gt.png", img_true)
            ignore_labels = 0
            if ONLY_FOREGROUND_LABELS:
                ignore_labels = background_label_ids
            
            # import ipdb; ipdb.set_trace()
            # miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = calculate_segmentation_metrics(semantic_true, semantic, len(np.unique(semantic_true)), ignore_label=0)
            miou, miou_valid_class, total_accuracy, class_average_accuracy, ious = calculate_segmentation_metrics(semantic_true, semantic, 30, ignore_labels=ignore_labels)
            ACCURACYs[method_idx] += total_accuracy
            mIOUs[method_idx] += miou

            PSNR = cv2.PSNR(img_true, img)
            # SSIM = structural_similarity(img_true, img, multichannel=True)
            SSIM = structural_similarity(img_true, img, channel_axis=-1)
            LPIPS = lpips_loss.forward(im2tensor(img_true), im2tensor(img)).item()

            PSNRs[method_idx] += PSNR
            SSIMs[method_idx] += SSIM
            LPIPSs[method_idx] += LPIPS



    PSNRs = PSNRs / nFrame
    SSIMs = SSIMs / nFrame
    LPIPSs = LPIPSs / nFrame
    ACCURACYs = ACCURACYs / nFrame
    mIOUs = mIOUs / nFrame

    return PSNRs, SSIMs, LPIPSs, ACCURACYs, mIOUs

if __name__ == '__main__':

    lpips_loss = lpips.LPIPS(net='alex') # best forward scores
    data_dir = '../results'
    # sequences = ['Balloon1', 'Balloon2', 'Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']
    # sequences = ['Jumping', 'Playground', 'Skating', 'Truck', 'Umbrella']
    # sequences = ['Balloon2','Balloon1',]
    sequences = ['Balloon1']
    # methods = ['NeRF', 'NeRF_t', 'Yoon', 'NR', 'NSFF', 'Ours']
    # methods = ['NeRF', 'NeRF_t', 'NR', 'NSFF', 'Ours']
    methods = ["020000","040000", "060000", "080000", "100000","120000", "140000"]
    # methods = list(range(2000, 150001, 2000))
    # methods = ["%06d" % i for i in methods]

    PSNRs_total = np.zeros((len(methods)))
    SSIMs_total = np.zeros((len(methods)))
    LPIPSs_total = np.zeros((len(methods)))
    ACCURACYs_total = np.zeros((len(methods)))
    mIOUs_total = np.zeros(len(methods))
    for seq_id, sequence in enumerate(sequences):
        print(sequence)
        PSNRs, SSIMs, LPIPSs, ACCURACYs, mIOUs = calculate_metrics(data_dir, sequence, methods, lpips_loss, seq_id=seq_id)
        for method_idx, method in enumerate(methods):
            print(method.ljust(12) + '%.2f'%(PSNRs[method_idx]) + ' / %.4f'%(SSIMs[method_idx]) + ' / %.3f'%(LPIPSs[method_idx]) + ' / %.3f'%(ACCURACYs[method_idx]) + ' / %.3f'%(mIOUs[method_idx]))

        PSNRs_total += PSNRs
        SSIMs_total += SSIMs
        LPIPSs_total += LPIPSs
        ACCURACYs_total += ACCURACYs
        mIOUs_total += mIOUs

    PSNRs_total = PSNRs_total / len(sequences)
    SSIMs_total = SSIMs_total / len(sequences)
    LPIPSs_total = LPIPSs_total / len(sequences)
    ACCURACYs_total = ACCURACYs_total / len(sequences)
    mIOUs_total = mIOUs_total / len(sequences)
    print('Avg.')
    for method_idx, method in enumerate(methods):
        print(method.ljust(7) + '%.2f'%(PSNRs_total[method_idx]) + ' / %.4f'%(SSIMs_total[method_idx]) + ' / %.3f'%(LPIPSs_total[method_idx]) + ' / %.3f'%(ACCURACYs_total[method_idx]) + ' / %.3f'%(mIOUs_total[method_idx]))
