# -*- coding: utf-8 -*-
import logging
import numpy as np
import time
import math

import torch
import torch.nn.functional as F

def random_extract(feat, t_max):
    r = np.random.randint(len(feat) - t_max)
    return feat[r:r + t_max]

def uniform_extract(feat, t_max):
    if len(feat.shape) == 1:
    	print("1 din")
    r = np.linspace(0, len(feat) - 1, t_max, dtype=np.uint16)
    return feat[r, :]


def pad(feat, min_len):
    if np.shape(feat)[0] <= min_len:
        return np.pad(feat, ((0, min_len - np.shape(feat)[0]), (0, 0)), mode='constant', constant_values=0)
    else:
        return feat


def process_feat(feat, length, is_random=True):
    if len(feat) > length:
        if is_random:
            return random_extract(feat, length)
        else:
            return uniform_extract(feat, length)
    else:
        return pad(feat, length)


def process_test_feat(feat, length):
    tem_len = len(feat)
    num = math.ceil(tem_len / length)
    if len(feat) < length:
        return pad(feat, length)
    else:
        return pad(feat, num * length)
        
        
def pad_text_to_same_length(f_a, f_v, f_f, f_t):

    max_len = max(f_a.size(1), f_v.size(1), f_f.size(1))

    text_len = f_t.size(1)
    
    if text_len < max_len:

        padding = (0, 0, 0, max_len - text_len)
        f_t = F.pad(f_t, padding)
    if text_len > max_len:
        f_t = f_t[:, :max_len, :]
    return f_a, f_v, f_f, f_t
    

def Prepare_logger():
    logger = logging.getLogger(__name__)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(0)
    logger.addHandler(handler)
    date = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    logfile = 'log/' + date + '.log'
    file_handler = logging.FileHandler(logfile, mode='w')
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num, num))
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label:
                gt[i, k] = 1
    return gt


def get_neutral_mask(video_ids, label):

    label = label.unsqueeze(0)
    video_ids = video_ids.unsqueeze(0)
    
    label_match = label != label.T
    id_match = video_ids == video_ids.T

    neutral_mask = label_match | id_match 
    neutral_mask = neutral_mask.float()

    return neutral_mask
    
