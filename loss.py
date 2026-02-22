import torch
import torch.nn as nn
import torch.nn.init as torch_init
import torch.nn.functional as F
from Transformer import *
import numpy as np
from utils import gen_label, get_neutral_mask





def MIL(logits, seq_len):
    logits = logits.squeeze()
    instance_logits = torch.zeros(0).cuda()
    for i in range(logits.shape[0]):
        if seq_len is None:
            return logits
        else:

            tmp, _ = torch.topk(logits[i][:seq_len[i]], k=int(seq_len[i] // 16 + 1), largest=True)
            tmp = torch.mean(tmp).view(1)
        instance_logits = torch.cat((instance_logits, tmp))
    return instance_logits


def VCA(video_feat, text_feat, video_ids, video_cls, logit_scale):
    video_feat = torch.mean(video_feat, dim=1)
    video_feat = video_feat.squeeze()
    text_feat = text_feat.squeeze()

    video_feat = F.normalize(video_feat, dim=-1)
    text_feat = F.normalize(text_feat, dim=-1)
    sim_v2t = torch.matmul(logit_scale * video_feat, text_feat.t())
    sim_t2v = torch.matmul(logit_scale * text_feat, video_feat.t())
    
    neutral_mask = get_neutral_mask(video_ids, video_cls)
    _, targets = torch.unique(video_ids, return_inverse=True)
    ground_truth = torch.tensor(gen_label(targets), dtype=video_feat.dtype).to(video_feat.device)
    ground_truth = F.softmax(ground_truth * 10, dim = 1)
	
    loss_img_matrix = F.kl_div(F.log_softmax(sim_v2t, dim=1), ground_truth, reduction='none')
    loss_txt_matrix = F.kl_div(F.log_softmax(sim_t2v, dim=1), ground_truth, reduction='none')

    masked_loss_img = loss_img_matrix * neutral_mask
    loss_img = masked_loss_img.sum() / masked_loss_img.size()[0]
    masked_loss_txt = loss_txt_matrix * neutral_mask
    loss_txt = masked_loss_txt.sum() / masked_loss_txt.size()[0]


    loss_vca = (loss_img + loss_txt) / 2

    return loss_vca
    


