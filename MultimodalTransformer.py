import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
import numpy as np
from Transformer import *
from MultiScaleBottleneckTransformer import MultiScale_Bottleneck_Transformer


def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def infoNCE(f_a, f_b, seq_len, temperature=0.5):
    n_batch = f_a.shape[0]
    total_loss = .0
    for i in range(n_batch):

        exp_mat = torch.exp(torch.mm(f_a[i][:seq_len[i]], f_b[i][:seq_len[i]].t()) / temperature)
        positive_mat = torch.diag(exp_mat)
        exp_mat_transpose = exp_mat.t()
        loss_i = torch.mean(-torch.log(positive_mat / torch.sum(exp_mat, dim=-1)))

        loss_i += torch.mean(-torch.log(positive_mat / torch.sum(exp_mat_transpose, dim=-1)))
        total_loss += loss_i
    return total_loss / n_batch


class MultimodalTransformer(nn.Module):
    def __init__(self, args):
        super(MultimodalTransformer, self).__init__()
        dropout = args.dropout
        nhead = args.nhead
        hid_dim = args.hid_dim
        ffn_dim = args.ffn_dim
        n_transformer_layer = args.n_transformer_layer
        n_bottleneck = args.n_bottleneck
        self.fc_v = nn.Linear(args.v_feature_size, hid_dim)
        self.fc_a = nn.Linear(args.a_feature_size, hid_dim)
        self.fc_f = nn.Linear(args.f_feature_size, hid_dim)
        self.msa = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.bottle_msa = SelfAttentionBlock(TransformerLayer(hid_dim, MultiHeadAttention(nhead, hid_dim), PositionwiseFeedForward(hid_dim, ffn_dim), dropout))
        self.MST = MultiScale_Bottleneck_Transformer(hid_dim, n_head=nhead, dropout=dropout, n_bottleneck=n_bottleneck, bottleneck_std=args.bottleneck_std)
        d_mmt = hid_dim * 6
        h_mmt = 6
        self.mm_transformer = MultilayerTransformer(TransformerLayer(d_mmt, MultiHeadAttention(h=h_mmt, d_model=d_mmt), PositionwiseFeedForward(d_mmt, d_mmt), dropout), n_transformer_layer)
        self.regressor_bw = nn.Sequential(nn.Linear(hid_dim, 64), nn.ReLU(), nn.Dropout(0.3),
                                       nn.Linear(64, 32), nn.Dropout(0.3),
                                       nn.Linear(32, 1), nn.Sigmoid())
        self.fav_out = nn.Linear(hid_dim*6, 512)

        self.regressor_mil = nn.Sequential(nn.ReLU(), nn.Dropout(0.6),
        				nn.Linear(512, 32), nn.Dropout(0.6),
                                       nn.Linear(32, 1), nn.Sigmoid())
        temp = 0.05
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / temp))



    def forward(self, f_a, f_v, f_f, seq_len):  # audio, RGB, flow
        f_a, f_v, f_f = self.fc_a(f_a), self.fc_v(f_v), self.fc_f(f_f)
        f_a, f_v, f_f = self.msa(f_a), self.msa(f_v), self.msa(f_f)

        f_av, b_av = self.MST(f_a, f_v)
        f_va, b_va = self.MST(f_v, f_a)
        f_af, b_af = self.MST(f_a, f_f)
        f_fa, b_fa = self.MST(f_f, f_a)
        f_vf, b_vf = self.MST(f_v, f_f)
        f_fv, b_fv = self.MST(f_f, f_v)
        bottle_cat = torch.cat([b_av, b_va, b_af, b_fa, b_vf, b_fv], dim=1)
        bottle_cat = self.bottle_msa(bottle_cat)
        bottle_weight = self.regressor_bw(bottle_cat)

        loss_infoNCE = .0
        if seq_len != None:
            cnt_n = 0

            n_av, n_va, n_af, n_fa, n_vf, n_fv = normalize(f_av, f_va, f_af, f_fa, f_vf, f_fv)
            n_list = [n_av, n_va, n_af, n_fa, n_vf, n_fv]

            for i in range(len(n_list)):
                for j in range(i + 1, len(n_list)):
                    cnt_n += 1
                    loss_infoNCE += infoNCE(n_list[i], n_list[j], seq_len)
            loss_infoNCE = loss_infoNCE / cnt_n

        f_av, f_va, f_af, f_fa, f_vf, f_fv = [bottle_weight[:, i, :].view([-1, 1, 1]) * f
                                              for i, f in enumerate([f_av, f_va, f_af, f_fa, f_vf, f_fv])]
                                                                             
        f_avf = torch.cat([f_av, f_va, f_af, f_fa, f_vf, f_fv], dim=-1)

        f_avf = self.mm_transformer(f_avf)
        f_avf = self.fav_out(f_avf)
        logits = self.regressor_mil(f_avf)
        return f_avf, logits , loss_infoNCE


