""" 
rpn.py
Created by zenn at 2021/5/8 20:55
"""
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from pointnet2.utils import pytorch_utils as pt_utils

from pointnet2.utils.pointnet2_modules import PointnetSAModule
import math


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """
    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))

    return res.reshape(*raw_size, -1) #(2,5,2,3)


class Transformer_Block(nn.Module):
    def __init__(self, points, d_points, d_model, k, state=None, center_rate=0.0) -> None:
        super().__init__()
        self.points = points
        self.k = k
        self.state = state
        self.center_rate = center_rate

        self.fc_pre = nn.Linear(d_points, d_model)
        self.fc_post = nn.Linear(d_model, d_points)

        self.fc_pos_encode = nn.Sequential(
            nn.Linear(3, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.fc_attention = nn.Sequential(
            nn.Linear(d_model, d_model, bias=False),
            nn.ReLU(),
            nn.Linear(d_model, d_model, bias=False)
        )
        self.WQ = nn.Linear(d_model, d_model, bias=False)
        self.WK = nn.Linear(d_model, d_model, bias=False)
        self.WV = nn.Linear(d_model, d_model, bias=False)

        self.proj = nn.Linear(d_model, d_model, bias=False)

        self.k = k
        self.norm_dm = nn.LayerNorm(d_model)
        self.norm_dp = nn.LayerNorm(d_points)

        # Init weights
        for m in self.WK.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WQ.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

        for m in self.WV.modules():
            m.weight.data.normal_(0, math.sqrt(2. / m.out_features))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, xyz, features):   #(2,128,3), (2,128,512)  batchsize=2, 128points, feature_dim=512
        if self.state == 'l':
            dists = torch.sum((xyz[:, :, None] - xyz[:, None]) ** 2, dim=-1)
            knn_idx = dists.argsort()[:, :, :self.k]
            idx = knn_idx
        elif self.state == 'g':
            dists = torch.sum((xyz[:, :, None] - xyz[:, None]) ** 2, dim=-1)
            knn_idx = dists.argsort()[:, :, :int(self.k * self.center_rate)]
            ss_idx = dists.argsort()[:, :, ::int(self.points / self.k)]
            idx = torch.cat((knn_idx, ss_idx), dim=2)
        knn_xyz = index_points(xyz, idx)

        raw_f = features
        input = self.fc_pre(features)
        B, N, C = input.shape

        query, key, value = self.norm_dm(self.WQ(input)), index_points(self.norm_dm(self.WK(input)), idx), \
                            index_points(self.norm_dm(self.WV(input)), idx)
        query = query.view(B, N, 1, -1).permute(0, 2, 1, 3).flatten(0, 1)
        pos_embedding = self.fc_pos_encode(xyz[:, :, None] - knn_xyz)
        attn = self.fc_attention(query[:, :, None] - key + pos_embedding)
        
        attn = F.softmax(attn / np.sqrt(key.size(-1)), dim=-2)

        res = torch.einsum('bmnf,bmnf->bmf', attn, value + pos_embedding)
        res = self.norm_dm(self.proj(res))
        res = self.norm_dp(self.fc_post(res)) + raw_f

        return res

class P2BVoteNetRPN(nn.Module):
    def __init__(self, feature_channel, vote_channel=256, num_proposal=64, normalize_xyz=False, transformer=True,
                 center_rate=0.0, de_head=True):
        super().__init__()
        self.num_proposal = num_proposal
        self.transformer = transformer
        self.center_rate = center_rate
        self.de_head = de_head

        if self.transformer:
            self.Global_transformer = Transformer_Block(points=128, d_points=256, d_model=256, k=16, state='g',
                                                        center_rate=self.center_rate)
            self.Local_transformer = Transformer_Block(points=128, d_points=256, d_model=256, k=16, state='l')

        if self.de_head:
            self.FC_proposal_offset = (
                pt_utils.Seq(vote_channel)
                    .conv1d(vote_channel, bn=True)
                    .conv1d(3, activation=None))

            self.FC_proposal_angle = (
                pt_utils.Seq(vote_channel)
                    .conv1d(vote_channel, bn=True)
                    .conv1d(1, activation=None))

            self.FC_proposal_score = (
                pt_utils.Seq(vote_channel)
                    .conv1d(vote_channel, bn=True)
                    .conv1d(1, activation=None))
        else:
            self.FC_proposal = (
                pt_utils.Seq(vote_channel)
                    .conv1d(vote_channel, bn=True)
                    .conv1d(vote_channel, bn=True)
                    .conv1d(3 + 1 + 1, activation=None))

        self.Importance_layer = (
            pt_utils.Seq(feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(1, activation=None))

        self.vote_layer = (
            pt_utils.Seq(3 + feature_channel)
                .conv1d(feature_channel, bn=True)
                .conv1d(3 + feature_channel, activation=None))

        self.vote_aggregation = PointnetSAModule(
            radius=0.3,
            nsample=16,
            mlp=[feature_channel, vote_channel, vote_channel, vote_channel],
            use_xyz=True,
            normalize_xyz=normalize_xyz)



    def forward(self, xyz, feature):
        """
        :param xyz: B,N,3
        :param feature: B,f,N
        :return: B,N,4+1 (xyz,theta,targetnessscore)
        """
        if self.transformer:
            feature = feature.permute(0, 2, 1)
            feature_g = self.Global_transformer(xyz, feature).permute(0, 2, 1)
            importances = self.Importance_layer(feature_g).squeeze(1)
            feature_gl = self.Local_transformer(xyz, feature_g.permute(0, 2, 1)).permute(0, 2, 1)
            xyz_feature = torch.cat((xyz.transpose(1, 2).contiguous(), feature_gl), dim=1)
        else:
            importances = self.Importance_layer(feature).squeeze(1)
            xyz_feature = torch.cat((xyz.transpose(1, 2).contiguous(), feature), dim=1)

        offset = self.vote_layer(xyz_feature)
        vote = xyz_feature + offset
        vote_xyz = vote[:, 0:3, :].transpose(1, 2).contiguous()
        vote_feature = vote[:, 3:, :].contiguous()

        center_xyzs, proposal_features = self.vote_aggregation(vote_xyz, vote_feature, self.num_proposal)

        if self.de_head:
            proposal_offsets = self.FC_proposal_offset(proposal_features)
            proposal_angles = self.FC_proposal_angle(proposal_features)
            proposal_scores = self.FC_proposal_score(proposal_features)
            estimation_boxes = torch.cat(
                (proposal_offsets + center_xyzs.transpose(1, 2).contiguous(), proposal_angles),
                dim=1)
            estimation_boxes = torch.cat((estimation_boxes, proposal_scores), dim=1)
        else:
            proposal_offsets = self.FC_proposal(proposal_features)
            estimation_boxes = torch.cat(
                (proposal_offsets[:, 0:3, :] + center_xyzs.transpose(1, 2).contiguous(), proposal_offsets[:, 3:5, :]),
                dim=1)
        estimation_boxes = estimation_boxes.transpose(1, 2).contiguous()

        return estimation_boxes, vote_xyz, center_xyzs, importances