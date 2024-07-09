#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

# Modifications:
# - Modified by FusionAD on 2023.5
# - Added extended support from FusionAD (https://arxiv.org/abs/2308.01006)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import pickle
from mmdet.models import LOSSES


@LOSSES.register_module()
class PlanningLoss(nn.Module):
    def __init__(self, loss_type='L2'):
        super(PlanningLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, sdc_traj, gt_sdc_fut_traj, mask):
        err = sdc_traj[..., :2] - gt_sdc_fut_traj[..., :2]
        err = torch.pow(err, exponent=2)
        err = torch.sum(err, dim=-1)
        err = torch.pow(err, exponent=0.5)
        return torch.sum(err * mask)/(torch.sum(mask) + 1e-5)


@LOSSES.register_module()
class CollisionLoss(nn.Module):
    def __init__(self, delta=0.5, weight=1.0):
        super(CollisionLoss, self).__init__()
        self.w = 1.85 + delta
        self.h = 4.084 + delta
        self.weight = weight
    
    def forward(self, sdc_traj_all, sdc_planning_gt, sdc_planning_gt_mask, future_gt_bbox):
        # sdc_traj_all (1, 6, 2)
        # sdc_planning_gt (1,6,3)
        # sdc_planning_gt_mask (1, 6)
        # future_gt_bbox 6x[lidarboxinstance]
        n_futures = len(future_gt_bbox)
        inter_sum = sdc_traj_all.new_zeros(1, )
        dump_sdc = []
        for i in range(n_futures):
            if len(future_gt_bbox[i].tensor) > 0:
                future_gt_bbox_corners = future_gt_bbox[i].corners[:, [0,3,4,7], :2] # (N, 8, 3) -> (N, 4, 2) only bev 
                # sdc_yaw = -sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype) - 1.5708
                sdc_yaw = sdc_planning_gt[0, i, 2].to(sdc_traj_all.dtype)
                sdc_bev_box = self.to_corners([sdc_traj_all[0, i, 0], sdc_traj_all[0, i, 1], self.w, self.h, sdc_yaw])
                dump_sdc.append(sdc_bev_box.cpu().detach().numpy())
                for j in range(future_gt_bbox_corners.shape[0]):
                    inter_sum += self.inter_circles(sdc_bev_box, future_gt_bbox_corners[j].to(sdc_traj_all.device))
        return inter_sum * self.weight
        
    def inter_bbox(self, corners_a, corners_b):
        xa1, ya1 = torch.max(corners_a[:, 0]), torch.max(corners_a[:, 1])
        xa2, ya2 = torch.min(corners_a[:, 0]), torch.min(corners_a[:, 1])
        xb1, yb1 = torch.max(corners_b[:, 0]), torch.max(corners_b[:, 1])
        xb2, yb2 = torch.min(corners_b[:, 0]), torch.min(corners_b[:, 1])
        
        xi1, yi1 = min(xa1, xb1), min(ya1, yb1)
        xi2, yi2 = max(xa2, xb2), max(ya2, yb2)
        intersect = max((xi1 - xi2), xi1.new_zeros(1, ).to(xi1.device)) * max((yi1 - yi2), xi1.new_zeros(1,).to(xi1.device))
        return intersect

    def inter_circles(self,corners_a, corners_b):
        a_centers = self.get_circle_centers(corners_a)
        b_centers = self.get_circle_centers(corners_b)
        min_dis = torch.min(torch.cdist(a_centers, b_centers)).to(corners_a.device)

        return torch.max((self.cal_rectangle_width(corners_a) + self.cal_rectangle_width(corners_b)) / 2 - min_dis, torch.zeros(1).to(corners_a.device)).to(corners_a.device)

    def cal_distance(self,p1, p2):
        return torch.sqrt(torch.sum((p1 - p2)) ** 2)

    def cal_rectangle_width(self, vertices):
        n = vertices.shape[0]
        distance = [self.cal_distance(vertices[i], vertices[(i+1) % n]) for i in range(n)]
        return min(distance).to(vertices.device)

    def cal_rectangle_length(self, vertices):
        edge_length = [torch.sqrt(torch.sum((vertices[i] - vertices[i - 1]) ** 2)) for i in range(4)]
        return torch.argmax(torch.tensor(edge_length)).to(vertices.device), max(edge_length).to(vertices.device)

    def rectangle_length_slope(self, vertices):
        long_edge_index, length = self.cal_rectangle_length(vertices)

        delta_x = vertices[long_edge_index][0] - vertices[long_edge_index - 1][0]
        delta_y = vertices[long_edge_index][1] - vertices[long_edge_index - 1][1]

        slope = delta_y / delta_x
        return torch.atan(slope)

    def rectangle_center(self, vertices):
        return torch.mean(vertices, dim = 0)

    def center_alone_length(self, center_point, slope, distance):
        detla_x = distance * torch.cos(slope)
        delta_y = distance * torch.sin(slope)
        return torch.tensor([center_point[0] + detla_x, center_point[1] + delta_y])

    def get_circle_centers(self, vertices):
        rectangle_width = self.cal_rectangle_width(vertices)
        length_index, rectangle_length = self.cal_rectangle_length(vertices)
        length_slope = self.rectangle_length_slope(vertices)

        center1 = self.rectangle_center(vertices)
        center2 = self.center_alone_length(center1, length_slope, rectangle_length / 2 - rectangle_width / 2).to(center1.device)
        center3 = self.center_alone_length(center1, length_slope, -(rectangle_length / 2 - rectangle_width / 2)).to(center1.device)
        center4 = self.center_alone_length(center1, length_slope, (rectangle_length / 2 - rectangle_width / 2) / 2).to(center1.device)
        center5 = self.center_alone_length(center1, length_slope, -(rectangle_length / 2 - rectangle_width / 2) / 2).to(center1.device)

        return torch.stack([center1, center2, center3, center4, center5], dim = 0)

    def to_corners(self, bbox):
        x, y, w, l, theta = bbox
        corners = torch.tensor([
            [w/2, -l/2], [w/2, l/2], [-w/2, l/2], [-w/2,-l/2]  
        ]).to(x.device) # 4,2
        rot_mat = torch.tensor(
            [[torch.cos(theta), torch.sin(theta)],
             [-torch.sin(theta), torch.cos(theta)]]
        ).to(x.device)
        new_corners = rot_mat @ corners.T + torch.tensor(bbox[:2])[:, None].to(x.device)
        return new_corners.T