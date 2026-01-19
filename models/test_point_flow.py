import argparse
import sys 
import os 

import torch, numpy as np, glob, math, torch.utils.data, scipy.ndimage, multiprocessing as mp
import torch.nn.functional as F
import time
import torch.nn as nn
import pickle 
import datetime
import logging

from tqdm import tqdm 


import sys
sys.path.append('../')

from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow
from PointPWC.models import multiScaleLoss
from PointPWC.pointnet2 import pointnet2_utils
from pathlib import Path
from collections import defaultdict

# import transforms
# import datasets
# import cmd_args 
# from main_utils import *
# from utils import geometry
from PointPWC.evaluation_utils import evaluate_2d, evaluate_3d

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


import random
from pyntcloud import PyntCloud


def index_points_gather(points, fps_idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """

    points_flipped = points.permute(0, 2, 1).contiguous()
    new_points = pointnet2_utils.gather_operation(points_flipped, fps_idx)
    return new_points.permute(0, 2, 1).contiguous()

def fps_(point,num):
    torch_point = torch.from_numpy(point).unsqueeze(0).contiguous().cuda()
    fps_idx = pointnet2_utils.furthest_point_sample(torch_point, num)
    new_xyz = index_points_gather(torch_point, fps_idx)
    
    print("======>",new_xyz.shape)
    
    return new_xyz.cpu().numpy()[0,...]
    

def processdata(point1,point2):
    if point1 is None:
            return None, None, None,
    num_points=52000
    num_points2=50000
    allow_less_points=True
    no_corr = True
    
    # pc1 = fps_(point=point1,num=num_points)
    # pc2 = fps_(point=point2,num=num_points2)
    
    offset = np.ones_like(point1)*5
    print("offset:",offset.shape)
    pc1=point1
    pc2=point2
    print(pc1[1:10,...])
    print(pc2[1:10,...])
    print("===============>",pc1.shape,pc2.shape)
    
#     # sf = pc2- pc1
    sf=0
    return torch.from_numpy(pc1), torch.from_numpy(pc2),sf

#     DEPTH_THRESHOLD=350
#     # if DEPTH_THRESHOLD > 0:
#     #     near_mask = np.logical_and(pc1[:, 2] < DEPTH_THRESHOLD, pc2[:, 2] < DEPTH_THRESHOLD)
#     # else:
#     #     near_mask = np.ones(pc1.shape[0], dtype=np.bool)
#     near_mask = np.ones(pc1.shape[0], dtype=np.bool)
#     near_mask2 = np.ones(pc2.shape[0], dtype=np.bool)
#     indices = np.where(near_mask)[0]
#     indices2 = np.where(near_mask2)[0]
#     if len(indices) == 0:
#         print('indices = np.where(mask)[0], len(indices) == 0')
#         return None, None, None

#     if num_points > 0:
#         try:
#             sampled_indices1 = np.random.choice(indices, size=num_points, replace=False, p=None)
#             if no_corr:
#                 sampled_indices2 = np.random.choice(indices2, size=num_points2, replace=False, p=None)
#             else:
#                 sampled_indices2 = sampled_indices1
#         except ValueError:
#             '''
#             if not self.allow_less_points:
#                 print('Cannot sample {} points'.format(self.num_points))
#                 return None, None, None
#             else:
#                 sampled_indices1 = indices
#                 sampled_indices2 = indices
#             '''
#             if not allow_less_points:
#                 #replicate some points
#                 sampled_indices1 = np.random.choice(indices, size=num_points, replace=True, p=None)
#                 if no_corr:
#                     sampled_indices2 = np.random.choice(indices, size=num_points2, replace=True, p=None)
#                 else:
#                     sampled_indices2 = sampled_indices1
#             else:
#                 sampled_indices1 = indices
#                 sampled_indices2 = indices
#         else:
#             sampled_indices1 = indices
#             sampled_indices2 = indices
        
        
#         print("sampled_indices1:",len(sampled_indices1))
#         pc1 = pc1[sampled_indices1]
#         # sf = sf[sampled_indices1]
#         pc2 = pc2[sampled_indices2]
        

        # return torch.from_numpy(pc1), torch.from_numpy(pc2), torch.from_numpy(sf)


# point_path ='/dengx/dengxuan/AVS_P/train_data/MPEG_8i/longdress_ori/Ply/'

point_path = "/dengx/dengxuan/AVS_P/D-DPCC-main/midde_result/"

# file_dir0 = point_path + 'longdress_vox10_1051.ply'
# file_dir1 = point_path + 'longdress_vox10_1052.ply'

file_dir0 = point_path + 'ys_ref.ply'
file_dir1 = point_path + 'ys_cur.ply'


scaling_factor=4

p0 = np.asarray(PyntCloud.from_file(file_dir0).points.values)[:,:3]
p1 = np.asarray(PyntCloud.from_file(file_dir1).points.values)[:,:3] 
print("1======>",p0.shape,p1.shape,np.max(p0),np.min(p0),np.max(p0)-np.min(p0))
p0 = np.unique(p0//scaling_factor,axis=0)
p1 = np.unique(p1//scaling_factor,axis=0)
print("2======>",p0.shape,p1.shape,np.max(p0),np.min(p0),np.max(p0)-np.min(p0))

p0 = (35 / 64) * p0- 35
p1 = (35 / 64) * p1- 35
print("3======>",p0.shape,p1.shape,np.max(p0),np.min(p0),np.max(p0)-np.min(p0))
p0[:,0]=p0[:,0]+5


model = PointConvSceneFlow()





pretrain = '/dengx/dengxuan/AVS_P/D-DPCC-main/PointPWC/pretrain_weights/PointConv_726_0.0463.pth' 
model.load_state_dict(torch.load(pretrain))
print('load model %s'%pretrain)
# logger.info('load model %s'%pretrain)

model.cuda()
model.eval()

print("len dict:",len(model.state_dict()))
# print("cost0.weightnet1.mlp_convs.0.weight:",model.state_dict()["cost0.weightnet1.mlp_convs.0.weight"])
# print("flow3.mlp_convs.1.composed_module.0.weight:",model.state_dict()["flow3.mlp_convs.1.composed_module.0.weight"])
# print("level2_0.composed_module.0.weight:",model.state_dict()["level2_0.composed_module.0.weight"])
# print("flow3.pointconv_list.0.weightnet.mlp_convs.0.weight:",model.state_dict()["flow3.pointconv_list.0.weightnet.mlp_convs.0.weight"])


pc1_transformed, pc2_transformed, sf_transformed = processdata(p0,p1)

pc1_norm = pc1_transformed
pc2_norm = pc2_transformed

flow = sf_transformed

pos1 = pc1_transformed.unsqueeze(0).contiguous().cuda()
pos2 = pc2_transformed.unsqueeze(0).contiguous().cuda() 
norm1 = pos1
norm2 = pos2
# flow = flow.unsqueeze(0).contiguous().cuda() 

# model.eval()
with torch.no_grad(): 
    
    
    print("pos1, pos2:",pos1.shape,pos2.shape)
    pred_flows, fps_pc1_idxs, _, _, _ = model(pos1, pos2, norm1, norm2)

    # loss = multiScaleLoss(pred_flows, flow, fps_pc1_idxs)

#     full_flow = pred_flows[0].permute(0, 2, 1)
#     epe3d = torch.norm(full_flow - flow, dim = 2).mean()
    
    print("pred_flows shape:",pred_flows[0].shape)
    print(pred_flows[0][0],torch.sum(torch.abs(pred_flows[0])))
    # print("gtflow:",torch.sum(torch.abs(flow)))
    # print(flow)
    
    # print("epe3d:",epe3d)
    
    motion =pred_flows[0].cpu().numpy().squeeze()
    points = pos1.cpu().numpy().squeeze()
    
    
    motion = motion.transpose(1,0).astype(np.float64)
    print("motion points:",motion.shape , points.shape)
    
    motion_norm = (motion - motion.min()) / (motion.max() - motion.min())
    motion_norm = motion_norm.astype(np.float64)
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(motion_norm)  # 用 motion 作为颜色
    
    print("save ...")
    o3d.io.write_point_cloud("motion_flow.ply", pcd)
