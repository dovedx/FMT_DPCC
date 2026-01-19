import numpy as np
import open3d
import torch
import torch.utils.data as data
from os.path import join
import os
import MinkowskiEngine as ME
import random
from pyntcloud import PyntCloud


class Dataset(data.Dataset):
    def __init__(self, root_dir, split, bit=10, maximum=20475, type='train', scaling_factor=1, time_step=1, format='npy'):
        self.maximum = maximum
        self.type = type
        self.scaling_factor = scaling_factor
        self.format = format
        sequence_list = ['soldier', 'redandblack', 'loot', 'longdress']
        self.sequence_list = sequence_list
        start = [536, 1450, 1000, 1051, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1]
        end = [835, 1749, 1299, 1350, 317, 600, 600, 215, 600, 244, 249, 215, 206, 600]
        num = [end[i] - start[i] for i in range(len(start))]
        self.lookup = []
        self.num_videos =1
        self.frames_per_video=600
        self.gop_size=16
        self.ra_order = [0, 8, 4, 2, 6, 1, 3, 5, 7, 12, 10, 14, 9, 11, 13, 15]
        
        # print("一共测试多少帧:",num)
        
        
        for i in split:
            video = []
            sequence_dir = join(root_dir, sequence_list[i]+'_ori/Ply')
            file_prefix  = sequence_list[i]+'_vox'+str(bit)+'_'
            file_subfix  = '.'+self.format
            s = start[i]
            e = end[i]
            for s in range(s, e):
                s0 = str(s+1).zfill(4)
                file_name = file_prefix + s0 + file_subfix
                file_dir = join(sequence_dir, file_name)
                video.append(file_dir)
                
            self.lookup.append(video)
        
        self.total_frames = self.num_videos * self.frames_per_video
        
        
        # 生成按RA顺序的索引
        self.indices = []
        for video_idx in [0]:
            for gop_idx in range(self.frames_per_video // self.gop_size):
                gop_start = gop_idx * self.gop_size
                for rel_idx in self.ra_order:
                    frame_idx = gop_start + rel_idx
                    self.indices.append((video_idx, frame_idx))
        
        
        # print(" self.indices:", self.indices)
        
    def __getitem__(self, idx):
        video_idx, frame_idx = self.indices[idx]
        
        
        
        # print(video_idx,"=====",frame_idx)
        
        current_frame_name =  self.lookup[video_idx][frame_idx]
        file_dir  = current_frame_name
        
        print("编码顺序:",video_idx,frame_idx,file_dir)
        
        if self.format == 'npy':
            p = np.load(file_dir)
        elif self.format == 'ply':
           
            p  = np.asarray(PyntCloud.from_file(file_dir).points.values) // self.scaling_factor
           
        pc = torch.tensor(p[:, :3]).cuda()
        pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
       
        # if self.scaling_factor != 1:
        #     pc = torch.unique(torch.floor(pc / self.scaling_factor), dim=0)
        #     pc1 = torch.unique(torch.floor(pc1 / self.scaling_factor), dim=0)
        xyz, point = pc, torch.ones_like(pc[:, :1])
       
        
        # print(xyz.shape)
        # print(point.shape)
        # print(xyz1.shape)
        # print(point1.shape)

        return xyz, point ,(video_idx,frame_idx)

    # def __len__(self):
    #     return self.total_frames

    def __len__(self):
        return len(self.indices)


def collate_pointcloud_fn(list_data):
    new_list_data = []
    num_removed = 0
    for data in list_data:
        if data is not None:
            new_list_data.append(data)
        else:
            num_removed += 1

    list_data = new_list_data

    if len(list_data) == 0:
        raise ValueError('No data in the batch')

    # coords, feats, labels = list(zip(*list_data))
    xyz, point,indices= list(zip(*list_data))
    
    xyz_batch = ME.utils.batched_coordinates(xyz)
    point_batch = torch.vstack(point).float()
    
    
    return xyz_batch, point_batch ,indices[0]

