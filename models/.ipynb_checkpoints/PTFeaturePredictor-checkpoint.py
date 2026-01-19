import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
# from torch_cluster import knn
from lib.pointops.functions import pointops
from pointconv_util import*

from pytorch3d.ops import knn_points

from point_conv import*
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib import cm
import os
from matplotlib.colors import ListedColormap

blue = np.array([0, 0, 255]) / 255.0
yellow = np.array([255, 215, 0]) / 255.0


def write_ply_data(filename, coords):
    if os.path.exists(filename):
        os.system('rm ' + filename)
    f = open(filename, 'a+')
    # print('data.shape:',data.shape)
    f.writelines(['ply\n', 'format ascii 1.0\n'])
    f.write('element vertex ' + str(coords.shape[0]) + '\n')
    f.writelines(['property float x\n', 'property float y\n', 'property float z\n'])
    f.write('end_header\n')
    for _, point in enumerate(coords):
        f.writelines([str(point[0]), ' ', str(point[1]), ' ', str(point[2]), '\n'])
    f.close()
    return

def save_error_map_as_ply(coords, feat_gt, feat_pred, out_ply="error_map_two_color.ply", colormap='jet'):
    """
    将特征对齐误差映射为颜色并保存为 PLY 点云文件
    """
    # Step 1: 计算误差（点特征的L2距离）
    errors = np.linalg.norm(feat_gt - feat_pred, axis=1)  # shape [N]
    
    # Step 2: 归一化误差到 [0, 1]
    norm_errors = (errors - errors.min()) / (errors.max() - errors.min() + 1e-8)
    
    # Step 3: 映射为 RGB 颜色（使用 colormap，如 jet, viridis, magma 等）
    colormap_func = cm.get_cmap(colormap)
    rgb_colors = colormap_func(norm_errors)[:, :3]  # shape [N, 3], 去掉 alpha 通道
    
    # Step 4: 创建 Open3D 点云并赋予颜色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    # Step 5: 保存为 .ply 文件
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"误差着色点云保存成功: {out_ply}")


def save_binary_error_map_as_ply(
    coords, 
    feat_gt, 
    feat_pred, 
    out_ply="binary_error_map.ply", 
    threshold=0.1, 
   
):
    """
    将误差映射为二值颜色（如蓝/红）并保存为 PLY 点云文件。
    
    参数：
    - coords: [N, 3] 点坐标
    - feat_gt: [N, C] 真实特征
    - feat_pred: [N, C] 预测特征
    - threshold: float，误差大于此值将标为失败（红色）
    """
    # Step 1: 计算误差
    errors = np.linalg.norm(feat_gt - feat_pred, axis=1)  # shape [N]
    print("error:",errors)

    # Step 2: 生成二值标签
    binary_mask = errors > 1  # True 表示成功，对应蓝色

    # Step 3: 分配颜色
   # Step 3: 映射为 RGB 颜色（两种颜色）
    cmap = ListedColormap(["red", "blue"])
    rgb_colors = cmap(binary_mask.astype(int))[:, :3]

    # Step 4: 创建点云并着色
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords)
    pcd.colors = o3d.utility.Vector3dVector(rgb_colors)

    # Step 5: 保存为 .ply 文件
    o3d.io.write_point_cloud(out_ply, pcd)
    print(f"二值误差图已保存到: {out_ply}")


class PointCloudFeaturePredictor(nn.Module):
    def __init__(self, in_channels=128, K=8,point_conv=False):
        """
        初始化点云特征预测网络
        参数:
            in_channels: 输入特征的通道数 (C=128)
            K: 最近邻数
        """
        super().__init__()
        self.K = K
        self.knn=K
        self.in_channels = in_channels
        self.point_conv=point_conv 
        if self.point_conv:
            
            self.encoder_point_weight = EncoderBlock(in_channels=68*self.knn,  out_channels=68*self.knn, n_neighbors=self.knn)
            
            self.weight_mlp = nn.Sequential(
                nn.Linear(68*self.knn, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.knn),
                nn.Softmax(dim=1)
            )
          
        else:
            # 插值权重网络
            self.weight_mlp = nn.Sequential(
                nn.Linear(68, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1)
            )
        
        if self.point_conv:
            self.encoder_point_out = EncoderBlock(in_channels=in_channels,  out_channels=64, n_neighbors=self.knn)
        else:
            # 输出映射网络
            self.out_mlp = nn.Sequential(
                nn.Linear(in_channels, 256),
                nn.ReLU(),
                nn.Linear(256, 64)
            )

    def compute_knn_matrix(self, coords1: torch.Tensor, coords2: torch.Tensor):
        """
        计算 Point2 到 Point1 的 KNN 对应矩阵
        参数:
            coords1: Point1 的坐标 (N1, D)
            coords2: Point2 的坐标 (N2, D)
            batch1: Point1 的批次索引 (N1,)
            batch2: Point2 的批次索引 (N2,)
        返回:
            edge_index: KNN 对应矩阵 (2, E)
        """
        coords1 =  coords1.unsqueeze(0).float()
        coords2 =  coords2.unsqueeze(0).float()
        dist, knn_idx, __ = knn_points(coords2, coords1, K=self.K)#xyz2在xyz1中搜索最近邻
        
        # print("coords1,coords2,knn_idx:",coords1.shape,coords2.shape,knn_idx.shape)
        
        return knn_idx
        # xyz1,xyz2,knn_idx: torch.Size([1, 76796, 3]) torch.Size([1, 77294, 3]) torch.Size([1, 77294, 3])
        
        # edge_index = knn(coords1, coords2, self.K, batch1, batch2)
        # return edge_index
        
    
    
    def decoder_side(self, f1, f2_coor, ys2_4_coor,f2=None):
        coords1 = f1.C.float()[:, 1:]  # Point1 坐标 (N1, D)
        # coords2 = f2_coor[:, 1:]  # Point2 坐标 (N2, D)
        coords2  = f2.C.float()[:, 1:]
        feats1  = f1.F  # Point1 特征 (N1, C)
        feats2 = f2.F
        if f2!=None:
            feat2 = f2.F
        
        knn_idx = self.compute_knn_matrix(coords1,coords2)
        
        # print("knnidx coord1:",coords1[1:10,...])
        # print("knnidx coord2:",coords2[1:10,...])
        
        #从坐标域中找到相对位置
        N2,C =coords2.shape
        grouped_xyz_norm = index_points_group(coords1.unsqueeze(0), knn_idx) - coords2.view(1, N2, 1, C) # B N2 3 C
        # print("grouped_xyz_norm:",grouped_xyz_norm.shape)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        coords_weight = (1.0 / dist) / norm 
        coords_weight = coords_weight.view(1, N2,self.K, 1)
        
        if f2==None:
            neighbor_feats = index_points_group(feats1.unsqueeze(0), knn_idx)
            fea_agg = torch.cat([grouped_xyz_norm,neighbor_feats,coords_weight],dim=3)#3+64+5
        else:
            # print("====================>")
            neighbor_feats = index_points_group(feats1.unsqueeze(0), knn_idx)
            # feats2_expanded = feats2.unsqueeze(1).unsqueeze(0)  # (N2, 1, C)
            # feat_diff = feats2_expanded - neighbor_feats
            # fea_agg = torch.cat([grouped_xyz_norm,feat_diff,coords_weight],dim=3)#3+64+5
            fea_agg = torch.cat([grouped_xyz_norm,neighbor_feats,coords_weight],dim=3)#3+64+5
        
        weights = self.weight_mlp(fea_agg)  # (N2, K, 1)#这个weight就相当与间接的隐式光流
        
        # print("weights:",weights[0,1:10,0,0])
        
        #最后得到的融合权重为weight: torch.Size([1, 58, 5, 1])
        # print("weight:",weights.shape)
        # 加权插值
        interpolated_feats = (weights * neighbor_feats)# (N2, C)
        # print("interpolated_feats:",interpolated_feats.shape)
        interpolated_feats = torch.sum(interpolated_feats,dim=2)
        # print("interpolated_feats:",interpolated_feats.shape)
        # 输出映射
        predicted_feats = self.out_mlp(interpolated_feats)  # (N2, 128)
        
        
        # predicted_f2 = ME.SparseTensor(predicted_feats[0], coordinates=f2_coor, tensor_stride=4)
        predicted_f2 = ME.SparseTensor(predicted_feats[0], coordinates=f2.C, tensor_stride=4)
        
        
        #保存5个特征coords1,coords2,feats1,feats2,predicted_feats[0]
        
       
        
        return predicted_f2


    def forward(self, y1: ME.SparseTensor, y2: ME.SparseTensor):
        """
        前向传播
        参数:
            y1: 前一帧点云 (Point1, feature1)
            y2: 当前帧点云 (Point2, feature2)
        返回:
            predicted_y2: 预测的 feature2'
            residual: 残差 R = feature2 - feature2'
        """
        # 提取坐标和特征
         # 提取坐标和特征
        
        coords1 = y1.C.float()[:, 1:]  # Point1 坐标 (N1, D)
        coords2 = y2.C.float()[:, 1:]  # Point2 坐标 (N2, D)
        feats1 = y1.F  # Point1 特征 (N1, C)
        feats2 = y2.F  # Point2 特征 (N2, C)
        batch1 = y1.C[:, 0]  # Point1 批次索引
        batch2 = y2.C[:, 0]  # Point2 批次索引
        N2=feats2.shape[0]

#         # 计算固定 KNN 对应矩阵
#         edge_index = self.compute_knn_matrix(coords1, coords2, batch1, batch2)
#         src, dst = edge_index  # src: Point1 索引 (E,), dst: Point2 索引 (E,)

#         # 提取邻域特征矩阵 (N2, K, C)
#         neighbor_feats = feats1[src]  # (E, C)
#         N2 = coords2.shape[0]
#         neighbor_feats = neighbor_feats.view(N2, self.K, -1)  # (N2, K, C)
        #计算固定的KNN矩阵
        knn_idx = self.compute_knn_matrix(coords1,coords2)
        
        # print("knnidx coord1:",coords1[1:20,...])
        # print("knnidx coord2:",coords2[1:20,...])
        
        #从坐标域中找到相对位置
        N2,C =coords2.shape
        grouped_xyz_norm = index_points_group(coords1.unsqueeze(0), knn_idx) - coords2.view(1, N2, 1, C) # B N2 3 C
        # print("grouped_xyz_norm:",grouped_xyz_norm.shape)
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        coords_weight = (1.0 / dist) / norm 
        coords_weight = coords_weight.view(1, N2,self.K, 1)
        # print("coords_weight:",coords_weight.shape)
        
        #从特征域中找到相邻特征
        neighbor_feats = index_points_group(feats1.unsqueeze(0), knn_idx)
        # print("neighbor_feats:",neighbor_feats.shape)
        
        # 计算相对特征距离矩阵 (N2, K, C)
        # feats2_expanded = feats2.unsqueeze(1).unsqueeze(0)  # (N2, 1, C)
        # feat_diff = feats2_expanded - neighbor_feats  # (N2, K, C)
        # # print("feat_diff:",feat_diff.shape)
        # #[1*65*5*128]
        # #有了coords的相对位置grouped_xyz_norm[1*55*5*3]
        # # featrue域上的相对特征feat_diff[1,55,5,128]
        # #coords域的权重位置，[1,55,5,1]
        # #将这些先验只是都concat在一起
        # fea_agg = torch.cat([grouped_xyz_norm,feat_diff,coords_weight],dim=3)#3+64+5
        fea_agg = torch.cat([grouped_xyz_norm,neighbor_feats,coords_weight],dim=3)#3+64+5
        # ===>: torch.Size([1, 47187, 5, 3]) torch.Size([1, 47187, 5, 64]) torch.Size([1, 47187, 5, 1])
        # print("fea_agg:",fea_agg.shape)
        # fea_agg: torch.Size([1, 62, 5, 132])
        # # 归一化相对特征距离
        # feat_diff = (feat_diff - feat_diff.mean(dim=(1, 2), keepdim=True)) / (feat_diff.std(dim=(1, 2), keepdim=True) + 1e-8)
        # print("feat_diff:",feat_diff.shape)
        #[N*K*C]
        # 学习插值权重 (N2, K, 1)
        
        
        
        #到了这里，有了当前编码帧的coordinate和之前编码帧相关的特征信息
        #坐标信息:f2.C
        #坐标对应的特征信息:fea_agg
        if self.point_conv:
            xyzs = y2.C[:,1:].unsqueeze(0).float()
            knn_idx = pointops.knnquery_heap(self.knn, xyzs.contiguous(),xyzs.contiguous()).long()
            #输入: # xyz: B * 3 * N  |  fea: B * C * N | knn_idx: B * N * K
            #fea_agg维度为(1xN*K*C)
            fea_agg = fea_agg.reshape(1,N2,-1)
            fea_agg = fea_agg.transpose(1, 2).contiguous()
            # print("point_conv input:",xyzs.transpose(1, 2).contiguous().shape,fea_agg.shape,knn_idx.shape)
            features_fusion = self.encoder_point_weight(xyzs=xyzs.transpose(1, 2).contiguous(), feats=fea_agg, knn_idx=knn_idx,index=1)
            features_fusion = features_fusion.transpose(1,2)
            # print("features_fusion:",features_fusion.shape)
            
            weights = self.weight_mlp(features_fusion)
            weights = weights.unsqueeze(3)
            
            # print("weights:",weights.shape,neighbor_feats.shape)
        else:
            weights = self.weight_mlp(fea_agg)  # (N2, K, 1)#这个weight就相当与间接的隐式光流
        # print("weights:",weights[0,1:10,0,0])
        
        #最后得到的融合权重为weight: torch.Size([1, 58, 5, 1])
        # print("weight:",weights.shape, neighbor_feats.shape)
        # weight: torch.Size([1, 51066, 8, 1]) torch.Size([1, 51066, 8, 64])
        # 加权插值
        interpolated_feats = (weights * neighbor_feats)# (N2, C)
        # print("interpolated_feats:",interpolated_feats.shape)
        interpolated_feats = torch.sum(interpolated_feats,dim=2)
        # interpolated_feats: torch.Size([1, 51066, 64])
        # print("interpolated_feats:",interpolated_feats.shape)
        # 输出映射
        if self.point_conv:
            interpolated_feats = interpolated_feats.transpose(1, 2).contiguous()
            knn_idx = pointops.knnquery_heap(self.knn, xyzs.contiguous(),xyzs.contiguous()).long()
            #输入: # xyz: B * 3 * N  |  fea: B * C * N | knn_idx: B * N * K
            # print("point_conv input:",xyzs.transpose(1, 2).contiguous().shape,interpolated_feats.shape,knn_idx.shape)
            predicted_feats = self.encoder_point_out(xyzs=xyzs.transpose(1, 2).contiguous(), feats=interpolated_feats, knn_idx=knn_idx,index=2)
            predicted_feats=predicted_feats.transpose(1,2)
            # print("predicted_feats:",predicted_feats.shape)
        else:
            predicted_feats = self.out_mlp(interpolated_feats)  # (N2, 128)
        # 计算残差
        # print("feats2",feats2.shape)
        # print("predicted_feats:",predicted_feats.shape)
        residual = feats2 - predicted_feats[0]  # (N2, 128)
        
        residual_y2=ME.SparseTensor(
            features=residual,
            coordinates=y2.C,
            coordinate_manager=y2.coordinate_manager,
            tensor_stride=4,
            device=y2.device
        )
        # 返回预测特征和残差
        predicted_y2 = ME.SparseTensor(
            features=predicted_feats[0],
            coordinates=y2.C,
            coordinate_manager=y2.coordinate_manager,
            tensor_stride=4,
            device=y2.device
        )
        
        # print("预测的feature:recon_predicted_y2:",predicted_y2.shape,torch.sum(predicted_y2.F) )
        
        # print("coords1:",coords1.shape)
        
#         write_ply_data('./visual_pcd/coords1.ply', np.array(coords1.cpu().numpy())//4)
#         write_ply_data('./visual_pcd/coords2.ply', np.array(coords2.cpu().numpy())//4)
        
#         print("y1 stride:",y1.tensor_stride)
#         print("np.array(coords2.cpu().numpy())//4",np.max(np.array(coords2.cpu().numpy())))
        
#         np.save("./visual_pcd/coords1.npy", np.array(coords1.cpu().numpy())//4)
#         print("coords2:",coords2.shape)
#         np.save("./visual_pcd/coords2.npy", np.array(coords2.cpu().numpy())//4)
#         print("feats1:",feats1.shape)
#         np.save("./visual_pcd/feats1.npy", np.array(feats1.cpu().numpy()))
#         print("feats2:",feats2.shape)
#         np.save("./visual_pcd/feats2.npy", np.array(feats2.cpu().numpy()))
#         print("predicted_feats[0]:",predicted_feats[0].shape)
#         np.save("./visual_pcd/predicted_feats.npy", np.array(predicted_feats[0].cpu().numpy()))
#         print("interpolated_feats:",interpolated_feats.shape)
#         np.save("./visual_pcd/interpolated_feats.npy", np.array(interpolated_feats[0].cpu().numpy()))
        
        # save_error_map_as_ply(coords=np.array(coords2.cpu().numpy())//4, feat_gt=np.array(feats2.cpu().numpy()), feat_pred=np.array(interpolated_feats[0].cpu().numpy()), out_ply="./visual_pcd/error_map.ply", colormap='jet')
        # save_binary_error_map_as_ply(coords=np.array(coords2.cpu().numpy()//4), \
        #                       feat_gt=np.array(feats2.cpu().numpy()),\
        #                       feat_pred=np.array(interpolated_feats[0].cpu().numpy()),\
        #                       out_ply="./visual_pcd/error_map_two_color.ply", threshold=0.1)
        
        return predicted_y2, residual_y2,residual

# 示例使用
if __name__ == "__main__":
    # 参数
    in_channels = 128
    K = 5

    # 创建模型
    model = PointCloudFeaturePredictor(in_channels, K)

    # 假设输入
    N1, N2 = 100, 120
    coords1 = torch.cat([torch.zeros(N1, 1), torch.randn(N1, 3)], dim=1)
    coords2 = torch.cat([torch.zeros(N2, 1), torch.randn(N2, 3)], dim=1)
    feats1 = torch.randn(N1, in_channels)
    feats2 = torch.randn(N2, in_channels)
    y1 = ME.SparseTensor(features=feats1, coordinates=coords1)
    y2 = ME.SparseTensor(features=feats2, coordinates=coords2)

    # 前向传播
    predicted_y2, residual = model(y1, y2)
    print(f"Predicted feature shape: {predicted_y2.F.shape}")  # (N2, 128)
    print(f"Residual shape: {residual.shape}")  # (N2, 128)

    # 假设损失函数
    loss = torch.mean(residual ** 2)  # MSE 损失
    print(f"Initial loss: {loss.item()}")
