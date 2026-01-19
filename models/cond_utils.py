import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.pointconv_util import*

from models.resnet import ResNet, InceptionResNet
from pytorch3d.ops import knn_points


class Point_context_produce(nn.Module):
    def __init__(self, channels=64):
        super().__init__() 
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        # self.res_block1 = InceptionResNet(channels=channels)
        
        self.conv_down1=ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.res_down1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.motion_down1=ME.MinkowskiConvolution(
            in_channels=64*3,
            out_channels=channels*3,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.motion_2=ME.MinkowskiConvolution(
            in_channels=64*3,
            out_channels=channels*3,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU()
        
        
    def feature_match(self,context1,context2):
        #context2要在context1中寻找最近邻的坐标，然后将最近邻特征提取出来，形成新的sparse_tensor
        
        N1, C = context1.C[:,1:].shape
        N2, _ = context2.C.shape
        
        xyz1 =  context1.C[:,1:].unsqueeze(0).float()
        xyz2 =  context2.C[:,1:].unsqueeze(0).float()
        # knn_idx = knn_point( 3, xyz1, xyz2 )
        dist, knn_idx, __ = knn_points(xyz2, xyz1, K=3)#xyz2在xyz1中搜索最近邻
        
        # print("xyz1,xyz2,knn_idx:",xyz1.shape,xyz2.shape,knn_idx.shape)
        # xyz1,xyz2,knn_idx: torch.Size([1, 76796, 3]) torch.Size([1, 77294, 3]) torch.Size([1, 77294, 3])

        grouped_xyz_norm = index_points_group(xyz1, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 
        
        grouped_f_ref = index_points_group(context1.F.unsqueeze(0), knn_idx)
        new_feature = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
        
        new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinate_map_key=context2.coordinate_map_key, coordinate_manager=context2.coordinate_manager)
        
        return new_context1  
        
    def forward(self,f1=None,f2=None,motion=None):
        
        new_f1_ref = self.feature_match(context1=f1,context2=f2)

        # f1 = self.conv1(f1)
        # context1 = self.res_block1(f1)
        
        f1_down1=self.conv_down1(f1)#参考特征的下采样 

        f2_down1 = self.res_down1(f2)#当前特征的下采样 
        motion_down1 = self.motion_down1(motion)#motion的下采样
        motion_down1 = self.motion_2(self.relu(motion_down1))
        # res_down1和motion_down1有相同的坐标
        # print("f1_down1.tensor_stride[0]",f1_down1.tensor_stride[0],f1.tensor_stride[0])
        xyz1,point1 = (f1_down1.C[:, 1:]/f1_down1.tensor_stride[0]).unsqueeze(0).to(torch.float32),f1_down1.F.unsqueeze(0).to(torch.float32)
        
        xyz2 = (f2_down1.C[:, 1:]/f2_down1.tensor_stride[0]).unsqueeze(0)
        
        # print("====>min max",torch.min(xyz1),torch.max(xyz1) , torch.min(xyz2),torch.max(xyz2))

        motion = motion_down1.F
        # B,N,C = point2_res.size()
        B,N,C =1 ,f2_down1.C.shape[0],f1_down1.F.shape[1]
        motion = motion.reshape(B,N,C,3)
        xyz2_  = (xyz2.unsqueeze(2) + motion).reshape(B, -1, 3)
        dist, knn_index1_, __ = knn_points(xyz2_, xyz1, K=3)#xyz2在xyz1中搜索最近邻
        dist += 1e-8
        knn_index1_ = knn_index1_.reshape(B, N, C, 3)
        knn_point1_ = index_by_channel(point1, knn_index1_, 3)
        dist = dist.reshape(B, N, C, 3)
        weights = 1 / dist
        weights = weights / torch.clamp(weights.sum(dim=3, keepdim=True), min=3)
        warped_point2 = (weights * knn_point1_).sum(dim=3).squeeze(0)
        warp_f2 = ME.SparseTensor(warped_point2, coordinates=f2_down1.C, coordinate_manager=f2_down1.coordinate_manager,tensor_stride=f2_down1.tensor_stride, device=f2_down1.device)
        
        context2 = warp_f2
        
        return context2,new_f1_ref
        
            
        
class Point_TemporalPriorEncoder(nn.Module):
    def __init__(self, channels=64 , block_layers=3 ):
        super().__init__()
     
        self.conv_down1 = ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3)
     
        self.conv_2= ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
        
        self.res_block= self.make_layer(InceptionResNet, block_layers, channels)
        
        self.relu = ME.MinkowskiReLU()


    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)
    
    def forward(self, context1):
        
        feature = self.relu(self.conv_down1(context1))
        
#         print("temporal 1 ===>context1_down:",sort_by_coor_sum(feature).C[1:10,...])
        
        feature = self.res_block(feature)
        # print("feature:",feature.shape)
        feature = self.conv_2(feature)
        
#         print("temporal 2 ===>context1_down:",sort_by_coor_sum(feature).C[1:10,...])
        
    
#         print("context1 经过1次下采样的坐标:",sort_by_coor_sum(feature).C[1:10,...])
        return feature

class Point_ContextualEncoder(nn.Module):
    def __init__(self, channels=64,channels_latent=8):
        super().__init__() 
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=channels*2,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        
        self.res1 = InceptionResNet(channels=channels)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels_latent,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        

    def forward(self, x, context1,scales=None):
        
        #将特征做concat
        if not torch.equal(x.C, context1.C):
            raise ValueError("Temporal context and hyper-prior must have the same coordinates")
        combined_features = torch.cat((x.F,context1.F), dim=1)

        # 创建新的稀疏张量
        combined_tensor = ME.SparseTensor(
            features=combined_features,
            coordinates=x.C,
            coordinate_manager=x.coordinate_manager,
            tensor_stride = x.tensor_stride,
            device=x.device
        )
        
        x_down1 = self.conv1(combined_tensor)
        x_down1 = self.res1(x_down1)
        
        x_down2 = self.conv2(x_down1)
        
    

        return x_down2


    
class Point_contextual_hyper_prior_encoder(nn.Module):
    def __init__(self, channels=64, channels_latent=96):
        super().__init__()
        self.conv1 =  ME.MinkowskiConvolution(
            in_channels=channels_latent,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv3 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
        self.leaky_relu = ME.MinkowskiLeakyReLU(inplace=True,negative_slope=0.1)
        # self.norm3 = ME.MinkowskiInstanceNorm(64)
        # self.norm3 = CustomMinkowskiLayerNorm(64)  # 使用自定义 LayerNorm
        self.norm3 = ME.MinkowskiBatchNorm(64)  # 使用自定义 LayerNorm
        self.scale = nn.Parameter(torch.ones(1) * 10.0)  # 添加可学习缩放

        
        # self.block = self.make_layer(
        #     InceptionResNet, block_layers=3, channels=channels)
        
    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("==============================")
        feature = self.leaky_relu(self.conv1(x))
        feature = self.leaky_relu(self.conv2(feature))
        feature = self.norm3(feature)
        feature = feature * self.scale
        
        feature = self.conv3(feature)
        # print("norm 正则")
        
        
        # print(f"After conv3: F min/max/mean/std: {feature.F.min().item():.4f}/"
        #              f"{feature.F.max().item():.4f}/{feature.F.mean().item():.4f}/"
        #              f"{feature.F.std().item():.4f}")

        return feature

    
class CustomMinkowskiLayerNorm(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(num_channels))
        self.beta = nn.Parameter(torch.zeros(num_channels))
    
    def forward(self, x):
        # x: ME.SparseTensor, 包含坐标 (C) 和特征 (F)
        features = x.F  # 形状 (N, num_channels)，N 为非零元素数
        coords = x.C    # 形状 (N, 4)，[:, 0] 为批次索引
        
        # 获取批次索引
        batch_indices = coords[:, 0].long()  # 形状 (N,), 转换为 torch.int64
        batch_size = batch_indices.max().item() + 1
        
        # 计算每个样本的点数
        point_counts = torch.zeros(batch_size, device=features.device, dtype=torch.float).scatter_add_(
            0, batch_indices, torch.ones_like(batch_indices, dtype=torch.float)
        )  # 形状 (batch_size,)
        # 计算均值：按批次索引聚合特征
        mean = torch.zeros(batch_size, self.num_channels, device=features.device)
        mean.scatter_add_(0, batch_indices.unsqueeze(1).expand(-1, self.num_channels), features)
        mean = mean / point_counts.unsqueeze(1).clamp(min=1)  # 形状 (batch_size, num_channels)
        
        # 计算方差：(x - mean)^2 的均值
        mean_per_point = mean[batch_indices]  # 形状 (N, num_channels)
        squared_diff = (features - mean_per_point) ** 2
        var = torch.zeros(batch_size, self.num_channels, device=features.device)
        var.scatter_add_(0, batch_indices.unsqueeze(1).expand(-1, self.num_channels), squared_diff)
        var = var / point_counts.unsqueeze(1).clamp(min=1)  # 形状 (batch_size, num_channels)
        
        # 归一化
        std = torch.sqrt(var + self.eps)  # 形状 (batch_size, num_channels)
        std_per_point = std[batch_indices]  # 形状 (N, num_channels)
        normalized_features = (features - mean_per_point) / std_per_point
        normalized_features = normalized_features * self.gamma + self.beta
        
        # 返回新的 SparseTensor
        return ME.SparseTensor(
            features=normalized_features,
            coordinates=coords,
            coordinate_manager=x.coordinate_manager,
            tensor_stride=x.tensor_stride
            
        )



class Point_contextual_hyper_prior_decoder(nn.Module):
    def __init__(self, channels=64 , block_layers=3):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.up =  ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=64,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, block_layers, 64)
        
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
        return nn.Sequential(*layers)

    def forward(self,z,ref=None):

        feature = self.relu(self.conv1(z))
        #得到了64维度的feature
        
        
        feature = self.relu(self.up(feature))

        feature = self.conv2(feature)
        
        # print("feature:",feature.shape)
        feature = self.block(feature)
        
        if ref is not None:
            mask = get_target_by_sp_tensor(feature, ref)
            feature = self.pruning(feature, mask.to(feature.device))

        return feature

class Point_contextual_entropy_parameter(nn.Module):
    def __init__(self, in_channels=64,channels=64):
        super().__init__()

        self.conv = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3),
            
            ME.MinkowskiReLU(inplace=True),
            
            ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3),
            
            ME.MinkowskiReLU(inplace=True),
            
            ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=32*2,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)

        )
    def forward(self,x):
        out = self.conv(x)

        return out

    
    
# class Point_TemporalPriorEncoder(nn.Module):
#     def __init__(self, channels=64):
#         super().__init__()
#         self.conv1 =  ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=2,
#             stride=2,
#             bias=True,
#             dimension=3)
        
       
#         self.conv2 =  ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
        
#         self.relu = ME.MinkowskiReLU(inplace=True)
  

#     def forward(self, context1):
        
#         feature = self.relu(self.conv1(context1))
#         feature = self.conv2(feature)
#         # print("feature:===>",feature.shape)
#         return feature
    

# class Point_TemporalPriorEncoder(nn.Module):
#     def __init__(self, channels=64 , block_layers=3 ):
#         super().__init__()
     
#         self.conv_down1 = ME.MinkowskiConvolution(
#                 in_channels=channels,
#                 out_channels=channels,
#                 kernel_size=2,
#                 stride=2,
#                 bias=True,
#                 dimension=3)
     
#         self.conv_2= ME.MinkowskiConvolution(
#                 in_channels=channels,
#                 out_channels=channels,
#                 kernel_size=3,
#                 stride=1,
#                 bias=True,
#                 dimension=3)
        
#         self.res_block= self.make_layer(InceptionResNet, block_layers, channels)
        
#         self.relu = ME.MinkowskiReLU()


#     def make_layer(self, block, block_layers, channels):
#         layers = []
#         for i in range(block_layers):
#             layers.append(block(channels=channels))

#         return nn.Sequential(*layers)
    
#     def forward(self, context1):
        
#         feature = self.relu(self.conv_down1(context1))
#         # print("feature:",feature.shape)
#         feature = self.conv_2(feature)
        
#         feature = self.res_block(feature)
  
#         return feature

    
class Context_UpsampleLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2):
        super(Context_UpsampleLayer, self).__init__()
        self.conv_context = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=hidden,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=input,
            out_channels=hidden,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        self.conv = ME.MinkowskiConvolution(
            in_channels=hidden,
            out_channels=output,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, block_layers, output)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels=output,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()
        
        
        self.context_block = self.make_layer(
            InceptionResNet, block_layers, output)

    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))
        return nn.Sequential(*layers)

    def get_cls(self, x):
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        out_cls = self.conv_cls(out)
        return out_cls

    def evaluate(self, x, adaptive, num_points, rho=1, residual=None, lossless=False):
        training = self.training
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        # if residual is not None:
        #     stride = out.tensor_stride[0]
        #     residual = ME.SparseTensor(residual.F, coordinates=residual.C,
        #                                coordinate_manager=out.coordinate_manager)
        #     out = out + residual
        #     out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=stride)
        out_cls = self.conv_cls(out)

        if adaptive:
            coords_nums = num_points
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                # print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, keep

    def forward(self, x,target_label, adaptive, rho=1, residual=None, lossless=False,context=None,f1_ref=None ):
        training = self.training
        
       
        # print("x:",x.shape)
        
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
        
        
        
        
        if context!=None:
            context = ME.SparseTensor(context.F, coordinates=context.C,
                                       coordinate_manager=out.coordinate_manager)
            # print("out.F,context.F:",out.F.shape,context.F.shape)
            stride = out.tensor_stride[0]
            fuse_context = out + context
            
            # print("diff:",torch.sum(fuse_context.F - out.F))
            
            # out = ME.SparseTensor( fuse_context, coordinates=out.C, coordinate_manager=out.coordinate_manager,tensor_stride=out.tensor_stride, device=out.device)
            
            out = ME.SparseTensor(fuse_context.F, coordinates=out.C, tensor_stride=stride)
            
            # print("new_context:",new_context.shape)
            out = self.context_block(out)
        
        if residual is not None:
            stride = out.tensor_stride[0]
            residual = ME.SparseTensor(residual.F, coordinates=residual.C,
                                       coordinate_manager=out.coordinate_manager)
            out = out + residual
            out = ME.SparseTensor(out.F, coordinates=out.C, tensor_stride=stride)
        out_cls = self.conv_cls(out)
        target = get_target_by_sp_tensor(out, target_label)

        if adaptive:
            coords_nums = [len(coords) for coords in target_label.decomposed_coordinates]
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0:
                # keep at least one points.
                # print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True

        # If training, force target shape generation, use net.eval() to disable
        if training or residual is not None:
            keep += target
        elif lossless:
            keep = target

        # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, target, keep