import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.pointconv_util import*


class Point_FeatureExtractor(nn.Module):
    def __init__(self, input_channel=64,output_channel=64,kernel=2 , resnet=InceptionResNet):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=input_channel,
            out_channels=output_channel, 
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.res_block1 = InceptionResNet(channels=output_channel)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        self.res_block2 = InceptionResNet(channels=output_channel)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        self.res_block3 = InceptionResNet(channels=output_channel)

    def forward(self, feature):
        layer1 = self.conv1(feature)
        layer1 = self.res_block1(layer1)

        layer2 = self.conv2(layer1)
        layer2 = self.res_block2(layer2)

        layer3 = self.conv3(layer2)
        layer3 = self.res_block3(layer3)

        return layer1, layer2, layer3


class PointFlow_down(nn.Module):
    def __init__(self, input_channel=3,output_channel=3,kernel=2 , resnet=InceptionResNet):
        super().__init__()
        
        self.conv1_down = ME.MinkowskiConvolution(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        
        self.conv1_1 = ME.MinkowskiConvolution(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
     
        self.conv2_down = ME.MinkowskiConvolution(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel,
            stride=2,
            bias=True,
            dimension=3)
        
        self.conv2_2 = ME.MinkowskiConvolution(
            in_channels=output_channel,
            out_channels=output_channel,
            kernel_size=1,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, point_flow):
        
        point_flow2 = self.conv1_1(self.relu(self.conv1_down(point_flow)))

        point_flow3 = self.conv2_2(self.relu(self.conv2_down(point_flow2)))


        return point_flow2,point_flow3

class Point_ContextualEncoder(nn.Module):
    def __init__(self, channel=64):
        super().__init__() 
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channel,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.res1 = InceptionResNet(channels=channel)
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channel,
            out_channels=channel,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.res2 =  InceptionResNet(channels=channel)

        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channel,
            out_channels=channel,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.res3 =  InceptionResNet(channels=channel)

        self.conv4 = ME.MinkowskiConvolution(
            in_channels=channel,
            out_channels=channel,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
#     def feature_match(self,context1,context2):
#         #context2要在context1中寻找最近邻的坐标，然后将最近邻特征提取出来，形成新的sparse_tensor
        
#         N1, C = context1.C[:,1:].shape
#         N2, _ = context2.C.shape
        
#         xyz1 =  context1.C[:,1:].unsqueeze(0).float()
#         xyz2 =  context2.C[:,1:].unsqueeze(0).float()
#         knn_idx = knn_point( 3, xyz1, xyz2 )
        
#         grouped_xyz_norm = index_points_group(xyz1, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
#         dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
#         norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
#         weight = (1.0 / dist) / norm 
        
#         grouped_f_ref = index_points_group(context1.F.unsqueeze(0), knn_idx)
#         new_feature = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
        
#         new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinate_map_key=context2.coordinate_map_key, coordinate_manager=context2.coordinate_manager)
        
#         return new_context1   

    def forward(self, x, context1, context2, context3,scales=None):
        
        context1_new = feature_match(context1,x)
        x_fuse = x + context1_new
        feature_1 = self.conv1(x_fuse)#第一次下采样256->128
        feature_1 = self.res1(feature_1)
        # print("feature1:",feature_1.shape)
        
        context2_new =  feature_match(context2,feature_1)
        x_fuse = context2_new + feature_1
        feature_2 = self.conv2(x_fuse)#第二次下采样128->64
        feature_2 = self.res2(feature_2)
        # print("feature2:",feature_2.shape)
        
        context3_new =  feature_match(context3,feature_2)
        x_fuse = context3_new + feature_2
        feature_3 = self.conv3(x_fuse)#第三次下采样64->32
        feature_3 = self.res3(feature_3)
        # print("feature3:",feature_3.shape)
        feature_4 = self.conv4(feature_3)#第4次下采样32->16
        # print("feature4:",feature_4.shape)

        return feature_4,context1_new,context2_new,context3_new


class Point_ContextualDecoder(nn.Module):
    def __init__(self, channels_in=64, channel_out=64,resnet=InceptionResNet):
        super().__init__()
        self.up1 = DeconvWithPruning(input=channels_in, output=channel_out)
        self.res1 = InceptionResNet(channels=channel_out)
       
        self.up2 = DeconvWithPruning(input=channel_out, output=channel_out)
        self.res2= InceptionResNet(channels=channel_out)

        # self.up3 = DeconvWithPruning(input=channel_out, output=channel_out)
        # self.res3= InceptionResNet(channels=channel_out)

        self.up3 = UpsampleLayer(64, 64, 64, 3)

#最后一次上采样的输出的分辨率要是ys_cur[2]

#     def feature_match(self,context1,context2):
#         #context2要在context1中寻找最近邻的坐标，然后将最近邻特征提取出来，形成新的sparse_tensor
        
#         N1, C = context1.C[:,1:].shape
#         N2, _ = context2.C.shape
        
#         xyz1 =  context1.C[:,1:].unsqueeze(0).float()
#         xyz2 =  context2.C[:,1:].unsqueeze(0).float()
#         knn_idx = knn_point( 3, xyz1, xyz2 )
        
#         grouped_xyz_norm = index_points_group(xyz1, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
#         dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
#         norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
#         weight = (1.0 / dist) / norm 
        
#         grouped_f_ref = index_points_group(context1.F.unsqueeze(0), knn_idx)
#         new_feature = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
        
#         new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinate_map_key=context2.coordinate_map_key, coordinate_manager=context2.coordinate_manager)
        
#         return new_context1   


    def forward(self, x,target_label,adaptive,rho, context2, context3,scales=None):
        # print("x===>",x.shape,target_label.shape,context2.shape, context3.shape)
        # ===> torch.Size([185, 64]) torch.Size([3169, 64]) torch.Size([771, 64])
        feature = self.up1(x,context3)#第一次上采样16->32
        context3_new = feature_match(context3,x)
        feature_fuse= feature+context3_new
        feature = self.res1(feature_fuse)
        
        # print("up1:",feature.shape ,context3.shape)
        feature = self.up2(feature,context2)#第二次上采样32->64
        # print("feature:",feature.shape)
        # feature_concat = ME.SparseTensor(features=torch.cat([feature.F, context3.F], dim=-1), coordinates=feature.C)
        context2_new = feature_match(context2,feature)
        feature_fuse= feature+context2_new
        # print("feature_fuse:",feature_fuse.shape)
        feature = self.res2(feature_fuse)

        # feature = self.up3(feature,context2)#第三次上采样64->128
        # feature = ME.SparseTensor(features=torch.cat([feature.F, context2.F], dim=-1), coordinates=feature.C)
        # feature = self.res2(feature)

        #第4次上采样
        out_pruned, out_cls, target, keep = self.up3(feature,target_label,adaptive,rho)

        return out_pruned, out_cls, target, keep

class Point_contextual_hyper_prior_encoder(nn.Module):
    def __init__(self, channels=64, channel_M=96):
        super().__init__()
        self.conv1 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        self.conv2 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv3 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)

    def forward(self, x):
        feature = self.relu(self.conv1(x))
        feature = self.relu(self.conv2(feature))

        feature = self.conv3(feature)

        return feature
    
class Point_TemporalPriorEncoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        # self.gdn2 = GDN(channel_M)
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        # self.gdn3 = GDN(channel_M * 3 // 2)
        self.conv4 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.conv5 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU(inplace=True)
#     def feature_match(self,context1,context2):
#         #context2要在context1中寻找最近邻的坐标，然后将最近邻特征提取出来，形成新的sparse_tensor
        
#         N1, C = context1.C[:,1:].shape
#         N2, _ = context2.C.shape
        
#         xyz1 =  context1.C[:,1:].unsqueeze(0).float()
#         xyz2 =  context2.C[:,1:].unsqueeze(0).float()
#         knn_idx = knn_point( 3, xyz1, xyz2 )
        
#         grouped_xyz_norm = index_points_group(xyz1, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
#         dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
#         norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
#         weight = (1.0 / dist) / norm 
        
#         grouped_f_ref = index_points_group(context1.F.unsqueeze(0), knn_idx)
#         new_feature = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
        
#         # new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinate_map_key=context2.coordinate_map_key, coordinate_manager=context2.coordinate_manager)
#         new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinates=context2.C, tensor_stride=context2.tensor_stride)
        
#         return new_context1   

    def forward(self, context1, context2, context3):
        feature = self.conv1(context1)
        new_context2 = feature_match(context2,feature)
        
#         print("feature:" ,feature.C[0:20,:])
#         print("context2:",context2.C[0:20,:])
        
        feature = feature + new_context2
        feature = self.conv2(feature)
        
        new_context3 = feature_match(context3,feature)
        feature = feature + new_context3
        feature = self.conv3(feature)
        
        feature = self.relu(self.conv4(feature))
        feature = self.conv5(feature)
        # print("feature:===>",feature.shape)
        return feature


class Point_contextual_hyper_prior_decoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.up = ME.MinkowskiGenerativeConvolutionTranspose(
            in_channels=channels,
            out_channels=channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.conv1 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.conv2 =  ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

    def forward(self,z,ref=None):
        feature = self.relu(self.up(z))
        feature = self.relu(self.conv1(feature))

        feature = self.conv2(feature)
        
        if ref is not None:
            mask = get_target_by_sp_tensor(feature, ref)
            feature = self.pruning(feature, mask.to(feature.device))

        return feature



class Point_contextual_entropy_parameter(nn.Module):
    def __init__(self, channels=64):
        super().__init__()

        self.conv = nn.Sequential(
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
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3),
            
            ME.MinkowskiReLU(inplace=True),
            
            ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels*2,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)

        )
    def forward(self,x):
        out = self.conv(x)

        return out



class Point_MultiScaleContextFusion(nn.Module):
    def __init__(self, channel_in=64, channel_out=64):
        super().__init__()
        self.conv3_up = DeconvWithPruning(channel_in, channel_out)#上采样2倍
        self.res_block3_up =InceptionResNet(channels=channel_out)
        self.conv3_out = ME.MinkowskiConvolution(
            in_channels=channel_out,
            out_channels=channel_out,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.res_block3_out = InceptionResNet(channels=channel_out)

        self.conv2_up = DeconvWithPruning(input=64, output=64)
        self.res_block2_up = InceptionResNet(channels=channel_out)
        self.conv2_out = ME.MinkowskiConvolution(
            in_channels=channel_out,
            out_channels=channel_out,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.res_block2_out = InceptionResNet(channels=channel_out)

        self.conv1_out = ME.MinkowskiConvolution(
            in_channels=channel_out,
            out_channels=channel_out,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.res_block1_out = InceptionResNet(channels=channel_out)
    
#     def feature_match(self,context1,context2):
#         #context2要在context1中寻找最近邻的坐标，然后将最近邻特征提取出来，形成新的sparse_tensor
        
#         N1, C = context1.C[:,1:].shape
#         N2, _ = context2.C.shape
        
#         xyz1 =  context1.C[:,1:].unsqueeze(0).float()
#         xyz2 =  context2.C[:,1:].unsqueeze(0).float()
#         knn_idx = knn_point( 3, xyz1, xyz2 )
        
#         grouped_xyz_norm = index_points_group(xyz1, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
#         dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
#         norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
#         weight = (1.0 / dist) / norm 
        
#         grouped_f_ref = index_points_group(context1.F.unsqueeze(0), knn_idx)
#         new_feature = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
        
#         new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinate_map_key=context2.coordinate_map_key, coordinate_manager=context2.coordinate_manager)

#         # new_context1 =  ME.SparseTensor(new_feature.squeeze(),coordinates=context2.C, tensor_stride=context2.tensor_stride)
        
#         return new_context1
                                                 
        
    
    def forward(self, context1, context2, context3):
        context3_up = self.conv3_up(context3,context2)
        # print("====>context3_up:",context3_up.shape)
        context3_up = self.res_block3_up(context3_up)
        context3_out = self.conv3_out(context3)
        context3_out = self.res_block3_out(context3_out)
        # print("====>context3_out:",context3_out.shape)
        
        # print("context3_up + context2:",context3_up.shape , context2.shape)
        # print("context3_up.F, context2.F:",context3_up.F.shape, context2.F.shape)
        context_tmp =feature_match(context3_up,context2)
        context2_fuse = context_tmp + context2
             
        # print("context_concat:",context2_fuse.shape)
        # print("context2===>:",context2.shape)
        context2_up = self.conv2_up(context2_fuse,context1)
        context2_up = self.res_block2_up(context2_up)
        
        context2_out = self.conv2_out(context2_fuse)
        context2_out = self.res_block2_out(context2_out)
        
        
        context_tmp =feature_match(context2_up,context1)
        context1_fuse = context_tmp + context1
        context1_out = self.conv1_out(context1_fuse)
        context1_out = self.res_block1_out(context1_out)

        context1 = context1 + context1_out
        context2 = context2 + context2_out
        context3 = context3 + context3_out
        
        
        # print("context1,context2,context3",context1.shape,context2.shape,context3.shape)
        return context1, context2, context3


class Point_Recongeneration_for_context1(nn.Module):
    def __init__(self, ctx_channel=64):
        super().__init__()
        self.conv = ME.MinkowskiConvolution(
            in_channels=ctx_channel,
            out_channels=ctx_channel,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, 2, ctx_channel)
        self.conv_cls = ME.MinkowskiConvolution(
            in_channels=ctx_channel,
            out_channels=1,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.pruning = ME.MinkowskiPruning()
        self.relu = ME.MinkowskiReLU()

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

    def forward(self, x, target_label, adaptive, rho=1, residual=None,training=False, lossless=False):
    
        out = self.relu(self.conv(x))
        out = self.block(out)

        out_cls = self.conv_cls(out)
        target = get_target_by_sp_tensor(out, target_label)

        if adaptive:
            coords_nums = [len(coords) for coords in target_label.decomposed_coordinates]
            keep = keep_adaptive(out_cls, coords_nums, rho=rho)
        else:
            keep = (out_cls.F > 0).squeeze()
            if out_cls.F.max() < 0: 
                # keep at least one points.
                print('===0; max value < 0', out_cls.F.max())
                _, idx = torch.topk(out_cls.F.squeeze(), 1)
                keep[idx] = True
        if training:
            keep += target
        elif lossless:
            keep = target
         # Remove voxels
        out_pruned = self.pruning(out, keep.to(out.device))
        return out_pruned, out_cls, target, keep