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
        
        print("xyz1,xyz2,knn_idx:",xyz1.shape,xyz2.shape,knn_idx.shape)
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
        
            
        

class Point_ContextualEncoder(nn.Module):
    def __init__(self, channels=64,out_channels=32 ,block_layers=3 ):
        super().__init__() 
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.down = ME.MinkowskiConvolution(
            in_channels=channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2,
            bias=True,
            dimension=3)
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=96,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=8,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        
        self.relu = ME.MinkowskiReLU()

        self.res_block = self.make_layer(block=InceptionResNet, block_layers=block_layers, channels=out_channels)
        
        self.enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=8, kernel_size=3, stride=1, bias=True, dimension=3)

    
    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)

    
    def forward(self,x,context2=None,f1_ref=None):
        #这里的x是残差好，还是不是残差好
        
        # print("enc x,f1_ref:",x.shape,f1_ref.shape)
        
        if f1_ref !=None:
            
            f1_ref = ME.SparseTensor( f1_ref.F, coordinates=x.C, coordinate_manager=x.coordinate_manager, tensor_stride=x.tensor_stride,device=x.device)
            x=x+f1_ref
        
        out = self.conv1(x)
        out = self.relu(out)
        out = self.down(out)
        out = self.res_block(self.relu(out))
        out = self.enc4(out)

        
        # if context2!=None:
#             print("step1===>:",out.F.shape,context2.F.shape)
            
#             print(out.C)
#             print(context2.C)
#             print("===>",torch.sum(out.C-context2.C))
            
#             new_feature = out + context2
#             new_feature = torch.concat([out.F,context2.F],dim=1)

#             out = ME.SparseTensor( new_feature, coordinates=out.C, coordinate_manager=out.coordinate_manager,tensor_stride=out.tensor_stride, device=out.device)

#             out = self.conv2(out)

#             out = self.res_block(self.relu(out))

#             out = self.conv3(out)
#         else:
            
            

        return out



# class Point_contextual_hyper_prior_encoder(nn.Module):
#     def __init__(self, channels=64, channel_M=96):
#         super().__init__()
#         self.conv1 =  ME.MinkowskiConvolution(
#             in_channels=8,
#             out_channels=channels,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
#         self.conv2 =  ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=channels,
#             kernel_size=2,
#             stride=2,
#             bias=True,
#             dimension=3)
#         self.conv3 =  ME.MinkowskiConvolution(
#             in_channels=channels,
#             out_channels=8,
#             kernel_size=3,
#             stride=1,
#             bias=True,
#             dimension=3)
        
#         self.relu = ME.MinkowskiReLU(inplace=True)

#     def forward(self, x,context):
#         if context!=None:
#             context = ME.SparseTensor(context, coordinates=context.C, coordinate_manager=x.coordinate_manager,
#                                        tensor_stride=x.stride, device=x.device)
            
#             x = x + context

#         feature = self.relu(self.conv1(x))
#         feature = self.relu(self.conv2(feature))

#         feature = self.conv3(feature)

#         return feature
    
class Point_contextual_hyper_prior_encoder(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 =  ME.MinkowskiConvolution(
            in_channels=8,
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
            out_channels=8,
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
    

class Point_contextual_hyper_prior_decoder(nn.Module):
    def __init__(self, channels=64 , block_layers=3):
        super().__init__()
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=8,
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
            out_channels=channels,
            kernel_size=3,
            stride=1,
            bias=True,
            dimension=3)
        self.block = self.make_layer(
            InceptionResNet, block_layers, channels)
        
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
                in_channels=in_channels,
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
                out_channels=8*2,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)

        )
    def forward(self,x):
        out = self.conv(x)

        return out

    
class Point_TemporalPriorEncoder(nn.Module):
    def __init__(self, channels=64 , block_layers=3 ):
        super().__init__()
        
        self.conv_fuse = ME.MinkowskiConvolution(
                in_channels=channels*2,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                bias=True,
                dimension=3)
        
        self.conv_down1 = ME.MinkowskiConvolution(
                in_channels=channels,
                out_channels=channels,
                kernel_size=2,
                stride=2,
                bias=True,
                dimension=3)
        self.conv_down2 = ME.MinkowskiConvolution(
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
    

    def forward(self, context1, context2,f1_ref):
        
        # print("context1, context2:",context1.shape, context2.shape)
        # context1为上一帧的f1,context2为才一个尺度估计的f2
        
        # feature1= self.conv_down1(context1)
        # print("prior:",context1.F.shape,f1_ref.F.shape)
        new_feature=torch.concat([context1.F,f1_ref.F],dim=1)
        feature = ME.SparseTensor( new_feature, coordinates=context1.C, coordinate_manager=context1.coordinate_manager,tensor_stride=context1.tensor_stride, device=context1.device)
        
        feature = self.relu(self.conv_fuse(feature))
        
        feature = self.relu(self.conv_down1(feature))
        # print("feature:",feature.shape)
        feature = self.conv_2(feature)
        
        feature = self.res_block(feature)
        
        #原来的超先验预测器
        
#         feature1 = self.conv_down1(context1)
        
#         new_feature = torch.concat([feature1.F,context2.F],dim=1)
        
#         feature = ME.SparseTensor( new_feature, coordinates=feature1.C, coordinate_manager=feature1..coordinate_manager,tensor_stride=feature1..tensor_stride, device=feature1..device)
        
#         feature = self.conv_3(feature)
        
#         feature = self.res_block(feature)
        
        return feature

    
class Context_UpsampleLayer(nn.Module):
    def __init__(self, input, hidden, output, block_layers, kernel=2):
        super(Context_UpsampleLayer, self).__init__()
        self.conv_context = ME.MinkowskiConvolution(
            in_channels=72,
            out_channels=input,
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

    def forward(self, x,target_label, adaptive, rho=1, residual=None, lossless=False,context=None, ):
        training = self.training
        
        if context!=None:
            
            cat_context = torch.concat([x.F,context.F],dim=1)
        
            new_context = ME.SparseTensor( cat_context, coordinates=x.C, coordinate_manager=x.coordinate_manager,tensor_stride=x.tensor_stride, device=x.device)
            
            # print("new_context:",new_context.shape)
            x = self.conv_context(new_context)
        # print("x:",x.shape)
        out = self.relu(self.conv(self.relu(self.up(x))))
        out = self.block(out)
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