import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.prior_module import*
from models.flow_loss import*

from models.pointconv_util import*



import sys
sys.path.append('../')
from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow

from PointPWC.models import multiScaleChamferSmoothCurvature

from entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from entropy_models.video_entropy_models import EntropyCoder

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

        self.up3 = UpsampleLayer(64, 64, 64, 3)




    def forward(self, x,target_label,adaptive,rho, context2, context3,scales=None):
        
        feature = self.up1(x,context3)#第一次上采样16->32
        context3_new = feature_match(context3,x)
        feature_fuse= feature+context3_new
        feature = self.res1(feature_fuse)
        
        feature = self.up2(feature,context2)#第二次上采样32->64
        context2_new = feature_match(context2,feature)
        feature_fuse= feature+context2_new
        feature = self.res2(feature_fuse)

        #第4次上采样
        out_pruned, out_cls, target, keep = self.up3(feature,target_label,adaptive,rho)

        return out_pruned, out_cls, target, keep




class get_model(nn.Module):
    def __init__(self, channels=8 ,training=False):
        super(get_model, self).__init__()
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3) 
        
        #点云光流网络相关
        self.optical_point_flow = PointConvSceneFlow()

        #分类_loss
        self.crit_bce = torch.nn.BCEWithLogitsLoss()
        
        
        self.context_encoder = DownsampleLayer(64, 64, 32, 3)
        self.context_enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)

        # self.BitEstimator = BitEstimator(channels, 3)
        # self.MotionBitEstimator = BitEstimator(48, 3)
        
        self.entropy_coder = None

        self.bit_estimator_z = BitEstimator(channel=64)
        self.bit_estimator_z_mv = BitEstimator(channel=32)
        self.gaussian_encoder = GaussianEncoder()

        #点云mv的编解码网络
        self.mv_encoder       = MV_DownsampleLayer(input=3, hidden=32, output=32, block_layers=0,resnet=None)# input, hidden, output, block_layers
        self.mv_prior_encoder = DownsampleLayer(input=32, hidden=32, output=32, block_layers=0,resnet=None)
        
        self.mv_prior_decoder = DeconvWithPruning(input=32, output=64)
        self.mv_decoder       = MV_DeconvWithPruning(input=32, output=3)
        #结束点云的编解码

        #下面网络模块用于提取上下文信息
        self.feature_extractor = ME.MinkowskiConvolution(in_channels=64, out_channels=64, kernel_size=3, stride=1,
                                                 bias=True,
                                                 dimension=3)
        self.multi_scale_feature_extractor =Point_FeatureExtractor(input_channel=64,output_channel=64)
        self.point_down = PointFlow_down(input_channel=3,output_channel=3)
        self.context_fusion_net = Point_MultiScaleContextFusion(channel_in=64, channel_out=64)

        #上下文编码点云的特征模块
        self.contextual_encoder =Point_ContextualEncoder()
        self.contextual_hyper_prior_encoder =Point_contextual_hyper_prior_encoder() 
        self.temporal_prior_encoder = Point_TemporalPriorEncoder()
        self.contextual_hyper_prior_decoder =Point_contextual_hyper_prior_decoder() 
        self.contextual_decoder = Point_ContextualDecoder()
        
        # self.dec1 = UpsampleLayer(channels, 64, 64, 3)
        self.dec2 = UpsampleLayer(64, 32, 32, 3)
        self.dec3 = UpsampleLayer(32, 16, 16, 3)

        #直接从context1恢复出重构点云
        self.context1_to_point = Point_Recongeneration_for_context1(ctx_channel=64)

        #上下文熵编码模块
        self.contextual_entropy_parameter=Point_contextual_entropy_parameter()

        self.training = training
        
        
        self.down_C=ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, bias=True, dimension=3)
    
    def flow_warp(self, f_ref, point_flow):
        
        f_warp = f_ref+point_flow
        
        return f_warp
    
   

    def pointWarping_feat(self,f_ref,f_cur_C,point_flow):
        
     
        xyz1_to_2 =f_ref.C[:,1:] + point_flow.F
        
        
        N1, C = f_ref.C[:,1:].shape
        N2, _ = f_cur_C.shape
        
        xyz1_to_2 = xyz1_to_2.unsqueeze(0).float()
        # xyz2      = f_cur.C[:,1:].unsqueeze(0).float()
        xyz2      = f_cur_C.unsqueeze(0).float()
        knn_idx = knn_point( 3, xyz1_to_2, xyz2 )
        grouped_xyz_norm = index_points_group(xyz1_to_2, knn_idx) - xyz2.view(1, N2, 1, C) # B N2 3 C
        dist = torch.norm(grouped_xyz_norm, dim = 3).clamp(min = 1e-10)
        norm = torch.sum(1.0 / dist, dim = 2, keepdim = True)
        weight = (1.0 / dist) / norm 
        
        grouped_f_ref = index_points_group(f_ref.F.unsqueeze(0), knn_idx)
        
        warp_f_ref = torch.sum(weight.view(1, N2, 3, 1) * grouped_f_ref, dim = 2)
      
        
        return warp_f_ref
        
    
        
    def motion_compensation(self, f_ref,f_cur_C, point_flow):
        f_ref_warp = self.flow_warp(f_ref.C[:,1:] , point_flow.F)
        
        ref_feature = self.feature_extractor(f_ref)#对参考特征做一次特征提取，此处是为了兼容多帧参考
        

        #对ref_feature做多尺度的下采样

        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref_feature)
        #原始尺度，2被才采样尺度，4倍下采样尺度
      
        #对点云的point flow 采样3个尺度，做多尺度的特征提取
        point_flow2 , point_flow3 = self.point_down(point_flow)


        context1 = self.pointWarping_feat(ref_feature1,f_cur_C[:,1:], point_flow)

        Feature = torch.ones_like(f_cur_C[:, :1].float())
        tensor_temp = ME.SparseTensor( Feature , coordinates=f_cur_C,tensor_stride=f_ref.tensor_stride,device=f_ref.device)
        f_cur2_temp =self.down_C(tensor_temp)

      
        context2 = self.pointWarping_feat(ref_feature2,f_cur2_temp.C[:,1:], point_flow2)
        
        Feature = torch.ones_like(f_cur2_temp.C[:, :1].float())
        tensor_temp = ME.SparseTensor( Feature,coordinates=f_cur2_temp.C,tensor_stride=f_cur2_temp.tensor_stride , device=f_ref.device)
        f_cur3 =self.down_C(tensor_temp)
        
        
        context3 = self.pointWarping_feat(ref_feature3,f_cur3.C[:,1:], point_flow3)
        
        
        
        context1_sparse =  ME.SparseTensor(features=context1[0], coordinates=f_cur_C,tensor_stride=f_ref.tensor_stride, device=f_ref.device)
        context2_sparse =  ME.SparseTensor(features=context2[0], coordinates=f_cur2_temp.C,tensor_stride=f_cur2_temp.tensor_stride, device=f_cur2_temp.device)
        context3_sparse =  ME.SparseTensor(features=context3[0], coordinates=f_cur3.C,tensor_stride=f_cur3.tensor_stride, device=f_cur3.device)

        context1, context2, context3 = self.context_fusion_net(context1_sparse, context2_sparse, context3_sparse)

        return context1, context2, context3, f_ref_warp


    def prune(self, f1, f2):
        mask = get_target_by_sp_tensor(f1, f2)
        out = self.pruning(f1, mask.to(f1.device))
        return out
 
    
    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob
    
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
    
    
    
    def forward(self,f_ref, f_cur,coding_mode, device, epoch=99999):

        num_points = f_cur.C.size(0)
        ys_ref, ys_cur = [f_ref, 0, 0, 0, 0], [f_cur, 0, 0, 0, 0]

        out2, out_cls2, targets2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

         # feature extraction
        #第一帧两次下采样
        ys_ref[1] = self.enc1(ys_ref[0])
        ys_ref[2] = self.enc2(ys_ref[1])
        #第二帧两次下采样
        ys_cur[1] = self.enc1(ys_cur[0])
        ys_cur[2] = self.enc2(ys_cur[1])
        
        
        point_mvs,fps_pc1_idxs, _, pc1, pc2 = self.optical_point_flow(ys_ref[2].C[:,1:].unsqueeze(0).to(torch.float32),ys_cur[2].C[:,1:].unsqueeze(0).to(torch.float32),\
                                     ys_ref[2].C[:,1:].unsqueeze(0).to(torch.float32),ys_cur[2].C[:,1:].unsqueeze(0).to(torch.float32))#[Nx3]
        
        # 根据coordinate以及光流网络点云的初步光流
        point_mv_F = point_mvs[0][0].transpose(0,1)
        #将点云光流转为稀疏tensor
        point_mv = ME.SparseTensor(point_mv_F, coordinates=ys_ref[2].C, tensor_stride=ys_ref[2].tensor_stride)
        #得到motion之后，可以指导feature生成context
        #开始编码点云的mv
        mv_y = self.mv_encoder(point_mv) #对mv做2次稀疏下采样
        mv_z = self.mv_prior_encoder(mv_y)#对mv_y做了一次下采样
       
        mv_z_q=ME.SparseTensor(quant(mv_z.F, training=self.training),
                                                  coordinate_map_key=mv_z.coordinate_map_key,
                                                  coordinate_manager=mv_z.coordinate_manager)#量化光流特征

        mv_params = self.mv_prior_decoder(mv_z_q , mv_y )##对prior做一次上采样，并输出用于估计scale和mean，mv_y是否可以直接用，存在疑问
        mv_params_new = feature_match_new(mv_params,mv_y.C)
        

        mv_scales_hat, mv_means_hat = mv_params_new.chunk(2, 1)
       
        mv_y_res = mv_y.F - mv_means_hat #稀疏卷积之间的减法 
  
        mv_y_q   = quant(mv_y_res, training=self.training)
        mv_y_hat = mv_y_q + mv_means_hat
        
        
        mv_y_hat = ME.SparseTensor( mv_y_hat,coordinates=mv_y.C,tensor_stride=mv_y.tensor_stride) 
                                                                                              
        mv_hat = self.mv_decoder(mv_y_hat,point_mv)#对特征做2次上采样得到重构光流
        


        if coding_mode =='align':
            f_ref_warp = self.flow_warp(ys_ref[2].C[:,1:] , mv_hat.F)#这里就是编码光流，将f_ref做warp，和目标点云做loss。
            

            loss_flow,chamfer_loss, curvature_loss, smoothness_loss = multiScaleChamferSmoothCurvature(pc1,pc2,point_mvs,f_ref_warp)
            #这里是无监督的光流loss

            total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_q, mv_scales_hat)
            total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_q.F, self.bit_estimator_z_mv)
            bpp_mv_y = total_bits_mv_y / num_points
            bpp_mv_z = total_bits_mv_z / num_points
            
            
            mv_bpp = bpp_mv_y + bpp_mv_z
                       
            return{
                "bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_mv":   mv_bpp,
                "loss_flow":loss_flow/num_points,
            }
        

        context1, context2, context3, warp_frame= self.motion_compensation(ys_ref[2], ys_cur[2].C, mv_hat)
        #下采样域的参考帧和光流，生成多尺度的上下文信息
        
        if coding_mode =='align2':
            #这里的训练和测试是有区别的
            out_pruned, out_cls, target, keep = self.context1_to_point(context1,target_label=ys_cur[2],adaptive=1,rho=1,training=self.training , lossless=None

            loss_bce = self.crit(out_cls.F.squeeze(),target.type(out_cls.F.dtype).to(device))

            total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_q, mv_scales_hat)
            total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_q.F, self.bit_estimator_z_mv)
            bpp_mv_y = total_bits_mv_y / num_points
            bpp_mv_z = total_bits_mv_z / num_points
            mv_bpp   = bpp_mv_y + bpp_mv_z
            
            return{
                "bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_mv":mv_bpp,
                "loss_bce":loss_bce,
            }


     
        y , context1_new,context2_new,context3_new = self.contextual_encoder(ys_cur[2],context1, context2, context3)
    
        z = self.contextual_hyper_prior_encoder(y)
   
        z_q  = ME.SparseTensor(quant(z.F, training=self.training),
                                                  coordinate_map_key=z.coordinate_map_key,
                                                  coordinate_manager=z.coordinate_manager)
        z_hat = z_q
        hierarchical_params     = self.contextual_hyper_prior_decoder(z_hat , y)#第一个疑问点，y在解码端可不可以得到
        hierarchical_params_new = feature_match_new(hierarchical_params,y.C)
        
        
        temporal_params = self.temporal_prior_encoder(context1,context2,context3)
        
        
        params_feature = temporal_params.F + hierarchical_params_new
        params = ME.SparseTensor(params_feature,coordinates=y.C,tensor_stride=y.tensor_stride)
        
        
        
        gaussian_params = self.contextual_entropy_parameter(params)
        
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        
        y_res = y.F - means_hat
        y_q =  quant(y_res , training=self.training)
        y_hat = y_q + means_hat

        y2_recon = ME.SparseTensor(features=y_hat,coordinates=y.C,tensor_stride=y.tensor_stride,device=y.device)

     
        out2[0], out_cls2[0], targets2[0], keep2[0] =  self.contextual_decoder(x=y2_recon, target_label=ys_cur[2], adaptive=True,rho=1,context2=context2,context3=context3)
        
        out2[1], out_cls2[1], targets2[1], keep2[1] = self.dec2(out2[0], ys_cur[1], True, 1 if self.training else 1)
        out2[2], out_cls2[2], targets2[2], keep2[2] = self.dec3(out2[1], ys_cur[0], True, 1 if self.training else 1)

        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_q.F
        mv_z_for_bit = mv_z_q.F
        total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
        total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_for_bit, mv_scales_hat)
        total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
        total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_for_bit, self.bit_estimator_z_mv)
        bpp_y = total_bits_y / num_points
        bpp_z = total_bits_z / num_points
        bpp_mv_y = total_bits_mv_y / num_points
        bpp_mv_z = total_bits_mv_z / num_points
        
        total_bits = total_bits_y + total_bits_z + total_bits_mv_y + total_bits_mv_z
        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z
        distortion = 0
        for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
            curr_loss = self.crit_bce(out_cl.F.squeeze(),target.type(out_cl.F.dtype).to(device))                               
            distortion += curr_loss / num_points
        
        loss_d = distortion
        
        recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C,tensor_stride=out2[-1].tensor_stride,device=out2[-1].device )
        
        return {
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "total_bit":total_bits ,
            "mv": mv_hat,
            "out":out2,
            "out_cls2":out_cls2,
            "targets2":targets2,
            'loss_d':loss_d,
            'recon_point':recon_f2
        }


