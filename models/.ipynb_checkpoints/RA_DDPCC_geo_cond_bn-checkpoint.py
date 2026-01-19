import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.cond_utils import *

from models.pointconv_util import*
from GPCC.gpcc_wrapper import *
from models.PTFeaturePredictor import PointCloudFeaturePredictor

from models.resnet import ResNet, InceptionResNet

class get_model(nn.Module):
    def __init__(self, channels=8,entropy_mode='context_cond',knn_fixed=True,point_conv =False,fuse_conv=False):
        super(get_model, self).__init__()
        
        self.gop_size=16
        self.buffer = {}
        self.ra_order = [0, 8, 4, 2, 6, 1, 3, 5, 7, 12, 10, 14, 9, 11, 13, 15]
        
        self.entropy_mode =entropy_mode
        self.knn_fixed = knn_fixed
        self.point_conv = point_conv
        self.fuse_conv = fuse_conv
        
        
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3)
        # self.inter_prediction = inter_prediction(64, 64, 48)
        
        if self.fuse_conv:
            self.fuse_conv_2 = self.make_layer(InceptionResNet, 2, 64)
        
        if self.knn_fixed:
            self.inter_prediction_v2 = PointCloudFeaturePredictor(in_channels=64, K=8,point_conv=self.point_conv)
        else:
            self.inter_prediction = inter_prediction(64, 64, 48)
        
        
        if self.entropy_mode=='ori':
        
            self.enc3 = DownsampleLayer(64, 64, 32, 3)
            self.enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)

            self.dec1 = UpsampleLayer(channels, 64, 64, 3)
            self.dec2 = UpsampleLayer(64, 32, 32, 3)
            self.dec3 = UpsampleLayer(32, 16, 16, 3)

            self.BitEstimator = BitEstimator(channels, 3)
            self.MotionBitEstimator = BitEstimator(48, 3)
            self.crit = torch.nn.BCEWithLogitsLoss()
        
        if self.entropy_mode =="context_cond":
            
            
            self.bit_estimator_z = BitEstimator(channel=64,dimension=3)
            
            # self.context_produce = Point_context_produce(channels=64)
            
            self.contextual_encoder_v2 = Point_ContextualEncoder(channels=64,channels_latent=channels )
            self.contextual_hyper_prior_encoder = Point_contextual_hyper_prior_encoder(channels=64,channels_latent=channels)
            self.contextual_hyper_prior_decoder =Point_contextual_hyper_prior_decoder(channels=64 , block_layers=3)
            
            self.temporal_prior_encoder = Point_TemporalPriorEncoder(channels=64,block_layers=3)
            
            self.contextual_entropy_parameter = Point_contextual_entropy_parameter(in_channels=128,channels=64)
            # self.dec1 = Context_UpsampleLayer(8, 64, 64, 3)
            self.dec1_v2 = UpsampleLayer(channels, 64, 64, 3)
            
            self.dec2 = UpsampleLayer(64, 32, 32, 3)
            self.dec3 = UpsampleLayer(32, 16, 16, 3)

            # self.BitEstimator = BitEstimator(channels, 3)
            # self.MotionBitEstimator = BitEstimator(48, 3)
            self.crit = torch.nn.BCEWithLogitsLoss()
            
            
#             self.bit_estimator_z = BitEstimator(channel=8,dimension=3)
            
#             self.BitEstimator = BitEstimator(channels, 3)
#             self.MotionBitEstimator = BitEstimator(48, 3)
#             self.crit = torch.nn.BCEWithLogitsLoss()
            
#             self.context_produce = Point_context_produce(channels=64)
            
#             self.contextual_encoder = Point_ContextualEncoder(channels=64 )
#             self.contextual_hyper_prior_encoder = Point_contextual_hyper_prior_encoder(channels=64)
#             self.contextual_hyper_prior_decoder =Point_contextual_hyper_prior_decoder(channels=64 , block_layers=3)
            
#             self.temporal_prior_encoder = Point_TemporalPriorEncoder(channels=64,block_layers=3)
            
#             self.contextual_entropy_parameter = Point_contextual_entropy_parameter(in_channels=128,channels=64)
#             # self.dec1 = Context_UpsampleLayer(8, 64, 64, 3)
#             self.dec1 = UpsampleLayer(channels, 64, 64, 3)
#             self.dec2 = UpsampleLayer(64, 32, 32, 3)
#             self.dec3 = UpsampleLayer(32, 16, 16, 3)
            
#             self.cond = True
    
    
    
    def make_layer(self, block, block_layers, channels):
        layers = []
        for i in range(block_layers):
            layers.append(block(channels=channels))

        return nn.Sequential(*layers)
        
    def reset_buffer(self):
        self.buffer = {}

    def update_buffer(self, reconstructed, video_idx, frame_idx):
        self.buffer[(video_idx, frame_idx)] = reconstructed.detach()   
        # self.buffer[(video_idx, frame_idx)] = reconstructed

        gop_start = (frame_idx // self.gop_size) * self.gop_size
        if frame_idx == gop_start + self.gop_size - 1:  # 当前帧是 GOP 的最后一帧
            self.clean_buffer(video_idx, gop_start)
            
    def clean_buffer(self, video_idx, current_gop_start):
        # 只保留当前 GOP 和前 GOP 的最后一帧
        prev_gop_start = current_gop_start - self.gop_size
        prev_gop_last = prev_gop_start + self.gop_size - 1
        keys_to_keep = [(video_idx, prev_gop_last)] + \
                       [(video_idx, current_gop_start + i) for i in range(self.gop_size)]
        # 删除不需要的键
        self.buffer = {k: v for k, v in self.buffer.items() if k in keys_to_keep}

    
    def get_reference_frames(self, video_idx, frame_idx, gop_start):
        rel_idx = frame_idx - gop_start
        if rel_idx == 0:  # I帧
            if gop_start == 0:
                return None, None
            else:
                prev_gop_last = gop_start - 1
                return self.buffer.get((video_idx, prev_gop_last)), None
        elif rel_idx == 8: return self.buffer.get((video_idx, gop_start + 0)), None
        elif rel_idx == 4: return self.buffer.get((video_idx, gop_start + 0)), self.buffer.get((video_idx, gop_start + 8))
        elif rel_idx == 2: return self.buffer.get((video_idx, gop_start + 0)), self.buffer.get((video_idx, gop_start + 4))
        elif rel_idx == 6: return self.buffer.get((video_idx, gop_start + 4)), self.buffer.get((video_idx, gop_start + 8))
        elif rel_idx == 1: return self.buffer.get((video_idx, gop_start + 0)), self.buffer.get((video_idx, gop_start + 2))
        elif rel_idx == 3: return self.buffer.get((video_idx, gop_start + 2)), self.buffer.get((video_idx, gop_start + 4))
        elif rel_idx == 5: return self.buffer.get((video_idx, gop_start + 4)), self.buffer.get((video_idx, gop_start + 6))
        elif rel_idx == 7: return self.buffer.get((video_idx, gop_start + 6)), self.buffer.get((video_idx, gop_start + 8))
        elif rel_idx == 12: return self.buffer.get((video_idx, gop_start + 8)), None
        elif rel_idx == 10: return self.buffer.get((video_idx, gop_start + 8)), self.buffer.get((video_idx, gop_start + 12))
        elif rel_idx == 14: return self.buffer.get((video_idx, gop_start + 12)), None
        elif rel_idx == 9: return self.buffer.get((video_idx, gop_start + 8)), self.buffer.get((video_idx, gop_start + 10))
        elif rel_idx == 11: return self.buffer.get((video_idx, gop_start + 10)), self.buffer.get((video_idx, gop_start + 12))
        elif rel_idx == 13: return self.buffer.get((video_idx, gop_start + 12)), self.buffer.get((video_idx, gop_start + 14))
        elif rel_idx == 15: return self.buffer.get((video_idx, gop_start + 14)), self.buffer.get((video_idx, gop_start + 12))
        return None, None
    
    def feature_add(self,residual_1, residual_2, target_tensor):
    # """
    #     将 residual_1 和 residual_2 插值到 target_tensor 的坐标系并相加。
    #     residual_1, residual_2, target_tensor: ME.SparseTensor
    #     """
        # 目标坐标（当前帧的坐标）
        target_coords = target_tensor.coordinates

        # 将 residual_1 和 residual_2 插值到 target_coords
        # 使用 ME.SparseTensor 的 decompose 方法获取特征和坐标
        res1_feats = residual_1.features_at_coordinates(target_coords.float())  # [N_target, C]
        res2_feats = residual_2.features_at_coordinates(target_coords.float())  # [N_target, C]

        # 如果某些点在 residual_1 或 residual_2 中不存在，返回零填充
        fused_feats = (res1_feats + res2_feats)/2  # [N_target, C]
        
        return fused_feats

        # # 创建新的稀疏张量
        # fused_residual = ME.SparseTensor(
        #     features=fused_feats,
        #     coordinates=target_coords,
        #     device=residual_1.device
        # )
        # return fused_residual
    
    def forward(self, current_f,video_idx, frame_idx, device, epoch=99999):
        
        # print("self.training===:",self.training)
        
        # video_idx, frame_idx = indices
        
        # print("video_idx, frame_idx:",video_idx, frame_idx)
        
        gop_start = (frame_idx // self.gop_size) * self.gop_size
        # gop_start = frame_idx
        # print("gop_start:",gop_start)
        
        ref1, ref2 = self.get_reference_frames(video_idx, frame_idx, gop_start)

        num_points = current_f.C.size(0)#获得点云的点数
        
        cur_f = [current_f, 0, 0, 0, 0 ]
        
        ys_ref1 =[ref1, 0, 0, 0, 0]
        ys_ref2 =[ref2, 0, 0, 0, 0]

        # ys1, ys2 = [f1, 0, 0, 0, 0], [f2, 0, 0, 0, 0]

        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
        
        
        if ref1 is None and ref2 is None:  # I帧
            # print("I frame")
            out2[2]=current_f
            self.update_buffer(out2[2], video_idx, frame_idx)
            bpp=0
            
            return cur_f, out2, out_cls2, target2, keep2, bpp 
            
        elif ref2 is None:  # P帧或B14
            # print("P frame ...")
             #第一帧两次下采样 
            ys_ref1[1] = self.enc1(ys_ref1[0])
            ys_ref1[2] = self.enc2(ys_ref1[1])
            #第二帧两次下采样
            cur_f[1] = self.enc1(cur_f[0])
            cur_f[2] = self.enc2(cur_f[1])
            
            if self.knn_fixed:
                predicted_point2, residual ,_= self.inter_prediction_v2(ys_ref1[2], cur_f[2])
                motion_bits=0
            else:
        
                residual, predicted_point2, quant_motion ,motion_hat = self.inter_prediction(ys_ref1[2], cur_f[2], stride=4)
                quant_motion_F = quant_motion.F.unsqueeze(0)#1*N*64*3
                motion_p = self.MotionBitEstimator(quant_motion_F+0.5) - self.MotionBitEstimator(quant_motion_F-0.5)
                motion_bits = torch.sum(torch.clamp(-1.0 * torch.log(motion_p + 1e-10) / math.log(2.0), 0, 50))
                motion_bits =  motion_bits
            
            
            if self.entropy_mode=="ori":
                cur_f[3] = self.enc3(residual)
                cur_f[4] = self.enc4(cur_f[3])

                cur_f[4]=sort_by_coor_sum(cur_f[4])

                quant_y = quant(cur_f[4].F.unsqueeze(0), training=self.training)
                # 这里有了要传输的两部分，分别是quant_motion_F和quant_y
                # bit rate calculation
                p = self.BitEstimator(quant_y+0.5) - self.BitEstimator(quant_y-0.5)
                bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
                y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=cur_f[4].coordinate_map_key,coordinate_manager=cur_f[4].coordinate_manager, device=cur_f[4].device)
                bpp = (bits + motion_bits) / num_points
                out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, cur_f[2], True, residual=predicted_point2)



                out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], cur_f[1], True, 1 if self.training else 1)
                out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], cur_f[0], True, 1 if self.training else 1)

                recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)

                # self.update_buffer(recon_f2, video_idx, frame_idx)
                self.update_buffer(current_f, video_idx, frame_idx)

                return cur_f, out2, out_cls2, target2, keep2, bpp 
            
            elif self.entropy_mode=="context_cond":
                
                context1 = predicted_point2 
                cur_f[4] = self.contextual_encoder_v2(cur_f[2],context1)
                
                z = self.contextual_hyper_prior_encoder(cur_f[4])#下采样一次
                z_q = ME.SparseTensor(quant(z.F, training=self.training),coordinate_map_key=z.coordinate_map_key,coordinate_manager=z.coordinate_manager)
                z_hat = z_q
                y_coor = ME.SparseTensor(torch.ones([cur_f[4].size()[0], 1], dtype=torch.float32, device=cur_f[4].device),cur_f[4].C, tensor_stride=cur_f[4].tensor_stride)
                hierarchical_params    = self.contextual_hyper_prior_decoder(z_hat, y_coor)#上采样一次
                temporal_params = self.temporal_prior_encoder(context1=context1)#下采样一次，维度和ys2[4]保持一致
                hierarchical_params = sort_by_coor_sum(hierarchical_params)
                temporal_params     = sort_by_coor_sum(temporal_params )
                if not torch.equal(hierarchical_params.C, temporal_params.C):
                    raise ValueError("Temporal context and hyper-prior must have the same coordinates")

                cat_params = torch.concat([hierarchical_params.F,temporal_params.F],dim=1)
                new_params = ME.SparseTensor( cat_params, coordinates=temporal_params.C, coordinate_manager=temporal_params.coordinate_manager,tensor_stride=temporal_params.tensor_stride, device=temporal_params.device)
                gaussian_params        =  self.contextual_entropy_parameter(new_params)
                scales_hat , means_hat = gaussian_params.F.chunk(2, 1)
                y_q = quant(cur_f[4].F.unsqueeze(0) , training=self.training)
                #量化后的稀疏卷积
                y2_recon = ME.SparseTensor(features=y_q.squeeze(0),coordinates=cur_f[4].C,tensor_stride=cur_f[4].tensor_stride,device=cur_f[4].device)
                z_for_bit =z_q.F.unsqueeze(0)
                total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
                y_for_bit = y_q
                # print("y_for_bit, scales_hat:",y_for_bit.shape, scales_hat.shape)
    #             print("y_for_bit:",y_for_bit.shape)
                total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
                bits = total_bits_z+total_bits_y
                bpp_y = total_bits_y/num_points
                bpp_z = total_bits_z/num_points
                motion_bpp = 0
                out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1_v2(y2_recon, cur_f[2], True, residual=context1,context2=None,f1_ref=None)
                out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], cur_f[1], True, 1 if self.training else 1)
                out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], cur_f[0], True, 1 if self.training else 1)
                
                recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)

                # self.update_buffer(recon_f2, video_idx, frame_idx)
                self.update_buffer(current_f, video_idx, frame_idx)

            
                bpp = bits/ num_points
                
                return cur_f, out2, out_cls2, target2, keep2, bpp 
                
            
        
        else:
            ys_ref1[1] = self.enc1(ys_ref1[0])
            ys_ref1[2] = self.enc2(ys_ref1[1])
            
            ys_ref2[1] = self.enc1(ys_ref2[0])
            ys_ref2[2] = self.enc2(ys_ref2[1])
            #第二帧两次下采样
            cur_f[1]   = self.enc1(cur_f[0])
            cur_f[2]   = self.enc2(cur_f[1])
            if self.knn_fixed==False:
                residual_1, predicted_point2_1, quant_motion_1 ,motion_hat_1 = self.inter_prediction(ys_ref1[2] ,cur_f[2], stride=4)
                residual_2, predicted_point2_2, quant_motion_2 ,motion_hat_2 = self.inter_prediction(ys_ref2[2] ,cur_f[2], stride=4)

                predicted_point2_F=self.feature_add(predicted_point2_1, predicted_point2_2, cur_f[2])

                predicted_point2=ME.SparseTensor(predicted_point2_F, coordinates=cur_f[2].C, coordinate_manager=cur_f[2].coordinate_manager,
                                           tensor_stride=4, device=cur_f[2].device)

                residual=cur_f[2]-predicted_point2 
                residual = ME.SparseTensor(residual.F, coordinates=residual.C, tensor_stride=4, device=cur_f[2].device)


                quant_motion_F_1 = quant_motion_1.F.unsqueeze(0)#1*N*64*3
                motion_p_1 = self.MotionBitEstimator(quant_motion_F_1+0.5) - self.MotionBitEstimator(quant_motion_F_1-0.5)
                motion_bits_1 = torch.sum(torch.clamp(-1.0 * torch.log(motion_p_1 + 1e-10) / math.log(2.0), 0, 50))
                motion_bits_1 = motion_bits_1

                quant_motion_F_2 = quant_motion_2.F.unsqueeze(0)#1*N*64*3
                motion_p_2 = self.MotionBitEstimator(quant_motion_F_2+0.5) - self.MotionBitEstimator(quant_motion_F_2-0.5)
                motion_bits_2 = torch.sum(torch.clamp(-1.0 * torch.log(motion_p_2 + 1e-10) / math.log(2.0), 0, 50))
                motion_bits_2 = motion_bits_2
                motion_bits = motion_bits_1 + motion_bits_2
                
            elif self.knn_fixed==True:
                predicted_point2_1, residual_1 ,_ = self.inter_prediction_v2(ys_ref1[2] ,cur_f[2])
                predicted_point2_2, residual_2 ,_ = self.inter_prediction_v2(ys_ref2[2] ,cur_f[2])
                
                predicted_point2_F=self.feature_add(predicted_point2_1, predicted_point2_2, cur_f[2])
                
                #
                
                predicted_point2=ME.SparseTensor(predicted_point2_F, coordinates=cur_f[2].C, coordinate_manager=cur_f[2].coordinate_manager,
                                           tensor_stride=4, device=cur_f[2].device)
                
               
                #对预测的point增加一层卷积
                if self.fuse_conv:
                    predicted_point2 = self.fuse_conv_2(predicted_point2)
                

                # residual=cur_f[2]-predicted_point2 
                # residual = ME.SparseTensor(residual.F, coordinates=residual.C, tensor_stride=4, device=cur_f[2].device)
                # motion_bits=0
                
            if self.entropy_mode=="ori":
            
                cur_f[3] = self.enc3(residual)
                cur_f[4] = self.enc4(cur_f[3])

                cur_f[4]=sort_by_coor_sum(cur_f[4])

                quant_y = quant(cur_f[4].F.unsqueeze(0), training=self.training)
                # 这里有了要传输的两部分，分别是quant_motion_F和quant_y
                # bit rate calculation
                p = self.BitEstimator(quant_y+0.5) - self.BitEstimator(quant_y-0.5)
                bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
                y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=cur_f[4].coordinate_map_key,coordinate_manager=cur_f[4].coordinate_manager, device=cur_f[4].device)
                bpp = (bits + motion_bits) / num_points
                out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, cur_f[2], True, residual=predicted_point2)



                out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], cur_f[1], True, 1 if self.training else 1)
                out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], cur_f[0], True, 1 if self.training else 1)
                recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)


                # self.update_buffer(recon_f2, video_idx, frame_idx)
                self.update_buffer(current_f, video_idx, frame_idx)

                return cur_f, out2, out_cls2, target2, keep2, bpp 
            
            if self.entropy_mode =="context_cond":
                
                context1 = predicted_point2 
                cur_f[4] = self.contextual_encoder_v2(cur_f[2],context1)
                
                z = self.contextual_hyper_prior_encoder(cur_f[4])#下采样一次
                z_q = ME.SparseTensor(quant(z.F, training=self.training),coordinate_map_key=z.coordinate_map_key,coordinate_manager=z.coordinate_manager)
                z_hat = z_q
                y_coor = ME.SparseTensor(torch.ones([cur_f[4].size()[0], 1], dtype=torch.float32, device=cur_f[4].device),cur_f[4].C, tensor_stride=cur_f[4].tensor_stride)
                hierarchical_params    = self.contextual_hyper_prior_decoder(z_hat, y_coor)#上采样一次
                temporal_params = self.temporal_prior_encoder(context1=context1)#下采样一次，维度和ys2[4]保持一致
                hierarchical_params = sort_by_coor_sum(hierarchical_params)
                temporal_params     = sort_by_coor_sum(temporal_params )
                if not torch.equal(hierarchical_params.C, temporal_params.C):
                    raise ValueError("Temporal context and hyper-prior must have the same coordinates")

                cat_params = torch.concat([hierarchical_params.F,temporal_params.F],dim=1)
                new_params = ME.SparseTensor( cat_params, coordinates=temporal_params.C, coordinate_manager=temporal_params.coordinate_manager,tensor_stride=temporal_params.tensor_stride, device=temporal_params.device)
                gaussian_params        =  self.contextual_entropy_parameter(new_params)
                scales_hat , means_hat = gaussian_params.F.chunk(2, 1)
                y_q = quant(cur_f[4].F.unsqueeze(0) , training=self.training)
                #量化后的稀疏卷积
                y2_recon = ME.SparseTensor(features=y_q.squeeze(0),coordinates=cur_f[4].C,tensor_stride=cur_f[4].tensor_stride,device=cur_f[4].device)
                z_for_bit =z_q.F.unsqueeze(0)
                total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
                y_for_bit = y_q
                # print("y_for_bit, scales_hat:",y_for_bit.shape, scales_hat.shape)
    #             print("y_for_bit:",y_for_bit.shape)
                total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
                bits = total_bits_z+total_bits_y
                bpp_y = total_bits_y/num_points
                bpp_z = total_bits_z/num_points
                motion_bpp = 0
                out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1_v2(y2_recon, cur_f[2], True, residual=context1,context2=None,f1_ref=None)
                out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], cur_f[1], True, 1 if self.training else 1)
                out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], cur_f[0], True, 1 if self.training else 1)
                
                recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)


                # self.update_buffer(recon_f2, video_idx, frame_idx)
                self.update_buffer(current_f, video_idx, frame_idx)
            
                bpp = bits/ num_points
                
                return cur_f, out2, out_cls2, target2, keep2, bpp 
                
    @staticmethod
    def get_y_bits_probs(y, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(y + 0.5) - gaussian.cdf(y - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, probs

    @staticmethod
    def get_z_bits_probs(z, bit_estimator):
        # print("=====>z:",z.shape)[649,32]
        
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, prob            

        
        
        
        
        # feature extraction
        #第一帧两次下采样 
#         ys1[1] = self.enc1(ys1[0])
#         ys1[2] = self.enc2(ys1[1])
#         #第二帧两次下采样
#         ys2[1] = self.enc1(ys2[0])
#         ys2[2] = self.enc2(ys2[1])

#         # inter prediction
#         residual, predicted_point2, quant_motion ,motion_hat = self.inter_prediction(ys1[2], ys2[2], stride=4)

#         # residual compression
#         quant_motion_F = quant_motion.F.unsqueeze(0)#1*N*64*3
#         motion_p = self.MotionBitEstimator(quant_motion_F+0.5) - self.MotionBitEstimator(quant_motion_F-0.5)
#         motion_bits = torch.sum(torch.clamp(-1.0 * torch.log(motion_p + 1e-10) / math.log(2.0), 0, 50))

#         factor = 0.95
#         if self.training:
#             motion_bits = factor * motion_bits

        
#         ys2[3] = self.enc3(residual)
#         ys2[4] = self.enc4(ys2[3])

#         quant_y = quant(ys2[4].F.unsqueeze(0), training=self.training)
#         # 这里有了要传输的两部分，分别是quant_motion_F和quant_y
#         # bit rate calculation
#         p = self.BitEstimator(quant_y+0.5) - self.BitEstimator(quant_y-0.5)
#         bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
#         y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=ys2[4].coordinate_map_key,coordinate_manager=ys2[4].coordinate_manager, device=ys2[4].device)
#         bpp = (bits + motion_bits) / num_points
#         out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, ys2[2], True, residual=predicted_point2)

        
        
#         out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], ys2[1], True, 1 if self.training else 1)
#         out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], ys2[0], True, 1 if self.training else 1)
        
        
#         return ys2, out2, out_cls2, target2, keep2, bpp ,bpp_y, bpp_z,motion_bpp