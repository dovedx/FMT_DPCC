import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.cond_utils import *

from models.pointconv_util import*
from GPCC.gpcc_wrapper import *

from models.PTFeaturePredictor import PointCloudFeaturePredictor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional

# from compressai.latent_codecs.entropy_bottleneck import EntropyBottleneckLatentCodec

# from compressai.models.priors import CompressionModel, GaussianConditional
# from compressai.ops import ste_round
from compressai.models.utils import update_registered_buffers

# from models.knn_context_model import KNNContextModel  # 假设的 k-NN 上下文模型
# from torch_cluster import knn
from torch import Tensor
def ste_round(x: Tensor) -> Tensor:
    return torch.round(x) - x.detach() + x

class get_model(nn.Module):
    def __init__(self, channels=8,entropy_mode='cond',knn_fixed=True,point_conv=False):
        super(get_model, self).__init__()
        
        self.entropy_mode =entropy_mode
        self.knn_fixed = knn_fixed
        self.point_conv = point_conv
        
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3) 
        
        if self.knn_fixed:
            self.inter_prediction_v2 = PointCloudFeaturePredictor(in_channels=64, K=8,point_conv=self.point_conv)
        else:
            self.inter_prediction = inter_prediction(64, 64, 48)

        #原始的残差编码和解码器
        if self.entropy_mode=="ori":
            self.enc3 = DownsampleLayer(64, 64, 32, 3)
            self.enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)
            self.dec1 = UpsampleLayer(channels, 64, 64, 3)

            self.dec2 = UpsampleLayer(64, 32, 32, 3)
            self.dec3 = UpsampleLayer(32, 16, 16, 3)

            self.BitEstimator = BitEstimator(channels, 3)
            # self.MotionBitEstimator = BitEstimator(48, 3)
            self.crit = torch.nn.BCEWithLogitsLoss()

        # 基于条件自适应的编码器和解码器 
        if self.entropy_mode=='cond':
            self.bit_estimator_z = BitEstimator(channel=8,dimension=3)
            
            self.contextual_encoder = Point_ContextualEncoder(channels=64 , block_layers=3)
            self.contextual_hyper_prior_encoder = Point_contextual_hyper_prior_encoder(channels=64)
            self.contextual_hyper_prior_decoder =Point_contextual_hyper_prior_decoder(channels=64 , block_layers=3)
            self.contextual_entropy_parameter = Point_contextual_entropy_parameter(in_channels=64,channels=64)
            
            self.dec1 = UpsampleLayer(8, 64, 64, 3)
        
        if self.entropy_mode =="context_cond":
            
            self.bit_estimator_z = BitEstimator(channel=64,dimension=3)
            
            # self.context_produce = Point_context_produce(channels=64)
            
            self.contextual_encoder_v2 = Point_ContextualEncoder(channels=64,channels_latent=channels )
            self.contextual_hyper_prior_encoder = Point_contextual_hyper_prior_encoder(channels=64)
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
            self.gaussian_conditional = GaussianConditional(None)
            self.entropy_bottleneck = EntropyBottleneck(64)

        
        if self.entropy_mode =="context_cond_v2":
             # 超编码器：映射潜在表示到超潜在空间
            # self.groups = [0, 16, 16, 32, 64, 192]
            self.groups = [0, 8, 8, 16, 16, 16]
            self.num_slices = num_slices
            M=64
            N=64
                
            self.h_a = nn.Sequential(
                ME.MinkowskiConvolution(M, N, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(N, N, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(N, N, kernel_size=3, dimension=3),
            )

            # 超解码器：预测潜在分布的均值和尺度
            self.h_s = nn.Sequential(
                ME.MinkowskiConvolutionTranspose(N, N, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolutionTranspose(N, N*3//2, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=True),
                ME.MinkowskiConvolution(N*3//2, 2*M, kernel_size=3, dimension=3),  # 输出均值和尺度
            )

            # 上下文变换：转换支持切片以预测均值和尺度
            self.cc_transforms = nn.ModuleList(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        self.groups[min(1, i) if i > 0 else 0] + self.groups[i if i > 1 else 0],
                        224, kernel_size=5, dimension=3
                    ),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(224, 128, kernel_size=5, dimension=3),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(128, self.groups[i + 1]*2, kernel_size=5, dimension=3),
                ) for i in range(1, num_slices)
            )

            # k-NN 上下文模型：替换棋盘格模型
            self.context_prediction = nn.ModuleList(
                KNNContextModel(
                    in_channels=self.groups[i+1],
                    out_channels=2*self.groups[i+1],
                    k=8,  # 使用 8 个最近邻
                    dimension=3
                ) for i in range(num_slices)
            )

            # 参数聚合：结合上下文和支持信息
            self.ParamAggregation = nn.ModuleList(
                nn.Sequential(
                    ME.MinkowskiConvolution(
                        640 + self.groups[i+1 if i > 0 else 0] * 2 + self.groups[i + 1] * 2,
                        640, kernel_size=1, dimension=3
                    ),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(640, 512, kernel_size=1, dimension=3),
                    ME.MinkowskiReLU(inplace=True),
                    ME.MinkowskiConvolution(512, self.groups[i + 1]*2, kernel_size=1, dimension=3),
                ) for i in range(num_slices)
            )

            self.quantizer = Quantizer()
            self.gaussian_conditional = GaussianConditional(None)
            
            

    def forward(self, f1, f2, device, epoch=99999):
        
        # print("self.training===:",self.training)

        num_points = f2.C.size(0)#获得点云的点数

        ys1, ys2 = [f1, 0, 0, 0, 0], [f2, 0, 0, 0, 0]

        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

        # feature extraction
        #第一帧两次下采样 
        ys1[1] = self.enc1(ys1[0])
        ys1[2] = self.enc2(ys1[1])
        #第二帧两次下采样
        ys2[1] = self.enc1(ys2[0])
        ys2[2] = self.enc2(ys2[1])

        # inter prediction
        if self.knn_fixed:
            predicted_point2, residual ,_= self.inter_prediction_v2(ys1[2], ys2[2])
        else:
        
            # residual, predicted_point2, quant_motion ,motion_hat = self.inter_prediction(ys1[2], ys2[2], stride=4)
            pass

        # residual compression
        if self.knn_fixed==False:
            quant_motion_F = quant_motion.F.unsqueeze(0)#1*N*64*3
            motion_p = self.MotionBitEstimator(quant_motion_F+0.5) - self.MotionBitEstimator(quant_motion_F-0.5)
            motion_bits = torch.sum(torch.clamp(-1.0 * torch.log(motion_p + 1e-10) / math.log(2.0), 0, 50))

            factor = 0.95
            if self.training:
                motion_bits = factor * motion_bits
        else:
            motion_bits = 0

        if self.entropy_mode=="ori":
            # print("residual",residual.shape)
            ys2[3] = self.enc3(residual)
            ys2[4] = self.enc4(ys2[3])
            
            ys2[4]=sort_by_coor_sum(ys2[4])
            
            quant_y = quant(ys2[4].F.unsqueeze(0), training=self.training)
            # print("self.training",self.training)
            # print("test 模式下 量化的quant_y:",quant_y[0][1:10,...],torch.sum(quant_y))
            # 这里有了要传输的两部分，分别是quant_motion_F和quant_y
            # bit rate calculation
            p = self.BitEstimator(quant_y+0.5) - self.BitEstimator(quant_y-0.5)
            bits = torch.sum(torch.clamp(-1.0 * torch.log(p + 1e-10) / math.log(2.0), 0, 50))
            
            y2_recon = ME.SparseTensor(quant_y.squeeze(0), coordinate_map_key=ys2[4].coordinate_map_key,coordinate_manager=ys2[4].coordinate_manager, device=ys2[4].device)
            
            if self.knn_fixed:
                bpp = bits / num_points
            else:
                bpp = (bits + motion_bits) / num_points
            
            # print("predicted_point2:",predicted_point2.F[1:10,0],torch.sum(predicted_point2.F))
            
            y2_recon = sort_by_coor_sum(y2_recon)
            
            # print("y2_recon:",y2_recon.F[1:10,...],y2_recon.C[1:20,1:],torch.sum(y2_recon.F))
            
            out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, ys2[2], True, residual=predicted_point2)
            
            bpp_y=bits / num_points
            bpp_z=0
            motion_bpp=motion_bits / num_points
         
        elif self.entropy_mode=='cond':
            
            y2_recon,y_q,z_q,scales_hat,ys2[4] = self.condtional_enc_model(residual,context1=None,context2=None,context3=None)
            
            y_for_bit = y_q
            z_for_bit = z_q.F.unsqueeze(0)#这里z_q是一个sparse_tensor,故要将他的特征提取出来
            
            # print("y_for_bit,z_for_bit:",y_for_bit.shape,z_for_bit.shape)
            
            total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
            
            total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
            
            bits = total_bits_y + total_bits_z
 
            bpp = (bits + motion_bits) / num_points
            out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1(y2_recon, ys2[2], True, residual=predicted_point2)
        
        elif self.entropy_mode=='context_cond':
            #根据要编码的残差和解码的motion,来得到上下文
            context1 = predicted_point2 #由隐式motion估计空间估计出的上下文先验
            
            ys2[4] = self.contextual_encoder_v2(ys2[2],context1)#对齐下采样得到要编码的隐式特征(下采样1次)
            
            print(f"输入hyper min/max/mean/std: {ys2[4].F.min().item():.4f}/{ys2[4].F.max().item():.4f}/{ys2[4].F.mean().item():.4f}/{ys2[4].F.std().item():.4f}")

            z = self.contextual_hyper_prior_encoder(ys2[4])#下采样一次
            print(f"输出 z.F min/max/mean/std: {z.F.min().item():.4f}/{z.F.max().item():.4f}/{z.F.mean().item():.4f}/{z.F.std().item():.4f}")
            #对z做量化
            # z_q = ME.SparseTensor(quant(z.F, training=self.training),coordinate_map_key=z.coordinate_map_key,coordinate_manager=z.coordinate_manager)
            

            # z_q = ME.SparseTensor(
            #     quant(z.F , training=self.training),
            #     coordinate_map_key=z.coordinate_map_key,
            #     coordinate_manager=z.coordinate_manager
            # )
            
#             print(f"z_q.F min/max: {z_q.F.min().item():.4f}/{z_q.F.max().item():.4f}")
            
#             z_hat = z_q
            
            
            z_for_bit =z.F.unsqueeze(0).transpose(1,2)#[1,N,C]
            _, z_likelihoods = self.entropy_bottleneck(z_for_bit)
            total_bits_z = torch.log(z_likelihoods).sum()/(-math.log(2))
            # print(f"total_bits_z: {total_bits_z.item():.4f}")
            
            if self.training:
                z_offset = self.entropy_bottleneck._get_medians()
                z_offset =z_offset.squeeze().unsqueeze(0)
                # print("z_offset:",z_offset.squeeze().unsqueeze(0).shape,z.F.shape)
                z_tmp_F = z.F - z_offset
                z_hat_F = ste_round(z_tmp_F) + z_offset
                # print("z_hat_F",z_hat_F.shape)
                z_q = ME.SparseTensor(
                    z_hat_F,
                    coordinate_map_key=z.coordinate_map_key,
                    coordinate_manager=z.coordinate_manager
                )
                z_hat = z_q
            
            else:
                z_offset = self.entropy_bottleneck._get_medians()
                z_offset =z_offset.squeeze().unsqueeze(0)
                # print("z_offset:",z_offset.squeeze().unsqueeze(0).shape,z.F.shape)
                z_tmp_F = z.F - z_offset
                z_hat_F = ste_round(z_tmp_F) + z_offset
                # print("z_hat_F",z_hat_F.shape)
                z_q = ME.SparseTensor(
                    z_hat_F,
                    coordinate_map_key=z.coordinate_map_key,
                    coordinate_manager=z.coordinate_manager
                )
                z_hat = z_q
                
            

            # 对z做解码，得到编码y_latent的均值和方差
            y_coor = ME.SparseTensor(torch.ones([ys2[4].size()[0], 1], dtype=torch.float32, device=ys2[4].device),ys2[4].C, tensor_stride=ys2[4].tensor_stride)
            hierarchical_params    = self.contextual_hyper_prior_decoder(z_hat, y_coor)#上采样一次

            temporal_params = self.temporal_prior_encoder(context1=context1)#下采样一次，维度和ys2[4]保持一致       
            hierarchical_params = sort_by_coor_sum(hierarchical_params)
            temporal_params     = sort_by_coor_sum(temporal_params )
            if not torch.equal(hierarchical_params.C, temporal_params.C):
                raise ValueError("Temporal context and hyper-prior must have the same coordinates")
            
            cat_params = torch.concat([hierarchical_params.F,temporal_params.F],dim=1)
            print("cat_params , ",cat_params.shape)
            new_params = ME.SparseTensor( cat_params, coordinates=temporal_params.C, coordinate_manager=temporal_params.coordinate_manager,tensor_stride=temporal_params.tensor_stride, device=temporal_params.device)
            gaussian_params        =  self.contextual_entropy_parameter(new_params)
            scales_hat , means_hat = gaussian_params.F.chunk(2, 1)
            
#             print("scales_hat , means_hat:",scales_hat.shape, means_hat.shape)
            
            # print(f"y.F min/max/mean/std: {ys2[4].F.min().item():.4f}/{ys2[4].F.max().item():.4f}/{ys2[4].F.mean().item():.4f}/{ys2[4].F.std().item():.4f}")
  
            #对要编码的y_latent做量化
            y_q = quant(ys2[4].F.unsqueeze(0) , training=self.training)
            # print(f"y_q.F min/max: {y_q.min().item():.4f}/{y_q.max().item():.4f}")
            #量化后的稀疏卷积
            y2_recon = ME.SparseTensor(features=y_q.squeeze(0),coordinates=ys2[4].C,tensor_stride=ys2[4].tensor_stride,device=ys2[4].device)
            
            
#             print("total_bits_z:",total_bits_z)
            #y码流：
            y_for_bit = y_q
            # print("y_for_bit, scales_hat:",y_for_bit.transpose(1,2).shape, scales_hat.unsqueeze(0).transpose(1,2).shape)
#             print("y_for_bit:",y_for_bit.shape)
            # total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
            
            _, y_likelihoods = self.gaussian_conditional(y_for_bit.transpose(1,2), scales_hat.unsqueeze(0).transpose(1,2), means=means_hat.unsqueeze(0).transpose(1,2))
            total_bits_y = torch.log(y_likelihoods).sum()/(-math.log(2))
            
            print(f"total_bits_z:{total_bits_z}  ,  total_bits_y:{total_bits_y}")

        
            bits = total_bits_z+total_bits_y
            bpp_y = total_bits_y/num_points
            bpp_z = total_bits_z/num_points
            motion_bpp = 0
            
#             print("bpp_y , bpp_z:",bpp_y, bpp_z)
            
            
            #开始对量化后的y_q做decoder
            out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1_v2(y2_recon, ys2[2], True, residual=context1,context2=None,f1_ref=None)
            
            bpp = bits/ num_points
            
        elif self.entropy_mode=='context_cond_v2':
            context1 = predicted_point2 #由隐式motion估计空间估计出的上下文先验
            
            ys2[4] = self.contextual_encoder_v2(ys2[2],context1)#对齐下采样得到要编码的隐式特征(下采样1次)
            z = self.h_a(ys2[4])
            z_hat, z_likelihoods = self.entropy_bottleneck(z.F)
            
            if not noisequant:#训练模式
                z_offset = self.entropy_bottleneck._get_medians()
                z_tmp = z.F - z_offset
                z_hat = ME.SparseTensor(
                    features=ste_round(z_tmp) + z_offset,
                    coordinates=z.C,
                    device=z.device
                )
            
            latent_params = self.h_s(z_hat)
            latent_means, latent_scales = latent_params.F.chunk(2, dim=1)
            # 分割潜在表示为切片
            y_slices = []
            for i in range(self.num_slices):
                start = self.groups[i]
                end = self.groups[i + 1]
                y_slices.append(
                    ME.SparseTensor(
                        features=y.F[:, start:end],
                        coordinates=y.C,
                        device=y.device
                    )
                )

            # 基于邻域的上下文分割（替换棋盘格）
            anchor = []
            non_anchor = []
            for i, y_slice in enumerate(y_slices):
                # 使用 k-NN 分割锚点和非锚点（基于点索引）
                indices = torch.arange(y_slice.F.shape[0], device=y_slice.device)
                anchor_mask = (indices % 2 == 0)  # 简单示例：偶数索引为锚点
                non_anchor_mask = ~anchor_mask
                anchor.append(
                    ME.SparseTensor(
                        features=y_slice.F[anchor_mask],
                        coordinates=y_slice.C[anchor_mask],
                        device=y_slice.device
                    )
                )
                non_anchor.append(
                    ME.SparseTensor(
                        features=y_slice.F[non_anchor_mask],
                        coordinates=y_slice.C[non_anchor_mask],
                        device=y_slice.device
                    )
                )

            ctx_params_anchor = [
                ME.SparseTensor(
                    features=torch.zeros_like(y_slice.F),
                    coordinates=y_slice.C,
                    device=y_slice.device
                ) for y_slice in y_slices
            ]
            
            y_hat_slices = []
            y_hat_slices_for_gs = []
            y_likelihood = []
            for slice_index, y_slice in enumerate(y_slices):
                if slice_index == 0:
                    support_slices = None
                elif slice_index == 1:
                    support_slices = y_hat_slices[0]
                    support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                    support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.F.chunk(2, dim=1)
                    support_slices_ch = ME.SparseTensor(
                        features=torch.cat([support_slices_ch_mean, support_slices_ch_scale], dim=1),
                        coordinates=support_slices.C,
                        device=support_slices.device
                    )
                else:
                    support_slices = ME.cat(y_hat_slices[0], y_hat_slices[slice_index-1])
                    support_slices_ch = self.cc_transforms[slice_index-1](support_slices)
                    support_slices_ch_mean, support_slices_ch_scale = support_slices_ch.F.chunk(2, dim=1)
                    support_slices_ch = ME.SparseTensor(
                        features=torch.cat([support_slices_ch_mean, support_slices_ch_scale], dim=1),
                        coordinates=support_slices.C,
                        device=support_slices.device
                    )

                support = ME.SparseTensor(
                    features=torch.cat([latent_means, latent_scales], dim=1),
                    coordinates=y.C,
                    device=y.device
                ) if slice_index == 0 else ME.cat(support_slices_ch, ME.SparseTensor(
                    features=torch.cat([latent_means, latent_scales], dim=1),
                    coordinates=y.C,
                    device=y.device
                ))

                y_anchor = anchor[slice_index]
                params = self.ParamAggregation[slice_index](
                    ME.cat(ctx_params_anchor[slice_index], support)
                )
                means_anchor, scales_anchor = params.F.chunk(2, dim=1)
                means_hat = torch.zeros_like(y_slice.F, device=y_slice.device)
                scales_hat = torch.zeros_like(y_slice.F, device=y_slice.device)
                anchor_indices = y_slice.decomposed_coordinates[0]
                means_hat[anchor_mask] = means_anchor
                scales_hat[anchor_mask] = scales_anchor

                if noisequant:
                    y_anchor_quantized = self.quantizer.quantize(y_anchor.F, "noise")
                    y_anchor_quantized_for_gs = self.quantizer.quantize(y_anchor.F, "ste")
                else:
                    y_anchor_quantized = self.quantizer.quantize(y_anchor.F - means_anchor, "ste") + means_anchor
                    y_anchor_quantized_for_gs = self.quantizer.quantize(y_anchor.F - means_anchor, "ste") + means_anchor

                y_anchor_quantized = ME.SparseTensor(
                    features=y_anchor_quantized,
                    coordinates=y_anchor.C,
                    device=y_anchor.device
                )
                y_anchor_quantized_for_gs = ME.SparseTensor(
                    features=y_anchor_quantized_for_gs,
                    coordinates=y_anchor.C,
                    device=y_anchor.device
                )

                masked_context = self.context_prediction[slice_index](y_anchor_quantized)
                params_non_anchor = self.ParamAggregation[slice_index](
                    ME.cat(masked_context, support)
                )
                means_non_anchor, scales_non_anchor = params_non_anchor.F.chunk(2, dim=1)
                means_hat[non_anchor_mask] = means_non_anchor
                scales_hat[non_anchor_mask] = scales_non_anchor

                _, y_slice_likelihood = self.gaussian_conditional(y_slice.F, scales_hat, means=means_hat)

                y_non_anchor = non_anchor[slice_index]
                if noisequant:
                    y_non_anchor_quantized = self.quantizer.quantize(y_non_anchor.F, "noise")
                    y_non_anchor_quantized_for_gs = self.quantizer.quantize(y_non_anchor.F, "ste")
                else:
                    y_non_anchor_quantized = self.quantizer.quantize(y_non_anchor.F - means_non_anchor, "ste") + means_non_anchor
                    y_non_anchor_quantized_for_gs = self.quantizer.quantize(y_non_anchor.F - means_non_anchor, "ste") + means_non_anchor

                y_non_anchor_quantized = ME.SparseTensor(
                    features=y_non_anchor_quantized,
                    coordinates=y_non_anchor.C,
                    device=y_non_anchor.device
                )
                y_non_anchor_quantized_for_gs = ME.SparseTensor(
                    features=y_non_anchor_quantized_for_gs,
                    coordinates=y_non_anchor.C,
                    device=y_non_anchor.device
                )

                y_hat_slice = ME.cat(y_anchor_quantized, y_non_anchor_quantized)
                y_hat_slice_for_gs = ME.cat(y_anchor_quantized_for_gs, y_non_anchor_quantized_for_gs)
                y_hat_slices.append(y_hat_slice)
                y_hat_slices_for_gs.append(y_hat_slice_for_gs)
                y_likelihood.append(y_slice_likelihood)

            y_likelihoods = torch.cat(y_likelihood, dim=1)
            y_hat = ME.cat(*y_hat_slices_for_gs)
            x_hat = self.g_s(y_hat)  # 重建点云坐标
            

        out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], ys2[1], True, 1 if self.training else 1)
        out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], ys2[0], True, 1 if self.training else 1)
        
        return ys2, out2, out_cls2, target2, keep2, bpp ,bpp_y, bpp_z,motion_bpp
    
        
            


    
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
    
    def condtional_enc_model(self,residual , context2,context3):
        
        # print("self.training:",self.training)
        
        # print("context2,context3:",context2.shape,context3.shape)
        
        # print("residual:",residual.shape)
        y = self.contextual_encoder(residual,context2=None)
        #y是即将要量化的目标，点的数量为Mx3
        # print("y:",y.shape)
        #这里根据y_voxel和y_point生成超先验
        # y->y.C,y.F 
        # print("y===>:",y.shape)
        
        z = self.contextual_hyper_prior_encoder(y)
        
        # print("z:",z.shape)
        z_q = ME.SparseTensor(quant(z.F, training=self.training),
                                                  coordinate_map_key=z.coordinate_map_key,
                                                  coordinate_manager=z.coordinate_manager)
        z_hat = z_q
        
        # print("z_q:",z_hat.shape)
        y_coor = ME.SparseTensor(torch.ones([y.size()[0], 1], dtype=torch.float32, device=y.device),y.C, tensor_stride=y.tensor_stride)
        
        # print("y_coor:",y_coor.shape)
                                    
        hierarchical_params    = self.contextual_hyper_prior_decoder(z_hat, y_coor)
        # print("hierarchical_params:",hierarchical_params.shape)
        if self.entropy_mode=="context_cond":
            temporal_params = self.temporal_prior_encoder(context2,context3)
            
            cat_params = torch.concat([hierarchical_params.F,temporal_params.F],dim=1)
        
            new_params = ME.SparseTensor( cat_params, coordinates=context2.C, coordinate_manager=temporal_params.coordinate_manager,tensor_stride=temporal_params.tensor_stride, device=temporal_params.device)
            gaussian_params        =  self.contextual_entropy_parameter(new_params)
            scales_hat , means_hat = gaussian_params.F.chunk(2, 1)
            
            # print("====>:",torch.sum(hierarchical_params.C-temporal_params.C))
        else:
        
            gaussian_params        =  self.contextual_entropy_parameter(hierarchical_params)
            scales_hat , means_hat = gaussian_params.F.chunk(2, 1)

            # print("scales_hat , means_hat:",scales_hat.shape , means_hat.shape)
        
#         if self.entropy_mode=="context_cond":
#             y_res = y.F - means_hat
#             y_q   =  quant(y_res , training=self.training)
#             y_hat = y_q + means_hat
            
#             y2_recon = ME.SparseTensor(features=y_hat,coordinates=y.C,tensor_stride=y.tensor_stride,device=y.device)

#             return y2_recon, y_hat, z_q, scales_hat,y
    
    
    
        y_q = quant(y.F.unsqueeze(0) , training=self.training)
        y2_recon = ME.SparseTensor(features=y_q.squeeze(0),coordinates=y.C,tensor_stride=y.tensor_stride,device=y.device)
        
        
        print("y_q:",y_q.shape)
        print("scales_hat:",scales_hat.shape)
        return y2_recon, y_q, z_q, scales_hat,y

       


    def get_scales(self,idx):
        y_enc_scale =self.y_enc_scale[idx].unsqueeze(0)
        y_dec_scale =self.y_dec_scale[idx].unsqueeze(0)
        mv_enc_scale = self.mv_enc_scale[idx].unsqueeze(0)
        mv_dec_scale = self.mv_dec_scale[idx].unsqueeze(0)
        mv_z_enc_scale = self.mv_z_enc_scale[idx].unsqueeze(0)
        mv_z_dec_scale = self.mv_z_dec_scale[idx].unsqueeze(0)
        z_enc_scale = self.z_enc_scale[idx].unsqueeze(0)
        z_dec_scale = self.z_dec_scale[idx].unsqueeze(0)
        return y_enc_scale,y_dec_scale,mv_enc_scale,mv_dec_scale,mv_z_enc_scale,mv_z_dec_scale,z_enc_scale,z_dec_scale


    


if __name__ == '__main__':
    from dataset_lossy import *
    import os

    torch.manual_seed(0)

    d_model = 32
    seq_len = 2000
    batch_size = 1
    num_heads = 4
    k_dim = 8

    tmp_dir = os.getcwd()
    '''    '''
    feat1 = torch.randint(low=0, high=2, size=(seq_len, 1), dtype=torch.float32)
    # coord1 = torch.randint(low=0, high=2000, size=(seq_len, 3), dtype=torch.float32)
    coord1 = [[2 * y for i in range(3)] for y in range(seq_len)]
    coord1 = torch.Tensor(coord1)
    coords1, feats1 = ME.utils.sparse_collate(coords=[coord1], feats=[feat1])
    # input1 = ME.SparseTensor(coordinates=coords1, features=feats1, tensor_stride=1)
    input1 = ME.SparseTensor(coordinates=coords1, features=feats1)

    feat2 = torch.randint(low=0, high=2, size=(seq_len, 1), dtype=torch.float32)
    # coord2 = torch.randint(low=0, high=2000, size=(seq_len, 3), dtype=torch.float32)
    coord2 = [[2 * y + 1 for i in range(3)] for y in range(seq_len)]
    coord2 = torch.Tensor(coord2)
    coords2, feats2 = ME.utils.sparse_collate(coords=[coord2], feats=[feat2])
    # input2 = ME.SparseTensor(coordinates=coords2, features=feats2, tensor_stride=1)
    input2 = ME.SparseTensor(coordinates=coords2, features=feats2)

    model_test = get_model(channels=8)
    _, out2, _, _, _, _ = model_test(input1, input2, device='cpu')  # device='cpu' may error in unpooling
    output = out2[-1]
    print(output.C.shape)  # output.C is final output points. 16-channel .F makes no sense