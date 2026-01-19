import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.cond_utils import *

from models.pointconv_util import*
from GPCC.gpcc_wrapper import *

from models.PTFeaturePredictor import PointCloudFeaturePredictor
from models.PTFeaturePredictor import CrossPointTransformer
from models.PTFeaturePredictor import CrossPointTransformer_v2


from models.distribution import compare_residual_distributions
from models.distribution import compare_three_residual_distributions
class get_model(nn.Module):
    def __init__(self, channels=8,entropy_mode='cond',knn_fixed=True,point_conv=False,scale_wave=False):
        super(get_model, self).__init__()
        
        self.entropy_mode =entropy_mode
        self.knn_fixed = knn_fixed
        self.point_conv = point_conv
        self.scale_wave = scale_wave
        
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3) 
        
        if self.knn_fixed:
            if scale_wave==False:
                self.inter_prediction_v2 = PointCloudFeaturePredictor(in_channels=64, K=32,point_conv=self.point_conv)
                
                self.CrossPointTransformer = CrossPointTransformer_v2(Cin=64, Cout=64, k=32)
                
            elif scale_wave==True:
                self.inter_prediction_v2 = PointCloudFeaturePredictor(in_channels=64, K=32,point_conv=self.point_conv)
                self.inter_prediction_v2_1 = PointCloudFeaturePredictor(in_channels=64, K=32,point_conv=self.point_conv)
                self.inter_prediction_v2_2 = PointCloudFeaturePredictor(in_channels=64, K=32,point_conv=self.point_conv)
                self.inter_prediction_v2_3 = PointCloudFeaturePredictor(in_channels=64, K=32,point_conv=self.point_conv)
                
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
            self.MotionBitEstimator = BitEstimator(48, 3)
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
            self.contextual_hyper_prior_encoder = Point_contextual_hyper_prior_encoder(channels=64,channels_latent=channels)
            self.contextual_hyper_prior_decoder =Point_contextual_hyper_prior_decoder(channels=64 , block_layers=3)
            
            self.temporal_prior_encoder = Point_TemporalPriorEncoder(channels=64,block_layers=3)
            
            self.contextual_entropy_parameter = Point_contextual_entropy_parameter(in_channels=128,channels=64)
            # self.dec1 = Context_UpsampleLayer(8, 64, 64, 3)
            # self.dec1_v2 = Context_UpsampleLayer(channels, 64, 64, 3)
            self.dec1_v2 = UpsampleLayer(channels, 64, 64, 3)
            
            self.dec2 = UpsampleLayer(64, 32, 32, 3)
            self.dec3 = UpsampleLayer(32, 16, 16, 3)

            # self.BitEstimator = BitEstimator(channels, 3)
            # self.MotionBitEstimator = BitEstimator(48, 3)
            self.crit = torch.nn.BCEWithLogitsLoss()
            
            

    def forward(self, f1, f2, device, epoch=99999,idx=0):
        
        # print("self.training===:",self.training)

        num_points = f2.C.size(0)#获得点云的点数

        ys1, ys2 = [f1, 0, 0, 0, 0], [f2, 0, 0, 0, 0]

        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]

        # feature extraction
        #第一帧两次下采样 
        
        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        torch.cuda.synchronize()
        start = time.time()
        ys1[1] = self.enc1(ys1[0])
        ys1[2] = self.enc2(ys1[1])
        #第二帧两次下采样
        ys2[1] = self.enc1(ys2[0])
        ys2[2] = self.enc2(ys2[1])
        
        torch.cuda.synchronize()
        end = time.time()

        # print(f"Execution time down: {end - start:.4f} seconds")
        # inter prediction
        if self.knn_fixed:
            if self.scale_wave==False:
                torch.cuda.synchronize()
                start = time.time()
                predicted_point2, residual ,_,residual_2 = self.inter_prediction_v2(ys1[2], ys2[2])
                torch.cuda.synchronize()
                end = time.time()
                # print(f"Execution time Bi-FMT: {end - start:.4f} seconds")
                
            if self.scale_wave==True:
                if idx==0:
                    predicted_point2, residual ,_= self.inter_prediction_v2(ys1[2], ys2[2])
                elif idx==1:
                    predicted_point2, residual ,_= self.inter_prediction_v2_1(ys1[2], ys2[2])
                elif idx==2:
                    predicted_point2, residual ,_= self.inter_prediction_v2_2(ys1[2], ys2[2])
                elif idx==3:
                    predicted_point2, residual ,_= self.inter_prediction_v2_3(ys1[2], ys2[2])
                
        else:
        
            residual, predicted_point2, quant_motion ,motion_hat = self.inter_prediction(ys1[2], ys2[2], stride=4)
            
            # end_event.record()
            # torch.cuda.synchronize()  # Wait for the events to be recorded!
            # elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f'encode motion predict: {elapsed_time_ms:.1f}ms')  # Elapsed: 212.1ms
            
            

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
            
            
            end_event.record()
            torch.cuda.synchronize()  # Wait for the events to be recorded!
            elapsed_time_ms = start_event.elapsed_time(end_event)
            # print(f'encode Elapsed: {elapsed_time_ms:.1f}ms')  # Elapsed: 212.1ms
            
            start_event.record()
            
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
         
        
        elif self.entropy_mode=='context_cond':
            
           
            torch.cuda.synchronize()
            start = time.time()
            #根据要编码的残差和解码的motion,来得到上下文
            context1 = predicted_point2 #由隐式motion估计空间估计出的上下文先验
            
            ys2[4] = self.contextual_encoder_v2(ys2[2],context1)#对齐下采样得到要编码的隐式特征(下采样1次)
            torch.cuda.synchronize()
            end = time.time()
            # print(f"Execution time contextual_encoder: {end - start:.4f} seconds")
            # print(f"输入hyper min/max/mean/std: {ys2[4].F.min().item():.4f}/{ys2[4].F.max().item():.4f}/{ys2[4].F.mean().item():.4f}/{ys2[4].F.std().item():.4f}")

            z = self.contextual_hyper_prior_encoder(ys2[4])#下采样一次
            # print(f"输出 z.F min/max/mean/std: {z.F.min().item():.4f}/{z.F.max().item():.4f}/{z.F.mean().item():.4f}/{z.F.std().item():.4f}")
            #对z做量化
            # z_q = ME.SparseTensor(quant(z.F, training=self.training),coordinate_map_key=z.coordinate_map_key,coordinate_manager=z.coordinate_manager)
            z_q = ME.SparseTensor(
                quant(z.F , training=self.training),
                coordinate_map_key=z.coordinate_map_key,
                coordinate_manager=z.coordinate_manager
            )
            
            # print(f"z_q.F min/max: {z_q.F.min().item():.4f}/{z_q.F.max().item():.4f}")
            
            
            z_hat = z_q
            # 对z做解码，得到编码y_latent的均值和方差
            y_coor = ME.SparseTensor(torch.ones([ys2[4].size()[0], 1], dtype=torch.float32, device=ys2[4].device),ys2[4].C, tensor_stride=ys2[4].tensor_stride)
            # z_hat.F =z_hat.F*0
            # print(".................................")
            hierarchical_params    = self.contextual_hyper_prior_decoder(z_hat, y_coor)#上采样一次
            
            # print(" hierarchical_params:", hierarchical_params)
            
            # print(f"输出 z_hat min/max/mean/std: {z_hat.F.min().item():.4f}/{z_hat.F.max().item():.4f}/{z_hat.F.mean().item():.4f}/{z_hat.F.std().item():.4f}")

            
            temporal_params = self.temporal_prior_encoder(context1=context1)#下采样一次，维度和ys2[4]保持一致       
            hierarchical_params = sort_by_coor_sum(hierarchical_params)
            temporal_params     = sort_by_coor_sum(temporal_params )
            if not torch.equal(hierarchical_params.C, temporal_params.C):
                raise ValueError("Temporal context and hyper-prior must have the same coordinates")
            
            cat_params = torch.concat([hierarchical_params.F,temporal_params.F],dim=1)
            # print("cat_params , ",cat_params.shape)
            new_params = ME.SparseTensor( cat_params, coordinates=temporal_params.C, coordinate_manager=temporal_params.coordinate_manager,tensor_stride=temporal_params.tensor_stride, device=temporal_params.device)
            gaussian_params        =  self.contextual_entropy_parameter(new_params)
            scales_hat , means_hat = gaussian_params.F.chunk(2, 1)
            
            # print("====>scales_hat",scales_hat.min().item(),scales_hat.max().item())
            # print("====>scales_hat",scales_hat)
            
#             print("scales_hat , means_hat:",scales_hat.shape, means_hat.shape)
            
            # print(f"y.F min/max/mean/std: {ys2[4].F.min().item():.4f}/{ys2[4].F.max().item():.4f}/{ys2[4].F.mean().item():.4f}/{ys2[4].F.std().item():.4f}")
  
            #对要编码的y_latent做量化
            ys2[4] =sort_by_coor_sum(ys2[4])
            y_q = quant(ys2[4].F.unsqueeze(0) , training=self.training)
            # print(f"y_q.F min/max: {y_q.min().item():.4f}/{y_q.max().item():.4f}")
            #量化后的稀疏卷积
            # print("y_q:",y_q.shape)
            y2_recon = ME.SparseTensor(features=y_q.squeeze(0),coordinates=ys2[4].C,tensor_stride=ys2[4].tensor_stride,device=ys2[4].device)
            
            z_for_bit =z_q.F.unsqueeze(0)
            
#             print("z_for_bit:",z_for_bit.shape)
#             print("z_forbit",z_for_bit)
            total_bits_z, _ = self.get_z_bits_probs(z_for_bit, self.bit_estimator_z)
            # print(f"total_bits_z: {total_bits_z.item():.4f}")
#             print("total_bits_z:",total_bits_z)
            #y码流：
            y_for_bit = y_q
            # print("y_for_bit, scales_hat:",y_for_bit.shape, scales_hat.shape)
#             print("y_for_bit:",y_for_bit.shape)
            total_bits_y, _ = self.get_y_bits_probs(y_for_bit, scales_hat)
            bits = total_bits_z+total_bits_y
            bpp_y = total_bits_y/num_points
            bpp_z = total_bits_z/num_points
            motion_bpp = 0
            
#             print("bpp_y , bpp_z:",bpp_y, bpp_z)
            
            
            #开始对量化后的y_q做decoder
            
            # print("context1:",context1.shape)
            #开始的版本
            torch.cuda.synchronize()
            start = time.time()
            out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1_v2(y2_recon, ys2[2], True, residual=context1,context2=None,f1_ref=None)
            
            # print("1 out2[0]",out2[0].shape)
            # torch.cuda.synchronize()
            # start = time.time()
            out2[0] = self.CrossPointTransformer(ref=ys1[2], cur=out2[0])
            
            
            residual_refine =ys2[2].F- out2[0].F
            
            # print("residual",residual.shape,residual_refine.shape ,residual_2.shape)
            # compare_three_residual_distributions(residual_A=residual.F, residual_B=residual_refine,residual_C=residual_2  )
            
            
            # print("2 out2[0]",out2[0].shape)
            torch.cuda.synchronize()
            end = time.time()

            # print(f"Execution time up1 + CTR: {end - start:.4f} seconds")
            #修改后版本
            # out2[0], out_cls2[0], target2[0], keep2[0] = self.dec1_v2(y2_recon, ys2[2], True, residual=None,context=context1)
           
            bpp = bits/ num_points
            
        torch.cuda.synchronize()
        start = time.time()
        out2[1], out_cls2[1], target2[1], keep2[1] = self.dec2(out2[0], ys2[1], True, 1 if self.training else 1)
        out2[2], out_cls2[2], target2[2], keep2[2] = self.dec3(out2[1], ys2[0], True, 1 if self.training else 1)
        
        torch.cuda.synchronize()
        end = time.time()

        # print(f"Execution time up2+3: {end - start:.4f} seconds")
        # print(f'decode Elapsed: {elapsed_time_ms:.1f}ms')  # 
        
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