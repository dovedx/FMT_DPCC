import torch
import torch.nn as nn
import MinkowskiEngine as ME
from models.model_utils import *
from models.prior_module import*
from models.flow_loss import*

# from pytorch3d.ops import knn_points
from models.pointconv_util import*
from GPCC.gpcc_wrapper import *


import sys
sys.path.append('../')
from PointPWC.models import PointConvSceneFlowPWC8192selfglobalPointConv as PointConvSceneFlow

from PointPWC.models import multiScaleChamferSmoothCurvature

from entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from entropy_models.video_entropy_models import EntropyCoder






class get_model(nn.Module):
    def __init__(self, channels=8 ,training=False):
        super(get_model, self).__init__()
        self.enc1 = DownsampleLayer(1, 16, 32, 3)
        self.enc2 = DownsampleLayer(32, 32, 64, 3) 
        # self.inter_prediction = inter_prediction(64, 64, 48)
        # self.motion_prediction = motion_prediction(64,64,8)
        #点云光流网络相关
        self.optical_point_flow = PointConvSceneFlow()
#         self.flow_loss_dict = {
#                                 'loss_type': 'unsup_l1',
#                                 'w_data': [0.75],
#                                 'w_smoothness': [0.25],
#                                 'smoothness_loss_params': {
#                                     'w_knn': 3.0,
#                                     'knn_loss_params': {
#                                         'k': 16,
#                                         'radius': 0.25,
#                                         'loss_norm': 1
#                                     }
#                                 },
#                                 'chamfer_loss_params': {
#                                     'loss_norm': 2,
#                                     'k': 1
#                                 }
#                             }
#         smoothness_loss_params={'w_knn': 3.0, 'knn_loss_params': {'k': 16,'radius': 0.25,'loss_norm': 1}}
                                                               
#         chamfer_loss_params={'loss_norm': 2,'k': 1}
                  
#         self.pointflow_loss = UnSupervisedL1Loss(w_data=[0.75] , w_smoothness=[0.25], smoothness_loss_params=smoothness_loss_params,chamfer_loss_params=chamfer_loss_params)

        #分类_loss
        self.crit_bce = torch.nn.BCEWithLogitsLoss()
        
        
        self.context_encoder = DownsampleLayer(64, 64, 32, 3)
        self.context_enc4 = ME.MinkowskiConvolution(in_channels=32, out_channels=channels, kernel_size=3, stride=1, bias=True, dimension=3)

        # self.BitEstimator = BitEstimator(channels, 3)
        # self.MotionBitEstimator = BitEstimator(48, 3)
        
        self.entropy_coder = None

        self.bit_estimator_z = BitEstimator(channel=64,dimension=3)
        self.bit_estimator_z_mv = BitEstimator(channel=32,dimension=3)
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
        
        
        
        self.mv_enc_scale = nn.Parameter(torch.ones((4, 1,32)))
        self.mv_dec_scale = nn.Parameter(torch.ones((4, 1,32)))
    
    def flow_warp(self, f_ref, point_flow):
        
        f_warp = f_ref+point_flow
        
        return f_warp
    
   

    def pointWarping_feat(self,f_ref,f_cur_C,point_flow):
        
        # N,C = f_cur.F.shape
        
        # print("f_ref,f_cur,point_flow:",f_ref.shape,f_cur.shape,point_flow.shape)
        # f_ref,f_cur,point_flow: torch.Size([3173, 64]) torch.Size([3169, 64]) torch.Size([3173, 3])
        xyz1_to_2 =f_ref.C[:,1:] + point_flow.F
        
        # print(f_cur.C[:,1:].unsqueeze(0).shape ,xyz1_to_2.unsqueeze(0).shape)
        
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
        # print("warp_f_ref:",warp_f_ref.shape)
        # warp_f_ref: torch.Size([1, 3169, 64])
        
        return warp_f_ref
        
    
        
    def motion_compensation(self, f_ref,f_cur_C, point_flow):
        f_ref_warp = self.flow_warp(f_ref.C[:,1:] , point_flow.F)
        
        # print("===>f_ref:",f_ref.F.shape)
        ref_feature = self.feature_extractor(f_ref)#对参考特征做一次特征提取，此处是为了兼容多帧参考
        
        # print("===ref_feature:",ref_feature.shape)

        #对ref_feature做多尺度的下采样

        ref_feature1, ref_feature2, ref_feature3 = self.multi_scale_feature_extractor(ref_feature)
        #原始尺度，2被才采样尺度，4倍下采样尺度
        # print("ref_feature1, ref_feature2, ref_feature3",ref_feature1.shape, ref_feature2.shape, ref_feature3.shape)
        # ref_feature1, ref_feature2, ref_feature3 torch.Size([3173, 64]) torch.Size([776, 64]) torch.Size([184, 64])
        #对点云的point flow 采样3个尺度，做多尺度的特征提取
        point_flow2 , point_flow3 = self.point_down(point_flow)
        # print("point_flow2 , point_flow3",point_flow2.shape , point_flow3.shape)
        # point_flow2 , point_flow3 torch.Size([776, 3]) torch.Size([184, 3])

        context1 = self.pointWarping_feat(ref_feature1,f_cur_C[:,1:], point_flow)

        Feature = torch.ones_like(f_cur_C[:, :1].float())
        tensor_temp = ME.SparseTensor( Feature , coordinates=f_cur_C,tensor_stride=f_ref.tensor_stride,device=f_ref.device)
        f_cur2_temp =self.down_C(tensor_temp)
        # f_cur2 = torch.from_numpy(f_cur.C[:,1:].cpu().numpy()//4)
        # f_cur2   =   torch.unique(f_cur2 ,dim=0).cuda()
      
        context2 = self.pointWarping_feat(ref_feature2,f_cur2_temp.C[:,1:], point_flow2)
        
        Feature = torch.ones_like(f_cur2_temp.C[:, :1].float())
        tensor_temp = ME.SparseTensor( Feature,coordinates=f_cur2_temp.C,tensor_stride=f_cur2_temp.tensor_stride , device=f_ref.device)
        f_cur3 =self.down_C(tensor_temp)
        # f_cur3   =   torch.from_numpy(f_cur2.cpu().numpy()//4)
        # f_cur3   =   torch.unique(f_cur3 ,dim=0).cuda()
        
        context3 = self.pointWarping_feat(ref_feature3,f_cur3.C[:,1:], point_flow3)
        
        # print("context1,context2,context3",context1.shape,context2.shape,context3.shape)
        # context1,context2,context3 torch.Size([1, 12759, 64]) torch.Size([1, 3169, 64]) torch.Size([1, 771, 64])
        
        context1_sparse =  ME.SparseTensor(features=context1[0], coordinates=f_cur_C,tensor_stride=f_ref.tensor_stride, device=f_ref.device)
        context2_sparse =  ME.SparseTensor(features=context2[0], coordinates=f_cur2_temp.C,tensor_stride=f_cur2_temp.tensor_stride, device=f_cur2_temp.device)
        context3_sparse =  ME.SparseTensor(features=context3[0], coordinates=f_cur3.C,tensor_stride=f_cur3.tensor_stride, device=f_cur3.device)

        context1, context2, context3 = self.context_fusion_net(context1_sparse, context2_sparse, context3_sparse)
        # print("context1,context2,context3",context1.shape,context2.shape,context3.shape)

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
        # print("=====>z:",z.shape)[649,32]
        
        prob = bit_estimator(z + 0.5) - bit_estimator(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob
    
    
    def update(self, force=False):
        self.entropy_coder = EntropyCoder()
        self.bit_estimator_z_mv.update(force=force, entropy_coder=self.entropy_coder)
        self.bit_estimator_z.update(force=force, entropy_coder=self.entropy_coder)
        self.gaussian_encoder.update(force=force, entropy_coder=self.entropy_coder)
    
    def get_scales(self,idx):
        mv_enc_scale = self.mv_enc_scale[idx]
        mv_dec_scale = self.mv_dec_scale[idx]
        
        return mv_enc_scale,mv_dec_scale
    
    def forward(self,f_ref, f_cur,coding_mode, device, epoch=99999 ,training=None,idx=None):
        if idx!=None:
            mv_enc_scale,mv_dec_scale = self.get_scales(idx)
        else:
            mv_enc_scale=1
            mv_dec_scale=1
            
            
        if training==False:
            self.training=training

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
        
        
        # print("完成第一阶段的下采样",ys_ref[2].C.shape,ys_cur[2].C.shape)
        
        # return
        #两次下采样后得到低分辨率的点云和点云特征
        
        # motion,quant_compressed_motion=self.motion_prediction(ys1[2], ys2[2], stride=4)
        # print("+++>",ys_ref[2].C[:,1:].shape,ys_cur[2].C[:,1:].unsqueeze(0).shape)
        # point_mv,fps_pc1_idxs, _, _, _ = self.point_flow(ys_ref[2].C[0:10000,1:].unsqueeze(0).to(torch.float32),ys_cur[2].C[0:10020,1:].unsqueeze(0).to(torch.float32),\
        #                              ys_ref[2].C[0:10000,1:].unsqueeze(0).to(torch.float32),ys_cur[2].C[0:10020,1:].unsqueeze(0).to(torch.float32))#[Nx3]
        
        
        # print("===input flownet shape:",ys_ref[2].C[:,1:].shape,ys_cur[2].C[:,1:].shape)
        # ===input flownet shape: torch.Size([19467, 3]) torch.Size([19687, 3])
        
        ys_ref[2]=sort_by_coor_sum(ys_ref[2])
        ys_cur[2]=sort_by_coor_sum(ys_cur[2])
        #这里要对输入数据做预处理
        
         
        # 保存下下采样的点云
        # write_ply_data("./midde_result/ys_ref.ply", ys_ref[2].C[:,1:].cpu().numpy())
        # write_ply_data("./midde_result/ys_cur.ply", ys_cur[2].C[:,1:].cpu().numpy())
        # print("write finished ...")
        
        pos1 = ys_ref[2].C[:,1:].unsqueeze(0).contiguous().cuda().to(torch.float32)
        pos2 = ys_cur[2].C[:,1:].unsqueeze(0).contiguous().cuda().to(torch.float32)
        norm1 = pos1
        norm2 = pos2
        
        # print("pos1,pos2:",pos1.shape, pos2.shape)
        
        
        
        point_mvs,fps_pc1_idxs, _, pc1, pc2 = self.optical_point_flow(pos1, pos2, norm1, norm2)#[Nx3]
        
        # print("pred_flows shape:",point_mvs[0].shape)
        # print("point_mvs[0]:",point_mvs[0])
        
        
        # 根据coordinate以及光流网络点云的初步光流
        # print("第二阶段 光流估计:",point_mv[0].shape)第二阶段 光流估计: torch.Size([1, 3, 3173])
        point_mv_F = point_mvs[0][0].transpose(0,1)
        #将点云光流转为稀疏tensor
        # print("point_mv_F",point_mv_F.shape)
        point_mv = ME.SparseTensor(point_mv_F, coordinates=ys_ref[2].C, tensor_stride=ys_ref[2].tensor_stride)
        #得到motion之后，可以指导feature生成context
        print("===>point_mv:",point_mv.shape,point_mv)
        
            
        #开始编码点云的mv
        mv_y = self.mv_encoder(point_mv) #对mv做2次稀疏下采样
        # print("===>mv_y:",mv_y.F.shape)# ===>mv_y: torch.Size([776, 32])
        # print("===>",mv_y.shape ,  mv_enc_scale.shape)
        if idx!=None:
            mv_y =mv_y * mv_enc_scale
        mv_z = self.mv_prior_encoder(mv_y)#对mv_y做了一次下采样
        # print("prior param: mv_z",mv_z.F .shape) #prior param: mv_z torch.Size([184, 16])
        # mv_z_q = self.quant(mv_z.F)
        # mv_z_q = quant(mv_z.F, training=self.training)#对prior的参数做量化
        # print("量化模式:",self.training)
        # print("before quant mv_z:",mv_z.F)
        mv_z_q=ME.SparseTensor(quant(mv_z.F, training=self.training),
                                                  coordinate_map_key=mv_z.coordinate_map_key,
                                                  coordinate_manager=mv_z.coordinate_manager)#量化光流特征
        
        # print("after quant mv_z_q",mv_z_q.F)
        mv_params = self.mv_prior_decoder(mv_z_q , mv_y )##对prior做一次上采样，并输出用于估计scale和mean，mv_y是否可以直接用，存在疑问
        mv_params_new = feature_match_new(mv_params,mv_y.C)
        
        # print("mv_y",mv_y.C[0:20,:])
        # print("mv_params",mv_params.C[0:20,:])
        

        # print("decode mv_params:",mv_params.F.shape)decode mv_params: torch.Size([776, 64])
        mv_scales_hat, mv_means_hat = mv_params_new.chunk(2, 1)
        # print("mv_scales_hat, mv_means_hat:",mv_scales_hat.shape, mv_means_hat.shape)
        # mv_scales_hat, mv_means_hat: torch.Size([776, 32]) torch.Size([776, 32])
        
        # print("========>mv_y.F:",mv_y.F)
        
        
        mv_y_res = mv_y.F - mv_means_hat #稀疏卷积之间的减法 
        # mv_y_q = quant(mv_y_res.F,training=self.training)#对光流特征做量化
        
        # print("before quant mv_y:",mv_y_res)
        mv_y_q   = quant(mv_y_res, training=self.training)
        # print("after quant mv_y:",mv_y_q)
        
        
        # print("mv_y_q",mv_y_q)
        mv_y_hat = mv_y_q + mv_means_hat
        if idx!=None:
            mv_y_hat = mv_y_hat * mv_dec_scale
        
        # mv_y_hat = ME.SparseTensor( mv_y_hat,coordinate_map_key=mv_y.coordinate_map_key,coordinate_manager=mv_y.coordinate_manager) 
        
        mv_y_hat = ME.SparseTensor( mv_y_hat,coordinates=mv_y.C,tensor_stride=mv_y.tensor_stride) 
                                                                                              
        mv_hat = self.mv_decoder(mv_y_hat,point_mv)#对特征做2次上采样得到重构光流
        
        match_mv_hat_F = feature_match_new(mv_hat,point_mv.C)
        
        mv_hat = ME.SparseTensor( match_mv_hat_F,coordinates=point_mv.C,tensor_stride=point_mv.tensor_stride)
        
        # print("===>mv_hat",mv_hat.F.shape)#===>mv_hat torch.Size([3173, 3])


        if coding_mode =='align':
            f_ref_warp = self.flow_warp(ys_ref[2].C[:,1:] , match_mv_hat_F)#这里就是编码光流，将f_ref做warp，和目标点云做loss。
            
            # print("f_ref_warp:",f_ref_warp.shape)
            # loss_flow = self.pointflow_loss(f_ref_warp,f_cur[2].C)#这里的loss函数得想一下
            # pc1=ys_ref[2].C[:,1:].unsqueeze(0).permute(0, 2, 1).float().contiguous()
            # pc2=ys_cur[2].C[:,1:].unsqueeze(0).permute(0, 2, 1).float().contiguous()
            
            
            
            loss_flow,chamfer_loss, curvature_loss, smoothness_loss = multiScaleChamferSmoothCurvature(pc1,pc2,point_mvs,f_ref_warp,match_mv_hat_F)
            #这里是无监督的光流loss

            total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_q, mv_scales_hat)
            total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_q.F, self.bit_estimator_z_mv)
            bpp_mv_y = total_bits_mv_y / num_points
            bpp_mv_z = total_bits_mv_z / num_points
            
            # print("bits:", total_bits_mv_y,total_bits_mv_z)
            
            mv_bpp = bpp_mv_y + bpp_mv_z
           
            # print("flow stage finished .....")
            
            return{
                "bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_mv":   mv_bpp,
                "loss_flow":loss_flow/num_points,
            }
        
        # print("start context ....")
        # print("f_ref[2],  mv_hat:",ys_ref[2].F.shape,  mv_hat.F.shape)
        context1, context2, context3, warp_frame= self.motion_compensation(ys_ref[2], ys_cur[2].C, mv_hat)
        #下采样域的参考帧和光流，生成多尺度的上下文信息
        
        if coding_mode =='align2':
            #这里的训练和测试是有区别的
            out_pruned, out_cls, target, keep = self.context1_to_point(context1,target_label=ys_cur[2],adaptive=1,rho=1,training=self.training , lossless=None)#直接从context中恢复出f_cur[2]的坐标
            # loss_bce = self.crit_bce(recon_xyz,f_cur[2].C)
            # print("out_pruned:",out_pruned.shape,out_cls.shape,target.shape)

            loss_bce = self.crit_bce(out_cls.F.squeeze(),target.type(out_cls.F.dtype).to(device))

            total_bits_mv_y, _ = self.get_y_bits_probs(mv_y_q, mv_scales_hat)
            total_bits_mv_z, _ = self.get_z_bits_probs(mv_z_q.F.unsqueeze(0), self.bit_estimator_z_mv)
            bpp_mv_y = total_bits_mv_y / num_points
            bpp_mv_z = total_bits_mv_z / num_points
            mv_bpp   = bpp_mv_y + bpp_mv_z
            
            # print("bits:", total_bits_mv_y,total_bits_mv_z)
            # print("flow stage2 finished .....")
            return{
                "bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_mv":mv_bpp,
                "loss_bce":loss_bce,
            }


        #y拥有了3个尺度的参考帧上下文信息，就可以对当前的点云特征做上下文编码
        # print("ys_cur[2],context1, context2, context3===>",ys_cur[2].shape ,context1.shape, context2.shape, context3.shape)
        # ys_cur[2],context1, context2, context3===> torch.Size([12759, 64]) torch.Size([12759, 64]) torch.Size([3169, 64]) torch.Size([771, 64])
        y , context1_new,context2_new,context3_new = self.contextual_encoder(ys_cur[2],context1, context2, context3)
        # print("y===>",y.shape)
        # y===> torch.Size([49, 64])
        #对特征进一步下采样编码
        z = self.contextual_hyper_prior_encoder(y)
        # z_q = self.quant(z)
        # print("z==>",z.shape)
        # z==> torch.Size([17, 64])
        # print("before quant z:",z.F)
        z_q  = ME.SparseTensor(quant(z.F, training=self.training),
                                                  coordinate_map_key=z.coordinate_map_key,
                                                  coordinate_manager=z.coordinate_manager)
        z_hat = z_q
        # print("after quant z:",z_q.F)
        hierarchical_params     = self.contextual_hyper_prior_decoder(z_hat , y)#第一个疑问点，y在解码端可不可以得到
        # hierarchical_params_new = feature_match(hierarchical_params,y)
        hierarchical_params_new = feature_match_new(hierarchical_params,y.C)
        # print("================>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        temporal_params = self.temporal_prior_encoder(context1,context2,context3)
        # print("temporal_params:",temporal_params.shape)
        params_feature = temporal_params.F + hierarchical_params_new
        params = ME.SparseTensor(params_feature,coordinates=y.C,tensor_stride=y.tensor_stride)
        # print("params+===>",params.shape)
        gaussian_params = self.contextual_entropy_parameter(params)
        # print("gaussian_params:",gaussian_params.shape)
        scales_hat, means_hat = gaussian_params.F.chunk(2, 1)
        # print("===>scales_hat, means_hat:",scales_hat.shape, means_hat.shape)
        # print("y",y)
        y_res = y.F - means_hat
        # print("before quant y:",y_res)
        y_q =  quant(y_res , training=self.training)
        # print("before quant y_q:",y_q)
        
        y_hat = y_q + means_hat

        # y2_recon = ME.SparseTensor(y_hat.squeeze(0), coordinate_map_key=y.coordinate_map_key,coordinate_manager=y.coordinate_manager, device=y.device)
        y2_recon = ME.SparseTensor(features=y_hat,coordinates=y.C,tensor_stride=y.tensor_stride,device=y.device)
        # print("y2_recon===>",y2_recon.shape)

        #上采样
        # y2_recon = self.contextual_decoder(y2_recon, context2, context3)#这一步要恢复到的分辨率为N/2/2
        #多次上采样恢复到原始点云的分辨率x,targetl_label,adaptive,rho, context2, context3
        out2[0], out_cls2[0], targets2[0], keep2[0] =  self.contextual_decoder(x=y2_recon, target_label=ys_cur[2], adaptive=True,rho=1,context2=context2,context3=context3)
        
        # print("===>out2[0]",out2[0].shape)
        out2[1], out_cls2[1], targets2[1], keep2[1] = self.dec2(out2[0], ys_cur[1], True, 1 if self.training else 1)
        # print("out2[1]==>",out2[1].shape , targets2[1].shape ,out_cls2[1].shape,ys_cur[1].shape)
        out2[2], out_cls2[2], targets2[2], keep2[2] = self.dec3(out2[1], ys_cur[0], True, 1 if self.training else 1)
        # print("out2[2]===>",out2[2].shape ,targets2[2].shape ,out_cls2[2].shape,ys_cur[0].shape)

        # recon_xyz =  self.recon_generation_net(recon_image_feature, context1)

        #对recon_xyz做上采样2次，得到原始分辨率的点云

        # loss_bce = self.crit_bce(recon_xyz,f_cur[2].C)

        #计算编码mv和ys[2]所使用的bits
        y_for_bit = y_q
        mv_y_for_bit = mv_y_q
        z_for_bit = z_q.F.unsqueeze(0)
        mv_z_for_bit = mv_z_q.F.unsqueeze(0)
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
        
        # print("bpp:===>",bpp.item() , bpp_y.item() , bpp_z.item() , bpp_mv_y.item() ,bpp_mv_z.item())
        
        distortion = 0
        for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
            curr_loss = self.crit_bce(out_cl.F.squeeze(),target.type(out_cl.F.dtype).to(device))                               
            distortion += curr_loss / num_points
        
        loss_d = distortion
        
        recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C,tensor_stride=out2[-1].tensor_stride,device=out2[-1].device )
        
        ys_cur[4]=y
        
        return {
            "bpp_mv_y": bpp_mv_y,
            "bpp_mv_z": bpp_mv_z,
            "bpp_y": bpp_y,
            "bpp_z": bpp_z,
            "bpp": bpp,
            "total_bit":total_bits ,
            "mv": mv_hat,
            "out2":out2,
            "out_cls2":out_cls2,
            "targets2":targets2,
            'loss_d':loss_d,
            'recon_point':recon_f2,
            'ys2':ys_cur
            
        }


