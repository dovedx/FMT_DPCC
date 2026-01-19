import argparse
import importlib
import logging
import sys
import os

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Test Script')
    parser.add_argument('--model', type=str, default='RA_DDPCC_geo_cond_bn_test')
    parser.add_argument('--lossless_model', type=str, default='DDPCC_lossless_coder')
    parser.add_argument('--log_name', type=str, default='aaa')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device [default: 0]')
    parser.add_argument('--channels', default=8, type=int)
    parser.add_argument('--ckpt_dir', type=str,default='./ddpcc_ckpts')
                        
    parser.add_argument('--pcgcv2_ckpt_dir', type=str,default='./pcgcv2_ckpts')
                        
    parser.add_argument('--frame_count', type=int, default=100, help='number of frames to be coded')
    parser.add_argument('--results_dir', type=str, default='results', help='directory to store results (in csv format)')
    parser.add_argument('--tmp_dir', type=str, default='tmp')
    parser.add_argument('--overwrite', type=bool, default=False, help='overwrite the bitstream of previous frame')
    parser.add_argument('--dataset_dir', type=str, default='/home/zhaoxudong/Owlii_10bit')
    parser.add_argument('--resolution', type=int, default=1023, help='')
    parser.add_argument('--scaling_factor', type=int, default=1, help='')
    
    parser.add_argument('--entropy_mode', default="cond", type=str)
    parser.add_argument('--knn_fixed', type=str2bool, default=True,help="True or False")
    parser.add_argument('--point_conv',type=str2bool, default=False,help="True or False")
    parser.add_argument('--fuse_conv',type=str2bool, default=False,help="True or False")
    return parser.parse_args()

args = parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, './PCGCv2'))

from models.model_utils import index_points
# from dataset_owlii import *
from dataset_lossy_RA import*
# from models.entropy_coding import *
from GPCC.gpcc_wrapper import *
# from PCGCv2.eval import test_one_frame
import pandas as pd
import collections, math
from pytorch3d.ops import knn_points
from SparsePCC.eval import test_one_frame_lossy,test_one_frame_lossless



# f_file = open("./record_txt/text_mpeg_cond_lambda10_1023_context_baseonddpcc.txt","w")
# f_file = open("./record_txt/text_mpeg_lambda15_ddpcc_cond_context_prior.txt","w")
# f_file = open("./record_txt_RA/test_owlii_lambda30_RA_cond_v3_0epoch_nobn_debug.txt","w")
# f_file = open("./record_txt_RA/test_owlii_lambda5_RA_cond_bn_3epoch_fuseconv.txt","w")
# f_file = open("./record_txt_RA/test_owlii_lambda3_RA_cond_bn_3epoch_736_836_solider.txt","w")
f_file = open("./record_txt_RA/test_owlii_lambda15_RA_cond_nobn_v2_4epoch.txt","w")


def log_string(string):
    logger.info(string)
    print(string)


# def PSNR(pc1, pc2, n1):
#     pc1, pc2 = pc1.to(torch.float32), pc2.to(torch.float32)
#     dist1, knn1, _ = knn_points(pc1, pc2, K=4)  # neighbors of pc1 from pc2
#     dist2, knn2, _ = knn_points(pc2, pc1, K=4)  # neighbors of pc2 from pc1
#     mask1 = (dist1 == dist1[:, :, :1])
#     mask2 = (dist2 == dist2[:, :, :1])
#     dist = max(dist1[:, :, 0].mean(), dist2[:, :, 0].mean())  # dists from knn_points are squared dists
#     cd = max(dist1[:, :, 0].sqrt().mean(), dist2[:, :, 0].sqrt().mean())
#     d1_psnr = 10*math.log(3*1023*1023/dist)/math.log(10)
#     knn1_ = knn1.reshape(-1)
#     n1_src = (n1.unsqueeze(2).repeat(1, 1, 4, 1)*(mask1.unsqueeze(-1))).reshape(-1, 3)
#     n2 = torch.zeros_like(pc2.squeeze(0), dtype=torch.float64)
#     n2.index_add_(0, knn1_, n1_src)
#     n2 = n2.reshape(1, -1, 3)

#     n2_counter = torch.zeros(pc2.size()[1], dtype=torch.float32, device=pc2.device)
#     counter_knn1 = knn1.reshape(-1)
#     n1_counter_src = mask1.reshape(-1).to(torch.float32)
#     n2_counter.index_add_(0, counter_knn1, n1_counter_src)
#     n2_counter = n2_counter.reshape(1, -1, 1)
#     n2_counter += 0.00000001

#     n2 /= n2_counter

#     v2 = index_points(pc1, knn2) - pc2.unsqueeze(2)
#     n2_ = index_points(n1, knn2)
#     n21 = (n2_*(mask2.unsqueeze(-1))).sum(dim=2) / (mask2.sum(dim=-1, keepdim=True))
#     n2 += (n2_counter < 0.0001) * n21

#     d2_ = (((v2*n2_).sum(dim=-1).square()*mask2).sum(dim=-1)/mask2.sum(dim=-1)).mean()
#     v1 = index_points(pc2, knn1) - pc1.unsqueeze(2)
#     n1_ = index_points(n2, knn1)
#     d1_ = (((v1 * n1_).sum(dim=-1).square() * mask1).sum(dim=-1) / mask1.sum(dim=-1)).mean()
#     dist_ = max(d1_, d2_)
#     d2_psnr = 10*math.log(3*1023*1023/dist_)/math.log(10)
#     return d1_psnr, d2_psnr, cd.item()
def PSNR(pc1, pc2, n1,resolution=1023):
    pc1, pc2 = pc1.to(torch.float32), pc2.to(torch.float32)
    dist1, knn1, _ = knn_points(pc1, pc2, K=4)  # neighbors of pc1 from pc2
    dist2, knn2, _ = knn_points(pc2, pc1, K=4)  # neighbors of pc2 from pc1
    mask1 = (dist1 == dist1[:, :, :1])
    mask2 = (dist2 == dist2[:, :, :1])
    dist = max(dist1[:, :, 0].mean(), dist2[:, :, 0].mean())  # dists from knn_points are squared dists
    cd = max(dist1[:, :, 0].sqrt().mean(), dist2[:, :, 0].sqrt().mean())
    d1_psnr = 10*math.log(3*resolution*resolution/dist)/math.log(10)
    knn1_ = knn1.reshape(-1)
    n1_src = (n1.unsqueeze(2).repeat(1, 1, 4, 1)*(mask1.unsqueeze(-1))).reshape(-1, 3)
    n2 = torch.zeros_like(pc2.squeeze(0), dtype=torch.float64)
    n2.index_add_(0, knn1_, n1_src)
    n2 = n2.reshape(1, -1, 3)

    n2_counter = torch.zeros(pc2.size()[1], dtype=torch.float32, device=pc2.device)
    counter_knn1 = knn1.reshape(-1)
    n1_counter_src = mask1.reshape(-1).to(torch.float32)
    n2_counter.index_add_(0, counter_knn1, n1_counter_src)
    n2_counter = n2_counter.reshape(1, -1, 1)
    n2_counter += 0.00000001

    n2 /= n2_counter

    v2 = index_points(pc1, knn2) - pc2.unsqueeze(2)
    n2_ = index_points(n1, knn2)
    n21 = (n2_*(mask2.unsqueeze(-1))).sum(dim=2) / (mask2.sum(dim=-1, keepdim=True))
    n2 += (n2_counter < 0.0001) * n21

    d2_ = (((v2*n2_).sum(dim=-1).square()*mask2).sum(dim=-1)/mask2.sum(dim=-1)).mean()
    v1 = index_points(pc2, knn1) - pc1.unsqueeze(2)
    n1_ = index_points(n2, knn1)
    d1_ = (((v1 * n1_).sum(dim=-1).square() * mask1).sum(dim=-1) / mask1.sum(dim=-1)).mean()
    dist_ = max(d1_, d2_)
    d2_psnr = 10*math.log(3*resolution*resolution/dist_)/math.log(10)
    return d1_psnr, d2_psnr, cd.item()


if __name__ == '__main__':
    device = torch.device('cuda')
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('./%s.txt' % args.log_name)
    file_handler = logging.FileHandler('./%s.txt' % args.log_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    tmp_dir = args.tmp_dir
    # tmp_dir = './tmp_'+''.join(random.sample('0123456789', 10))
    tmp_dir_ = Path(tmp_dir)
    tmp_dir_.mkdir(exist_ok=True)
    results_dir = args.results_dir
    results_dir_ = Path(results_dir)
    results_dir_.mkdir(exist_ok=True)
    gpcc_bitstream_filename = os.path.join(tmp_dir, 'gpcc.bin')

    # load model
    log_string('PARAMETER ...')
    log_string(args)
    MODEL = importlib.import_module(args.model)
    # model = MODEL.get_model(channels=args.channels,entropy_mode=args.entropy_mode)
    model = MODEL.get_model(channels=args.channels ,entropy_mode=args.entropy_mode,knn_fixed=args.knn_fixed,point_conv = args.point_conv,fuse_conv=args.fuse_conv)
    model.eval()

    LOSSLESS_MODEL = importlib.import_module(args.lossless_model)
    lossless_model = LOSSLESS_MODEL.get_model()
    lossless_checkpoint = torch.load('./ddpcc_ckpts/lossless_coder.pth')
    old_paras = lossless_model.state_dict()
    new_state_dict = collections.OrderedDict()
    for k, v in lossless_checkpoint['model_state_dict'].items():
        k1 = k.replace('module.', '')
        if k1 in old_paras:
            new_state_dict[k1] = v
    old_paras.update(new_state_dict)
    lossless_model.load_state_dict(old_paras)
    lossless_model = lossless_model.to(device).eval()

    results = {
        'loot': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'cd': [],'num_points': [], 'exp_name': []},
        'redandblack': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'cd': [],'num_points': [], 'exp_name': []},
        'longdress': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'cd': [],'num_points': [], 'exp_name': []},
        'soldier': {'bpp': [], 'd1-psnr': [], 'd2-psnr': [], 'cd': [], 'num_points': [], 'exp_name': []}
    }
    '''
    start testing
    5: basketballplayer
    6: dancer
    8: exercise
    13: model
    '''
    # ckpts = {
    #     'r1_0.025bpp.pth': 'r1.pth',
    #     'r2_0.05bpp.pth': 'r2.pth',
    #     'r3_0.10bpp.pth': 'r3.pth',
    #     'r4_0.15bpp.pth': 'r4.pth',
    #     'r5_0.25bpp.pth': 'r5.pth',
    #     'r6_0.3bpp.pth': 'r6.pth',
    #     'r7_0.4bpp.pth': 'r7.pth',
    # }
    ckpts = {
        
        # 'r3_0.10bpp.pth': 'best_model_lambda3_600frame.pth',
        # 'r6_0.3bpp.pth': 'best_model_lambda10_600frame.pth',
        # 'r3_0.10bpp.pth': 'r1.pth',
        # 'r4_0.15bpp.pth': 'r2.pth',
        # 'r5_0.25bpp.pth': 'r3.pth',
        # 'r6_0.3bpp.pth': 'r4.pth',
        'r7_0.4bpp.pth': 'r5.pth'
    }
    #左边是P帧模型，右边是I帧模型
    with torch.no_grad():
        for pcgcv2_ckpt in ckpts:
            print("pcgcv2_ckpt====>",pcgcv2_ckpt)
            exp_name = str(ckpts[pcgcv2_ckpt]).split('.')
            exp_name = exp_name[0]
            # ddpcc_ckpt = os.path.join(args.ckpt_dir, ckpts[pcgcv2_ckpt])
            # ddpcc_ckpt ='./log/DDPCC_geo_cond/20241204_lambda10_600frame_cond2/checkpoint/best_model.pth'
            # ddpcc_ckpt ='./log/DDPCC_geo_cond/20241204_lambda10_600frame_cond2_basedonDDPC_navie/checkpoint/best_model.pth'
            # ddpcc_ckpt ='./log/DDPCC_geo_cond/20241204_lambda10_600frame_cond2_new_v3/checkpoint/best_model.pth'
            # ddpcc_ckpt ='./log_fuxian/DDPCC_geo/20241204_lambda10_600frame/checkpoint/best_model.pth'
            # ddpcc_ckpt = './ddpcc_ckpts/r4.pth'
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond/20241204_lambda30_300frame_fixedknn_v3_point_conv_k8_RA/checkpoint/2.pth"
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond/20241204_lambda5_300frame_fixedknn_v3_point_conv_k8_RA/checkpoint/2.pth"
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond/20241204_lambda8_300frame_fixedknn_v3_point_conv_k8_RA_v2/checkpoint/2.pth"
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda7_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_bn/checkpoint/2.pth"
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda30_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_no_bn/checkpoint/0.pth"
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda5_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_no_bn/checkpoint/2.pth"
            
            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda3_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_bn/checkpoint/3.pth"
            
            ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda15_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_no_bn/checkpoint/0.pth"


            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda30_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_no_bn_fuseconv/checkpoint/3.pth"

            # ddpcc_ckpt = "./log_RA_on_owlii/RA_DDPCC_geo_cond_bn/20241204_lambda1.5_300frame_fixedknn_v3_point_conv_k8_RA_contextcond_bn/checkpoint/2.pth"

            pcgcv2_ckpt = os.path.join(args.pcgcv2_ckpt_dir, pcgcv2_ckpt)
            
            
            checkpoint = torch.load(ddpcc_ckpt, map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            model = model.to(device).eval()
            for sequence in (0, 1, 2, 3):
            # for sequence in (3,2):
                dataset = Dataset(root_dir=args.dataset_dir, split=[sequence], type='test',scaling_factor=args.scaling_factor, format='ply')
                print("总数据为:",len(dataset))
                sequence_name = dataset.sequence_list[sequence]
                log_string('start testing sequence ' + sequence_name + ', rate point ' + ddpcc_ckpt)
                log_string('f bpp     d1PSNR  d2PSNR  numpoint')
                print('name bpp     d1PSNR  d2PSNR  numpoint',file=f_file)
                d1_psnr_sum = 0
                d2_psnr_sum = 0
                bpp_sum = 0
                bits_sum = 0
                num_points_sum = 0
                cd_sum = 0

                # encode the first frame
                xyz, point ,indices  = collate_pointcloud_fn([dataset[0]])
                f1 = ME.SparseTensor(features=point, coordinates=xyz, device=device)
                # bpp, d1psnr, d2psnr, f1 = test_one_frame(f1, pcgcv2_ckpt, os.path.join(tmp_dir,'PCGCv2'))
                bpp, d1psnr, d2psnr, f1 = test_one_frame_lossy(f1, ckptdir='./SparsePCC/ckpts/dense/epoch_last.pth', ckptdir_sr='./SparsePCC/ckpts/dense_1stage/epoch_last.pth', ckptdir_ae='./SparsePCC/ckpts/dense_slne/epoch_last.pth' , out_path='./SparsePCC/result/', scaling_factor=1.0, rho=1.0, res=1024,device=device)
                
                torch.cuda.empty_cache()
                                                                                       
                f1 = ME.SparseTensor(torch.ones_like(f1.F[:, :1]), coordinates=f1.C)
                log_string(str(0) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] +  str(num_points_sum) + '\n')
                print(f"{sequence_name},{0},{bpp},{d1psnr},{d2psnr},{num_points_sum}",file=f_file)
                
                bpp_sum += bpp
                d1_psnr_sum += d1psnr
                d2_psnr_sum += d2psnr
                num_points_sum += (f1.size()[0] * 1.0)
                bits_sum += (f1.size()[0] * bpp)
                
                #编码结束I帧

                for i in range(0, args.frame_count):
                    
                    if i==0:
                        out2, out_cls2, target2, keep2 = [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
                        current_f = f1
                        indeics=indices
                        
                        current_f, out2, out_cls2, target2, keep2, ddpcc_bpp = model(current_f=current_f,indices=indices, device=current_f.device)
                        print("完成第一帧的编码:",ddpcc_bpp)
                    if i>0:
                       
                        xyz, point,indices = collate_pointcloud_fn([dataset[i]])
                        current_f  = ME.SparseTensor(features=point, coordinates=xyz, device=device)#取出要编码的当前帧
                        num_points = current_f.size()[0]
                        
                        
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        start_event.record()
                        current_f_recon, out2, out_cls2, target2, keep2, ddpcc_bpp = model(current_f=current_f,indices=indices, device=current_f.device)
                        end_event.record()
                        torch.cuda.synchronize()  # Wait for the events to be recorded!
                        elapsed_time_ms = start_event.elapsed_time(end_event)
                        print(f'Elapsed: {elapsed_time_ms:.1f}ms')  # Elapsed: 212.1ms
                        
                        print("ddpcc_bpp:",ddpcc_bpp,indices)
                        
                        ddpcc_bpp  = ddpcc_bpp.item()
                        
                        ys2_4_C = (current_f_recon[4].C[:, 1:]//8).detach().cpu().numpy()
                        write_ply_data(os.path.join(tmp_dir, 'ys2_4.ply'), ys2_4_C)
                        gpcc_encode(os.path.join(tmp_dir, 'ys2_4.ply'), gpcc_bitstream_filename)
                        # ys2_2 = ME.SparseTensor(torch.ones_like(current_f_recon[2].F[:, :1]), coordinate_manager=current_f_recon[2].coordinate_manager, coordinate_map_key=current_f_recon[2].coordinate_map_key)
                        ys2_2 =  ME.SparseTensor(features=torch.ones_like(current_f_recon[2].F[:, :1]) , coordinates=current_f_recon[2].C ,  coordinate_manager=current_f_recon[2].coordinate_manager,tensor_stride =current_f_recon[2].tensor_stride, device=current_f_recon[2].device)

                        bits_ys2_2, quant_out2, cls, target = lossless_model.compressor(ys2_2, -1)
                        ys2_2_bpp = bits_ys2_2 / num_points
                        ys2_2_bpp = ys2_2_bpp.item()
                        

                        gpcc_bpp = os.path.getsize(gpcc_bitstream_filename) * 8 / num_points
                        bpp = ddpcc_bpp + gpcc_bpp + ys2_2_bpp
                        pc_ori =  current_f.C[:, 1:]
                        recon_f2 = ME.SparseTensor(torch.ones_like(out2[-1].F[:, :1]), coordinates=out2[-1].C)
                        pc_recon = recon_f2.C[:, 1:]
                        pcd = open3d.geometry.PointCloud()
                        pcd.points = open3d.utility.Vector3dVector(pc_ori.detach().cpu().numpy())
                        pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamKNN(knn=20))  # todo: NOTICE knn value!!! Here is the same as in PCGCv2
                        n1 = torch.tensor(np.asarray(pcd.normals)).cuda()
                        pc_ori, pc_recon, n1 = pc_ori.unsqueeze(0), pc_recon.unsqueeze(0), n1.unsqueeze(0)
                        
                        # print("pc_recon[0]",sequence_name,pc_recon[0].shape)
                        # print(pc_recon[0])
                        # write_ply_data(os.path.join("./test_result/mpeg_code/recon_point/"+sequence_name+'/', str(i)+'.ply'), pc_recon[0].detach().cpu().numpy())

                        d1psnr, d2psnr, cd = PSNR(pc_ori, pc_recon, n1,resolution=args.resolution)
                        
                        
                        
                        log_string(str(i) + ' ' + str(bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] +  str(num_points_sum) + '\n')
                        print(f"{sequence_name},{i},{bpp},{d1psnr},{d2psnr},{num_points_sum}",file=f_file)
                        # log_string(str(i) + ' ' + str(bpp)[:7] + ' ' +str(ddpcc_bpp_y)[:7] + ' ' +str(ddpcc_bpp_z)[:7] + ' ' +str(motion_bpp)[:7] + ' ' + str(d1psnr)[:7] + ' ' + str(d2psnr)[:7] +  str(num_points_sum) + '\n')
                        # print(f"{sequence_name},{i},{bpp},{ddpcc_bpp_y},{ddpcc_bpp_z},{motion_bpp},{d1psnr},{d2psnr},{num_points_sum}",file=f_file)
                        f_file.flush()

                        f1 = recon_f2
                        #重构的f2成为新的参考帧
                        bpp_sum += bpp
                        d1_psnr_sum += d1psnr
                        d2_psnr_sum += d2psnr
                        num_points_sum += (num_points * 1.0)

                        # print("num_points_sum:======>",num_points_sum)

                        cd_sum += cd
                
                model.reset_buffer()
                
                bpp_avg        = bpp_sum / args.frame_count
                d1_psnr_avg    = d1_psnr_sum / args.frame_count
                d2_psnr_avg    = d2_psnr_sum / args.frame_count
                cd_avg         = cd_sum / args.frame_count
                num_points_avg = num_points_sum/args.frame_count
                # print("num_points_avg:",num_points_avg)
                results[sequence_name]['bpp'].append(bpp_avg)
                results[sequence_name]['d1-psnr'].append(d1_psnr_avg)
                results[sequence_name]['d2-psnr'].append(d2_psnr_avg)
                results[sequence_name]['cd'].append(cd_avg)
                results[sequence_name]['num_points'].append(num_points_avg)
                results[sequence_name]['exp_name'].append(exp_name)
                
                print(f"{sequence_name},{args.frame_count},{bpp_avg},{d1_psnr_avg},{d2_psnr_avg},{num_points_avg}",file=f_file)
                f_file.flush()
                torch.cuda.empty_cache()  # 清理 PyTorch 缓存
                log_string(dataset.sequence_list[sequence] + ' average bpp: ' + str(bpp_avg))
                log_string(dataset.sequence_list[sequence] + ' average d1-psnr: ' + str(d1_psnr_avg))
                log_string(dataset.sequence_list[sequence] + ' average d2-psnr: ' + str(d2_psnr_avg))
                log_string(dataset.sequence_list[sequence] + ' average num-points: ' + str(num_points_avg))
                log_string(dataset.sequence_list[sequence] + ' average cd: ' + str(cd_avg))
        f_file.close()
        
    for sequence_name in results:
        df = pd.DataFrame(results[sequence_name])
        # df.to_csv(os.path.join(results_dir, sequence_name + '.csv'), index=False)
        file_path = os.path.join(results_dir, sequence_name + '.csv')

        if os.path.exists(file_path):
            # 如果文件已存在，追加数据且不写入标题
            df.to_csv(file_path, mode='a', header=False, index=False)
        else:
            # 如果文件不存在，正常写入包括标题
            df.to_csv(file_path, mode='w', header=True, index=False)
