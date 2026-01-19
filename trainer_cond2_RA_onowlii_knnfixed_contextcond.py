import argparse
import time

def str2bool(v):
    return str(v).lower() in ("yes", "y", "true", "t", "1")
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('DDPCC')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in training [default: 24]')
    parser.add_argument('--model', default='RA_DDPCC_geo_cond_bn', help='model name [default: pointnet_cls]')
    parser.add_argument('--epoch', default=100, type=int, help='number of epoch in training [default: 200]')
    parser.add_argument('--learning_rate', default=0.0008, type=float,
                        help='learning rate in training [default: 0.0008]')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device [default: 0]')
    parser.add_argument('--decay_rate', type=float, default=1e-5, help='decay rate [default: 1e-4]')
    parser.add_argument('--lamb', default=10.0, type=float, help='loss=lambda*D+R')
    parser.add_argument('--pretrained', default='', type=str, help='path of pretrained model')
    parser.add_argument('--exp_name', default='I10', type=str, help='directory to store the result')
    parser.add_argument('--cpu', default=False, type=bool)
    parser.add_argument('--channels', default=8, type=int, help='bottleneck size')
    parser.add_argument('--activation', default='relu', type=str)
    parser.add_argument('--dataset_dir', default='/dengx/dengxuan/AVS_P/Owlii/', type=str)
    parser.add_argument('--entropy_mode', default="cond", type=str)
    parser.add_argument('--knn_fixed', type=str2bool, default=True,help="True or False")
    parser.add_argument('--point_conv',type=str2bool, default=False,help="True or False")
    parser.add_argument('--fuse_conv',type=str2bool, default=False,help="True or False")

    return parser.parse_args()

args = parse_args()
import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

import importlib
import logging
import random
import shutil
from pathlib import Path
import torch
import sys
from tqdm import tqdm
# from dataset_lossy import *
from dataset_owlii_RA import*

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from tensorboardX import SummaryWriter
import MinkowskiEngine as ME
# torch.autograd.set_detect_anomaly(True)


def log_string(str):
    logger.info(str)
    print(str)


if __name__ == '__main__':

    '''mkdir'''
    if args.exp_name is None:
        args.exp_name = ''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 10))
    log_root_dir = Path('./log_RA_on_owlii')
    log_root_dir.mkdir(exist_ok=True)
    model_dir = log_root_dir.joinpath(args.model)
    model_dir.mkdir(exist_ok=True)
    experiment_dir = model_dir.joinpath(args.exp_name)
    experiment_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('log')
    checkpoint_dir = experiment_dir.joinpath('checkpoint')
    log_dir.mkdir(exist_ok=True)
    checkpoint_dir.mkdir(exist_ok=True)
    
    print("checkpoint_dir===>:",checkpoint_dir)
    
    '''logger'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''tensorboardX'''
    writer = SummaryWriter(comment='_add_regularization')

    '''dataset'''
    # train_dataset = Dataset(root_dir=args.dataset_dir, split=[0, 1, 2, 3], type='train', scaling_factor=4,time_step=1,format='ply')
    
    
    # val_dataset = Dataset(root_dir=args.dataset_dir, split=[0, 1, 2, 3], type='val', scaling_factor=4,time_step=1,format='ply')
    
    
    # h5_file = './h5_dataset/video_frames_dynamic_mpeg.h5'
    # h5_file = "./h5_dataset/video_frames_dynamic_mpeg_10bit_train.h5"
    # h5_file = './h5_dataset/video_frames_dynamic_owlii_train_9bit.h5'
    h5_file = './h5_dataset/video_frames_dynamic_owlii_train_10bit.h5'
    train_dataset = PointCloudVideoDataset(h5_file, clip_length=32)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1,num_workers=4,shuffle=True,pin_memory=True)
                                                    
    # val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False,
    #                                               collate_fn=collate_pointcloud_fn)

    '''model'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/model_utils.py', str(experiment_dir))
    model = MODEL.get_model(channels=args.channels ,entropy_mode=args.entropy_mode,knn_fixed=args.knn_fixed,point_conv = args.point_conv,fuse_conv=args.fuse_conv)

    '''pretrained'''
#     try:
#         checkpoint = torch.load(str(experiment_dir) + '/checkpoint/best_model.pth', map_location='cuda:0')
#         start_epoch = checkpoint['epoch']
#         model.load_state_dict(checkpoint['model_state_dict'])
#         log_string('Detected existed model')
#         exist = True
        
#         print("====>",str(experiment_dir) + '/checkpoint/best_model.pth')
#     except:
#         log_string('No existing model')
#         start_epoch = 0
#         exist = False
    
    start_epoch=0
    if len(args.pretrained) != 0:
        checkpoint = torch.load(args.pretrained)
        start_epoch = 0
        import collections

        old_paras = model.state_dict()
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k1 = k.replace('module.', '')
            if k1 in old_paras:
                new_state_dict[k1] = v
        old_paras.update(new_state_dict)
        
        model.load_state_dict(old_paras,strict=False)
        log_string('Finetuning')

    if not args.cpu:
        model = model.cuda()

    '''optimizer'''
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=args.decay_rate
    )
    # if exist:
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     log_string('use exist optimizer')
    #     best_loss_test = 99999999
    # else:
    #     best_loss_test = 99999999
    best_loss_test = 99999999    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.2)
    # for i in range(start_epoch % 15):  # % scheduler step_size
    #     scheduler.step()

    '''training'''
    log_string('start training')
    step = 0
    for epoch in range(start_epoch, args.epoch):
        log_string('\nEpoch: ' + str(epoch+1))
        total_bpp = 0
        total_cls = 0
        # total_loss = 0
        counter = 0
        model.train()
        device = torch.device('cuda' if not args.cpu else 'cpu')
        lamb = args.lamb
        # if epoch >= 10 or len(args.pretrained) != 0:
        #     lamb = args.lamb
        # else:
        #     lamb = 20.0
        log_string('current lambda: ' + str(lamb))
        log_string('训练数据总长度: ' + str(len(train_dataloader)))#376 
        
        
        
        for idx, (clip_list, metadata) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
            # try:
            clip_list = [frame.cuda() for frame in clip_list]  # [32 x [N_i, C]]
            # print("metadata:",metadata)
            video_idx, start_idx = metadata[0].item(),metadata[1].item()
            
            # print("len(clip_list)",len(clip_list))
            
            frame_indices_display = list(range(start_idx, start_idx + 32))
            gop_starts = [start_idx, start_idx + 16]  # e.g., [208, 224]
            
            encoding_order = []
            for gop_start in gop_starts:
                encoding_order.append(gop_start)
                gop_frames = [idx for idx in range(gop_start, gop_start + 16) if idx != gop_start]
                rel_indices = [idx - gop_start for idx in gop_frames]
                sorted_rel_indices = sorted(rel_indices, key=lambda x: model.ra_order.index(x) if x in model.ra_order[1:] else float('inf'))
                encoding_order.extend([gop_start + rel_idx for rel_idx in sorted_rel_indices])
            
            # Process each frame
            clip_loss = 0.0
            
            # print("encoder_order:",video_idx, start_idx,encoding_order)
            
            min_index = min(encoding_order)
            
            # print("min_index:",min_index)
            
            # encoding_order = encoding_order - min_index
            encoding_order = [i - min_index for i in encoding_order]
            
            # print("encoder order:",encoding_order)
            
            model.reset_buffer()
            optimizer.zero_grad()
            
            total_loss = 0
            accumulation_steps = 8  


            for i, frame_idx in enumerate(encoding_order):
                
                current_xyz = clip_list[i][0,...]
                current_point= np.expand_dims(np.ones(current_xyz.shape[0]),1)

                # current_xyz=ME.utils.batched_coordinates(current_xyz)
                # current_point = torch.ones_like(current_xyz[:, :1])
                list_data=[(current_xyz,current_point)]
                coords, feats = list(zip(*list_data))
                coords_batch, feats_batch = ME.utils.sparse_collate(coords, feats)
                current_f = ME.SparseTensor(features=feats_batch.float(), coordinates=coords_batch.float(), device=device)
                
                # print("current_xyz:",current_xyz.shape)
                # print("current_point:",current_point.shape)
                
                # current_f = ME.SparseTensor(features=current_point, coordinates=current_xyz, device=device)
                
                # optimizer.zero_grad()
                
                cur_f, out2, out_cls2, targets2, keeps2, bpp = model(current_f=current_f,video_idx=video_idx, frame_idx=frame_idx , device=device)
                
                
                
                if out2[0]==0:
                    continue
                
                distortion = 0
                distortion_ = 0
                num_losses = len(out_cls2)
                for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
                    curr_loss = model.crit(out_cl.F.squeeze(),
                                           target.type(out_cl.F.dtype).to(device))
                    distortion += curr_loss / float(num_losses)
                loss = distortion * lamb + bpp
                # step = (step + 1) % args.batch_size
                loss.backward()
                # if step == 0:
                optimizer.step()
                optimizer.zero_grad()
                total_bpp += bpp.item()
                total_cls += distortion.item()
                # total_loss += loss
                counter += 1
                
                # if (i + 1) % 2 == 0 or (i + 1) == len(encoding_order):
                #     avg_loss = total_loss / counter
                #     avg_loss.backward()
                #     optimizer.step()
                #     optimizer.zero_grad()
                #     total_loss = 0
                #     counter = 0

                
            # loss =total_loss/31.0
            # loss.backward()
            #     # if step == 0:
            # optimizer.step()
            
            model.reset_buffer()
        
        
        avg_bpp = total_bpp / counter
        avg_cls = total_cls / counter
        # avg_loss = total_loss / counter

        log_string('\naverage bpp: ' + str(avg_bpp))
        log_string('\naverage cls: ' + str(avg_cls))
        # log_string('\naverage loss: ' + str(avg_loss))
        log_string('\naverage learning rate: ' + str(optimizer.param_groups[0]['lr']))
        scheduler.step()
        if epoch % 1 == 0:
            savepath = str(checkpoint_dir) + '/' + str(epoch) + '.pth'
            state = {
                'epoch': epoch,
                'loss': avg_bpp,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
        torch.save(state, savepath)
                
                
                
#             frames,indices = data
#             print("frames,indices:",len(frames),len(indices))

#             for i in range(32):
#                 current_xyz,current_point=frames[i]
#                 indice = indices[i]
#                 # print("current_xyz,current_point:",current_xyz.shape,current_point.shape)
#             # print("indices:",indices)

#                 current_f = ME.SparseTensor(features=current_point[0,...], coordinates=current_xyz[0,...], device=device)
#                 cur_f, out2, out_cls2, targets2, keeps2, bpp = model(current_f=current_f,indices=indice, device=device)
#                 if out2[0]==0:
#                     continue

#                 distortion = 0
#                 distortion_ = 0
#                 num_losses = len(out_cls2)
#                 for i, (out_cl, target) in enumerate(zip(out_cls2, targets2)):
#                     curr_loss = model.crit(out_cl.F.squeeze(),
#                                            target.type(out_cl.F.dtype).to(device))
#                     distortion += curr_loss / float(num_losses)
#                 loss = distortion * lamb + bpp

#                 step = (step + 1) % args.batch_size
#                 loss.backward()
#                 if step == 0:
#                     optimizer.step()
#                     optimizer.zero_grad()
#                 total_bpp += bpp.item()
#                 total_cls += distortion.item()
#                 total_loss += loss.item()
#                 counter += 1

                
#         avg_bpp = total_bpp / counter
#         avg_cls = total_cls / counter
#         avg_loss = total_loss / counter

#         log_string('\naverage bpp: ' + str(avg_bpp))
#         log_string('\naverage cls: ' + str(avg_cls))
#         log_string('\naverage loss: ' + str(avg_loss))
#         log_string('\naverage learning rate: ' + str(optimizer.param_groups[0]['lr']))
#         writer.add_scalar('Train Loss', avg_loss, epoch)
#         writer.add_scalar('Train bpp', avg_bpp, epoch)
#         writer.add_scalar('Train cls', avg_cls, epoch)
#         scheduler.step()
#         if epoch % 1 == 0:
#             savepath = str(checkpoint_dir) + '/' + str(epoch) + '.pth'
#             state = {
#                 'epoch': epoch,
#                 'loss': avg_bpp,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#             }
#             torch.save(state, savepath)
#         log_string('evaluating')
#         optimizer.zero_grad()
#         torch.cuda.empty_cache()
    

