import os, sys
# rootdir = os.path.split(__file__)[0]
# sys.path.append(rootdir)
# rootdir = os.path.split(rootdir)[0]
# sys.path.append(rootdir)
import time
import numpy as np
import os, glob, tqdm
import torch
import pandas as pd


from SparsePCC.models.model import PCCModel
from SparsePCC.coder import BasicCoder, LossyCoderDense


import os, time
import torch
import MinkowskiEngine as ME
import numpy as np
from SparsePCC.data_utils.data_loader import load_sparse_tensor
from SparsePCC.data_utils.quantize import quantize_sparse_tensor
from SparsePCC.data_utils.quantize import quantize_precision, dequantize_precision
from SparsePCC.data_utils.quantize import quantize_resolution, dequantize_resolution
from SparsePCC.data_utils.quantize import quantize_octree, dequantize_octree
from SparsePCC.data_utils.inout import read_ply_o3d, write_ply_o3d, read_coords
from SparsePCC.data_utils.sparse_tensor import sort_sparse_tensor
from SparsePCC.extension.metrics import pc_error, get_PSNR_VCN, get_PSNR_attn


def load_data(filedir, voxel_size=1, posQuantscale=1,device=None):
        """load data & pre-quantize if posQuantscale>1
        """
        x_raw = load_sparse_tensor(filedir, voxel_size=voxel_size, device=device)
        x = quantize_sparse_tensor(x_raw, factor=1/posQuantscale, quant_mode='round')
        if x.C.min() < 0:
            ref_point = x.C.min(axis=0)[0]
            print('DBG!!! min_points', ref_point.cpu().numpy())
            x = ME.SparseTensor(features=x.F, coordinates=x.C - ref_point, 
                                tensor_stride=x.tensor_stride, device=x.device)
        else: ref_point = None
        # self.filename = os.path.split(filedir)[-1].split('.')[0]
        # self.ref_point = ref_point# TODO

        return x

def scale_sparse_tensor(x, factor):
    coords = (x.C[:,1:]*factor).round().int()
    feats = torch.ones((len(coords),1)).float()
    coords, feats = ME.utils.sparse_collate([coords], [feats])
    x = ME.SparseTensor(features=feats, coordinates=coords, tensor_stride=1, device=x.device)
    
    return x    

def test_one_frame_lossy(x, ckptdir, ckptdir_sr, ckptdir_ae , out_path, scaling_factor=1.0, rho=1.0, res=1024,device=None):
    
    filename ="longdress_10bit"
    idx_rate=1
    psnr_resolution=1023
    
    model = PCCModel(stage=8, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir)
    model.load_state_dict(ckpt['model'])
    basic_coder = BasicCoder(model, device=device)

    model_SR = PCCModel(stage=1, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir_sr)
    ckpt = torch.load(ckptdir_sr)
    model_SR.load_state_dict(ckpt['model'])

    model_AE = PCCModel(stage=1, kernel_size=3, enc_type='ae').to(device)
    assert os.path.exists(ckptdir_ae)
    ckpt = torch.load(ckptdir_ae)
    model_AE.load_state_dict(ckpt['model'])

    lossy_coder = LossyCoderDense(basic_coder, model_AE, model_SR, device=device)
    
    
    bin_dir = os.path.join(out_path, filename+'_R'+str(idx_rate)+'.bin')
    dec_dir = os.path.join(out_path, filename+'_R'+str(idx_rate)+'.ply')
    
    scale_AE = 1
    scale_SR = 0
    results,points_dec = lossy_coder.test(x, bin_dir, dec_dir,scale_AE=scale_AE, scale_SR=scale_SR, psnr_resolution= psnr_resolution)
    
    # print("results:",results)
    bpp = results['bpp']
    d1psnr = results['mseF,PSNR (p2point)']
    d2psnr = results['mseF,PSNR (p2plane)']
    x_dec = points_dec
    
    return bpp, d1psnr, d2psnr, x_dec



def test_one_frame_lossless(x, ckptdir, out_path, scaling_factor=1.0, rho=1.0, res=1024,device=None):
    
    filename ="longdress_10bit_lossless"
    idx_rate=1
    psnr_resolution=1023
    
    model = PCCModel(stage=8, kernel_size=3, enc_type='pooling').to(device)
    assert os.path.exists(ckptdir)
    ckpt = torch.load(ckptdir)
    model.load_state_dict(ckpt['model'])
    basic_coder = BasicCoder(model, device=device)
    
    bin_dir = os.path.join(out_path, filename+'.bin')
    dec_dir = os.path.join(out_path, filename+'_dec.ply')
    results ,x_dec = basic_coder.test(x, bin_dir, dec_dir, voxel_size=1, posQuantscale=1) 
    bpp = results['bpp']
    bits = results['file_size']
    
    return bits, bpp, x_dec
    

os.environ['CUDA_VISIBLE_DEVICES'] = '0'   
if __name__ == '__main__':
    
    
    
# python test_ours_dense.py --mode='lossless' --ckptdir='./ckpts/dense/epoch_last.pth' --filedir='./dataset/testdata/8iVFB/' --prefix='ours_8i'

# python test_ours_dense.py --mode='lossy' --ckptdir='./ckpts/dense/epoch_last.pth' --ckptdir_sr='./ckpts/dense_1stage/epoch_last.pth' --ckptdir_ae='./ckpts/dense_slne/epoch_last.pth' --filedir='./dataset/testdata/8iVFB/' --psnr_resolution=1023 --prefix='ours_8i_lossy'
    
    # x = ME.SparseTensor(features=point, coordinates=xyz, device=device)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x=load_data(filedir="/dengx/dengxuan/AVS_P/train_data/MPEG_8i/longdress_ori/Ply/longdress_vox10_1051.ply",device = device)
    print("x:",x)
    
    
    bpp, d1psnr, d2psnr, x_dec= test_one_frame_lossy(x, ckptdir='./ckpts/dense/epoch_last.pth', ckptdir_sr='./ckpts/dense_1stage/epoch_last.pth', ckptdir_ae='./ckpts/dense_slne/epoch_last.pth' , out_path='./result/', scaling_factor=1.0, rho=1.0, res=1024)
    
    print("bpp, d1psnr, d2psnr:",bpp, d1psnr, d2psnr)
    
    bits,bpp, x_dec = test_one_frame_lossless(x, ckptdir='./ckpts/dense/epoch_last.pth', out_path='./result/', scaling_factor=1.0, rho=1.0, res=1024)
    
    print("bpp:",bpp)
    
    print(x_dec)
    
  