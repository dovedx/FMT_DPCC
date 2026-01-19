# Requirements
cuda~=11.5.50
numpy~=1.21.2
open3d~=0.14.1
pandas~=1.2.3
torch~=1.10.0
MinkowskiEngine~=0.5.4
pytorch3d~=0.6.1
tqdm~=4.62.3
tensorboardX~=2.5
matplotlib~=3.5.1
h5py~=3.6.0
torchac~=0.9.3
setuptools~=58.0.4
scipy~=1.7.3
scikit-learn~=1.0.2
pointops 
h5py 

## Testing Instructions

We provide testing scripts that support evaluation on the **MPEG 8i** dataset.

- The default **Group of Frames (GOF)** is set to **16**, and the model is trained accordingly with `GOF = 16`.
- The **K-Nearest Neighbors (KNN)** hyperparameter is set to **K = 32**.  
  A larger `K` may lead to better alignment results, but will incur higher computational cost.

### compress coordinate

- For **GPCC** and **lossless compression** baselines, we follow the settings described in the *D-DPCC* paper.

### First Frame Compression

- For compressing the **first frame** of each sequence, static point cloud compressors  **SparsePCC** can be applied.

## Test:
python test_mpeg_cond_RA_onowlii_knnfixed.py 
--log_name='test_mpeg_RA' 
--gpu=0 
--frame_count=32 
--results_dir='results' 
--channels=32 
--tmp_dir='./test_result/mpeg_code/tmp_mpeg_test_RA' 
--ckpt_dir='./ddpcc_ckpts' 
--dataset_dir='your_path/MPEG_8i/' 
--resolution=1023 
--scaling_factor=1 
--entropy_mode="context_cond" 
--knn_fixed=True 
--point_conv=False 
--fuse_conv=False

## Test:
python trainer_cond2_onowlii_knnfixed.py
 --batch_size=1 
 --gpu=0 
 --lamb=20 
 --channels=32 
 --exp_name=20241204_lambda20_300frame_fixedknn_v5_onowlii_context_cond_bn_crosstransfomerv2_KNN32_new 
 --dataset_dir='/dengx/dengxuan/AVS_P/Owlii/' 
 --pretrained='model_path'   
 --entropy_mode='context_cond' 
 --learning_rate=0.0001 
 --knn_fixed=True 
 --point_conv=False 
 --scale_wave=False