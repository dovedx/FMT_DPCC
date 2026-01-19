import torch.nn.functional as F
import torch.nn as nn
import torch
import math
import torch.nn.init as init
import numpy as np
# import torch
import matplotlib.pyplot as plt

def analyze_tensor_distribution(tensor, save_path=None, file_format='png', num_bins=50, plot_title="Tensor Value Distribution"):
    """
    分析并可视化PyTorch张量的值分布，并支持保存图片
    
    参数:
    tensor (torch.Tensor): 需要分析的四维张量
    save_path (str): 图片保存路径（如"distribution_plot.png"），若为None则不保存
    file_format (str): 保存格式（支持'png', 'pdf', 'svg'等）
    num_bins (int): 直方图的分箱数量
    plot_title (str): 图表标题
    """
    # 确保张量在CPU上
    if tensor.is_cuda:
        tensor = tensor.cpu()
    
    # 将四维张量展平为一维数组
    flat_tensor = tensor.reshape(-1).numpy()
    
    # 计算基本统计量
    # 统计大于1的值
    values_gt_1 = flat_tensor[flat_tensor > 1]
    count_gt_1 = len(values_gt_1)
    percent_gt_1 = (count_gt_1 / len(flat_tensor)) * 100
    print("===>",count_gt_1,len(flat_tensor))
    
    min_val = np.min(flat_tensor)
    max_val = np.max(flat_tensor)
    mean_val = np.mean(flat_tensor)
    median_val = np.median(flat_tensor)
    std_dev = np.std(flat_tensor)
    
    print(f"张量基本统计信息:")
    print(f"  最小值: {min_val:.4f}")
    print(f"  最大值: {max_val:.4f}")
    print(f"  平均值: {mean_val:.4f}")
    print(f"  中位数: {median_val:.4f}")
    print(f"  标准差: {std_dev:.4f}")
    print(f"  值 >1 的数量: {count_gt_1:,} ({percent_gt_1:.2f}%)")
    
    # 创建直方图（纵坐标为值的个数）
    plt.figure(figsize=(10, 6))
    n, bins, patches = plt.hist(flat_tensor, bins=num_bins, density=False, alpha=0.7, color='skyblue')
    
    # 高亮显示 >1 的区间
    for i, rect in enumerate(patches):
        if bins[i] > 1:
            rect.set_facecolor('red')
            rect.set_alpha(0.7)
    
    # 添加统计信息文本
    stats_text = (f'Min: {min_val:.4f}\n'
                 f'Max: {max_val:.4f}\n'
                 f'Mean: {mean_val:.4f}\n'
                 f'Median: {median_val:.4f}\n'
                 f'Std Dev: {std_dev:.4f}\n'
                 f'Total Values: {len(flat_tensor):,}\n'
                 f'Values >1: {count_gt_1:,} ({percent_gt_1:.2f}%)')
    plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 设置图表属性
    plt.title(plot_title)
    plt.xlabel('Tensor Value')
    plt.ylabel('Number of Values')
    plt.grid(axis='y', alpha=0.75)
    
    # 标记平均值、中位数和阈值1的位置
    plt.axvline(1, color='black', linestyle='solid', linewidth=1.5, label='Threshold: 1')
    plt.axvline(mean_val, color='r', linestyle='dashed', linewidth=1, label=f'Mean: {mean_val:.4f}')
    plt.axvline(median_val, color='g', linestyle='dashed', linewidth=1, label=f'Median: {median_val:.4f}')
    
    # 添加图例说明红色区域
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='skyblue', edgecolor='black', alpha=0.7, label='Values ≤1'),
        Patch(facecolor='red', edgecolor='black', alpha=0.7, label='Values >1'),
        plt.Line2D([0], [0], color='black', linestyle='-', linewidth=1.5, label='Threshold: 1'),
        plt.Line2D([0], [0], color='r', linestyle='--', linewidth=1, label=f'Mean: {mean_val:.4f}'),
        plt.Line2D([0], [0], color='g', linestyle='--', linewidth=1, label=f'Median: {median_val:.4f}')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图片（如果指定路径）
    if save_path:
        plt.savefig(save_path, format=file_format, dpi=300, bbox_inches='tight')
        print(f"直方图已保存至: {save_path}")
    

def batch_gather(x, knn_idx, mode='plain'):
    b, n, k = knn_idx.shape
    if mode == 'plain':
        idx = torch.arange(b).to(x.device).view(-1, 1, 1).expand(-1, n, k)
        out = x[idx, :, knn_idx].permute(0, 3, 2, 1)                                             # B * C * K * N2

    elif mode == 'residual':
        idx = torch.arange(b).to(x.device).view(-1, 1, 1).expand(-1, n, k-1)
        center = x.unsqueeze(2)
        out = torch.cat([center, x[idx, :, knn_idx[..., 1:]].permute(0, 3, 2, 1)-center], 2)

    return out

class PointConv(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors=5, kernel_size=5, bias=True, radius=4):
        super(PointConv, self).__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.use_bias     = bias
        self.radius       = radius
        
        self.conv_1x1 = nn.Conv1d(in_channels, out_channels//2, 1, bias=bias)
            
        self.conv_dw = nn.Parameter(torch.randn(1, out_channels//2, kernel_size, kernel_size, kernel_size))

        # initialization
        fan = n_neighbors
        gain = init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
       
        self.conv_dw.data.uniform_(-bound, bound)

    def forward(self, xyz, fea, knn_idx,index=1):
        # xyz: B * 3 * N  |  fea: B * C * N | knn_idx: B * N * K
        b, n, k = knn_idx.shape
        neighbor_xyz = batch_gather(xyz, knn_idx)
    
        rel_xyz = neighbor_xyz - xyz.unsqueeze(2)  # B, 3, K, N
        #动态的改变self.radius
        
        # print("rel_xyz:",torch.min(rel_xyz),torch.max(rel_xyz))
        
        # print("rel_xyz.std(dim=(2, 3):",rel_xyz.std(dim=(2, 3),rel_xyz.std(dim=(2, 3))))
                                                      
        sample_xyz = rel_xyz / (self.radius* rel_xyz.std(dim=(2, 3), keepdim=True))#
        # print("sample_xyz:",torch.min(sample_xyz),torch.max(sample_xyz))  
        # print("分析sample_xyzs.....")
        # analyze_tensor_distribution(tensor=sample_xyz, save_path='./hist_image/'+str(index)+".png", file_format='png', num_bins=50, plot_title="Tensor Value Distribution")
        # print("分析结束sample_xyzs.....")
                                                      
        #sample_xyz中大部分值位于[-2,2]之间
        sample_xyz = sample_xyz.permute(0, 2, 3, 1).unsqueeze(-2)
        # 1x1 conv
        neighbor_fea = batch_gather(fea, knn_idx, 'residual')
        neighbor_fea = self.conv_1x1(neighbor_fea.view(b, -1, k * n)).view(b, -1, k, n)
        # print("neighbor_fea:",neighbor_fea.shape)
        # aggregation，(b,c,k,k,k)->grid(B,k,N,1,3)
       
        kernel = F.grid_sample(self.conv_dw.expand(b, -1, -1, -1, -1), sample_xyz,mode='nearest', padding_mode='border', align_corners=False).squeeze()
        kernel = kernel.view(b, self.out_channels//2, -1, n)  # B * C_out * K * N
        out = torch.cat([(kernel * neighbor_fea).sum(2), (neighbor_fea).mean(2)], 1)
        
        # analyze_tensor_distribution(tensor=sample_xyz, save_path='./hist_image/'+str(index)+".png", file_format='png', num_bins=50, plot_title="Tensor Value Distribution")
        # print("分析结束sample_xyzs.....")

        return out

    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_neighbors=5,args=None):
        super(EncoderBlock, self).__init__()
        self.relu = nn.LeakyReLU(0.1, True)
        
        self.conv1 = PointConv(in_channels, in_channels, n_neighbors, kernel_size=5, bias=True)
        self.conv2 = PointConv(in_channels, out_channels, n_neighbors, kernel_size=5, bias=True)

        # shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        else:
            self.shortcut = nn.Identity()

    def forward(self, xyzs, feats, knn_idx,index=1):
        # body
        
        out = self.relu(self.conv1(xyzs, feats, knn_idx,index=index*2))
        out = self.conv2(xyzs, out, knn_idx,index=index*2+1)

        # tail
        out = self.relu(out + self.shortcut(feats))

        return out