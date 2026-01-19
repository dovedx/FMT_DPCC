import torch
import torchac
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from entropy_coding import factorized_entropy_coding, factorized_entropy_decoding

class Bitparm(nn.Module):
    # save params
    def __init__(self, channel, dimension=4, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        para = [1 for i in range(dimension)]
        para[dimension - 1] = -1
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(para), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)


class BitEstimator(nn.Module):
    def __init__(self, channel=8, dimension=3):
        super(BitEstimator, self).__init__()
        self.f1 = Bitparm(channel, dimension=dimension)
        self.f2 = Bitparm(channel, dimension=dimension)
        self.f3 = Bitparm(channel, dimension=dimension)
        self.f4 = Bitparm(channel, dimension=dimension, final=True)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)


def test_factorized_entropy_codec():
    """测试因子熵编码和解码函数是否能正确恢复原始数据"""
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    np.random.seed(42)
    
#     # 创建一个简单的BitEstimator模型 (在实际应用中这可能是一个训练好的神经网络)
#     class BitEstimator(torch.nn.Module):
#         """模拟位估计器，返回输入的sigmoid值作为累积分布函数估计"""
#         def __init__(self):
#             super().__init__()
#             self.scale = torch.nn.Parameter(torch.tensor(1.0))
            
#         def forward(self, x):
#             # 模拟一个简单的累积分布函数
#             return torch.sigmoid(x * self.scale)
    
    # 初始化位估计器
    bit_estimator = BitEstimator()
    
    # 生成测试数据
    # 这里使用随机整数作为测试数据，范围从-10到10，形状为(1, 5, 8)
    batch_size = 1
    channels = 10
    spatial_size = 8
    test_input = torch.randint(-10, 11, (batch_size, channels, spatial_size), dtype=torch.float32)
    
    print(f"原始输入数据 (shape={test_input.shape}):")
    print(test_input)
    
    # 执行编码
    bitstream, min_val, max_val = factorized_entropy_coding(bit_estimator, test_input)
    print(f"\n编码后:")
    print(f"- 比特流长度: {len(bitstream)} 字节")
    print(f"- 最小值: {min_val.item()}")
    print(f"- 最大值: {max_val.item()}")
    
    # 执行解码
    decoded_output = factorized_entropy_decoding(
        bit_estimator, 
        shape=test_input.shape[1:],  # 传递除batch维度外的形状
        bitstream=bitstream,
        min_v=min_val.item(),
        max_v=max_val.item(),
        device=test_input.device
    )
    
    print(f"\n解码后的数据 (shape={decoded_output.shape}):")
    print(decoded_output)
    
    # 验证解码结果是否与原始输入匹配
    # 注意：由于浮点数精度问题，我们使用容差比较
    max_diff = torch.max(torch.abs(test_input.squeeze(0) - decoded_output)).item()
    print(f"\n最大差异: {max_diff}")
    
    if max_diff < 1e-6:
        print("✅ 测试通过：解码结果与原始输入一致")
    else:
        print("❌ 测试失败：解码结果与原始输入存在差异")
    
    # 计算压缩率
    original_bits = test_input.numel() * 32  # float32每元素32位
    compressed_bits = len(bitstream) * 8
    compression_ratio = original_bits / compressed_bits
    print(f"\n压缩率: {compression_ratio:.2f}x")

if __name__ == "__main__":
    test_factorized_entropy_codec()    