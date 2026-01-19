import torch
import torch.nn as nn
import MinkowskiEngine as ME
# from torch_cluster import knn

class KNNContextModel(nn.Module):
    def __init__(self, in_channels, out_channels, k=8, dimension=3):
        """
        基于 k-NN 的上下文模型，用于点云几何压缩。

        参数:
            in_channels (int): 输入特征通道数（锚点潜在表示的通道数）。
            out_channels (int): 输出通道数（预测均值和尺度的通道数）。
            k (int): 最近邻数量，默认为 8。
            dimension (int): 点云维度，默认为 3（x, y, z）。
        """
        super(KNNContextModel, self).__init__()
        self.k = k
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dimension = dimension

        # 邻域特征处理网络
        self.neighbor_processor = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=in_channels * (k + 1),  # 中心点 + k 个邻居
                out_channels=128,
                kernel_size=1,  # 1x1 卷积，聚合特征
                dimension=dimension
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
        )

        # 上下文预测网络
        self.context_predictor = nn.Sequential(
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                dimension=dimension
            ),
            ME.MinkowskiBatchNorm(128),
            ME.MinkowskiReLU(inplace=True),
            ME.MinkowskiConvolution(
                in_channels=128,
                out_channels=2 * out_channels,  # 输出均值和尺度
                kernel_size=1,
                dimension=dimension
            ),
        )

    def forward(self, x):
        """
        前向传播，基于 k-NN 预测上下文。

        参数:
            x (SparseTensor): 输入稀疏张量，包含锚点特征和坐标。

        返回:
            SparseTensor: 输出稀疏张量，特征为预测的均值和尺度。
        """
        # 提取坐标和特征
        coords = x.C  # 坐标，形状 [N, 4]（batch_idx, x, y, z）
        feats = x.F   # 特征，形状 [N, in_channels]
        device = x.device

        # k-NN 查找
        batch_size = coords[:, 0].max().item() + 1
        row, col = knn(coords[:, 1:], coords[:, 1:], k=self.k, batch_x=coords[:, 0], batch_y=coords[:, 0])
        # row: 邻居点索引，col: 中心点索引

        # 收集邻居特征
        neighbor_feats = feats[row]  # 形状 [N*k, in_channels]
        center_feats = feats[col]    # 形状 [N*k, in_channels]
        center_feats = center_feats.repeat_interleave(self.k, dim=0)  # 重复以匹配邻居

        # 拼接中心点和邻居特征
        combined_feats = torch.cat([center_feats, neighbor_feats], dim=1)  # 形状 [N*k, in_channels*(k+1)]

        # 构建稀疏张量
        neighbor_coords = coords[col]  # 使用中心点坐标
        neighbor_tensor = ME.SparseTensor(
            features=combined_feats,
            coordinates=neighbor_coords,
            device=device
        )

        # 处理邻域特征
        processed_feats = self.neighbor_processor(neighbor_tensor)

        # 预测上下文（均值和尺度）
        context_tensor = self.context_predictor(processed_feats)

        return context_tensor

    def init_weights(self):
        """
        初始化权重。
        """
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.bias, 0)
                nn.init.constant_(m.bn.weight, 1.0)