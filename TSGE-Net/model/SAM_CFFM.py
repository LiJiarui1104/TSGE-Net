# ljr
import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import numpy as np
from sklearn.neighbors import NearestNeighbors
from .kan import KAN
from .DCLS import Dcls2d
from .GCN import CTRGC
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class AdaptiveKAttention(nn.Module):


    def __init__(self, dim, k_min=3, k_max=16):
        super().__init__()
        self.k_min = k_min
        self.k_max = k_max
        self.k_range = k_max - k_min + 1


        self.k_predictor = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim // 8, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 8, 1, 1),
            nn.Sigmoid()
        )


        self.global_pool = nn.AdaptiveAvgPool2d(1)


        self.local_weight = nn.Parameter(torch.ones(1) * 0.7)
        self.global_weight = nn.Parameter(torch.ones(1) * 0.3)

    def forward(self, x):


        B, C, H, W = x.shape


        local_k_logits = self.k_predictor(x)  # [B, 1, H, W]
        local_k_normalized = local_k_logits.squeeze(1)  # [B, H, W]

        # 全局K值预测
        global_features = self.global_pool(x)  # [B, C, 1, 1]
        global_k_logits = self.k_predictor(global_features)  # [B, 1, 1, 1]
        global_k_normalized = global_k_logits.squeeze()  # [B]

        # 融合局部和全局信息
        # 将全局K值广播到所有位置
        global_k_expanded = global_k_normalized.unsqueeze(-1).unsqueeze(-1).expand(B, H, W)


        fused_k = self.local_weight * local_k_normalized + self.global_weight * global_k_expanded


        k_values = fused_k * (self.k_max - self.k_min) + self.k_min
        k_values = torch.clamp(k_values, self.k_min, self.k_max)

        return k_values, global_k_normalized


class SAM(nn.Module):
    def __init__(self, dim, k_min=3, k_max=16):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)

        self.conv_spatial = Dcls2d(
            in_channels=dim,
            out_channels=dim,
            kernel_count=5,
            stride=1,
            padding=1,
            dilated_kernel_size=3,
            groups=dim,
            version='gauss',
            use_implicit_gemm=False
        )
        self.conv1 = nn.Conv2d(dim, dim, 1)
        self.ctrgc = CTRGC(in_channels=dim, out_channels=dim)


        self.adaptive_k = AdaptiveKAttention(dim, k_min, k_max)
        self.k_min = k_min
        self.k_max = k_max

    def compute_adaptive_knn_adjacency(self, features, k_values):

        B, C, H, W = features.shape
        V = H * W

        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'),
            dim=-1
        ).reshape(-1, 2)  # [V, 2]
        coords = coords.cpu().numpy()

        adjacency_matrices = []

        for b in range(B):

            batch_k_values = k_values[b].detach().cpu().numpy().flatten()  # [V]


            max_k = int(np.ceil(batch_k_values.max()))
            max_k = min(max_k, V - 1)


            if max_k <= 0:
                max_k = 1

            try:
                nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='auto').fit(coords)
                distances, indices = nbrs.kneighbors(coords)
            except ValueError as e:
                print(f"KNN Error: max_k={max_k}, V={V}, coords shape={coords.shape}")
                print(f"batch_k_values range: [{batch_k_values.min():.3f}, {batch_k_values.max():.3f}]")
                raise e


            A = np.zeros((V, V), dtype=np.float32)

            for i in range(V):

                current_k = int(np.round(batch_k_values[i]))
                current_k = max(1, min(current_k, max_k))


                for j in range(current_k):
                    if j < len(indices[i]):
                        neighbor_idx = indices[i, j]
                        if neighbor_idx != i:

                            weight = 1.0 / (distances[i, j] + 1e-8)
                            A[i, neighbor_idx] = weight
                            A[neighbor_idx, i] = weight

            adjacency_matrices.append(torch.tensor(A, dtype=torch.float32, device=features.device))

        return torch.stack(adjacency_matrices, dim=0)  # [B, V, V]

    def compute_global_knn_adjacency(self, features, global_k):

        B, C, H, W = features.shape
        V = H * W

        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij'),
            dim=-1
        ).reshape(-1, 2)  # [V, 2]
        coords = coords.cpu().numpy()

        adjacency_matrices = []

        for b in range(B):
            k = int(torch.round(global_k[b]).detach().cpu().item())
            k = min(max(k, self.k_min), min(self.k_max, V - 1))

            nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(coords)
            knn_graph = nbrs.kneighbors_graph(coords).toarray()  # [V, V]

            A = torch.tensor(knn_graph, dtype=torch.float32, device=features.device)
            adjacency_matrices.append(A)

        return torch.stack(adjacency_matrices, dim=0)  # [B, V, V]

    def forward(self, x):
        u = x.clone()

        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        b, c, h, w = attn.shape
        V = h * w


        k_values, global_k = self.adaptive_k(attn)  # [B, H, W], [B]


        A = self.compute_adaptive_knn_adjacency(attn, k_values)  # [B, V, V]



        x_in = attn.view(b, c, 1, V)  # 转换为 [B, C, T=1, V]


        if hasattr(self.ctrgc, 'batch_forward'):
            out = self.ctrgc.batch_forward(x_in, A)
        else:
            outputs = []
            for i in range(b):
                out_i = self.ctrgc(x_in[i:i + 1], A[i])
                outputs.append(out_i)
            out = torch.cat(outputs, dim=0)

        return out.view(b, c, h, w)  # 输出恢复为 [B, C, H, W]


class CFFM(nn.Module):
    def __init__(self, dim, num_heads=8, LayerNorm_type='WithBias'):
        super(CFFM, self).__init__()
        self.num_heads = num_heads


        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)
        self.KAN = KAN(layers_hidden=[dim, dim])

        self.channel_weight1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )
        self.channel_weight2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

        self.att1 = SAM(dim)
        self.att2 = SAM(dim)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)
        out1=self.att1(x1)
        out2=self.att2(x2)
        KAN1 = self.KAN(out1)
        KAN2 = self.KAN(out2)
        # 生成通道注意力权重
        w1 = self.channel_weight1(out1)  # 形状: [b, c, 1, 1]
        w2 = self.channel_weight2(out2)  # 形状: [b, c, 1, 1]
        Kout1 = w1 * KAN1 + out1
        Kout2 = w2 * KAN2 + out2
        k1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        v1 = rearrange(out1, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        k2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        v2 = rearrange(out2, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q2 = rearrange(Kout1, 'b (head c) h w -> b head w (h c)', head=self.num_heads)
        q1 = rearrange(Kout2, 'b (head c) h w -> b head h (w c)', head=self.num_heads)
        q1 = torch.nn.functional.normalize(q1, dim=-1)
        q2 = torch.nn.functional.normalize(q2, dim=-1)
        k1 = torch.nn.functional.normalize(k1, dim=-1)
        k2 = torch.nn.functional.normalize(k2, dim=-1)
        attn1 = (q1 @ k1.transpose(-2, -1))
        attn1 = attn1.softmax(dim=-1)
        out3 = (attn1 @ v1) + q1
        attn2 = (q2 @ k2.transpose(-2, -1))
        attn2 = attn2.softmax(dim=-1)
        out4 = (attn2 @ v2) + q2
        out3 = rearrange(out3, 'b head h (w c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out4 = rearrange(out4, 'b head w (h c) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out3) + self.project_out(out4) + x1 + x2

        return out


if __name__ == '__main__':
    CFFM = CFFM(dim=16)  # 指定通道数
    x1 = torch.randn(16, 16, 16, 16)  # b c h w  输入
    x2 = torch.randn(16, 16, 16, 16)

    output = CFFM(x1, x2)
    print(output.size())  # torch.Size([3, 32, 64, 64])
