import re
import torch
import torch.nn as nn
import torch.nn.functional as F


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        # modules = [nn.Linear(243, 243)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_depth_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_depth_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
        # return nn.Linear(config.hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size*2, config.hidden_size)]
        # modules = [nn.Linear(config.hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')



class MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, depth_features, visual_features):
        batch_size = depth_features.size(0)
        
        # 线性变换并分头
        queries = self.query_proj(depth_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        keys = self.key_proj(visual_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)      # 同上
        values = self.value_proj(visual_features).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # 同上

        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch_size, num_heads, seq_len, seq_len]
        attention_weights = self.softmax(attention_scores)  # [batch_size, num_heads, seq_len, seq_len]

        # 计算加权和
        attention_output = torch.matmul(attention_weights, values)  # [batch_size, num_heads, seq_len, head_dim]
        
        # 拼接所有头的输出
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # [batch_size, seq_len, d_model]

        # 通过线性层变换
        output_features = self.out_proj(attention_output)  # [batch_size, seq_len, d_model]
        
        return output_features

class FeedForwardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim):
        super(TransformerBlock, self).__init__()
        self.multi_head_cross_attention = MultiHeadCrossAttention(d_model, num_heads)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardNetwork(d_model, hidden_dim, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        
    def forward(self, depth_features, visual_features):
        # 多头交叉注意力层
        attention_output = self.multi_head_cross_attention(depth_features, visual_features)
        attention_output = self.layer_norm1(depth_features + attention_output)  # 残差连接和层归一化
        
        # 前馈网络层
        ff_output = self.feed_forward(attention_output)
        output = self.layer_norm2(attention_output + ff_output)  # 残差连接和层归一化
        
        return output

class MultiLayerTransformer(nn.Module):
    def __init__(self, d_model, num_heads, hidden_dim, num_layers, output_dim):
        super(MultiLayerTransformer, self).__init__()
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads, hidden_dim) for _ in range(num_layers)])
        self.mlp = FeedForwardNetwork(d_model, hidden_dim, output_dim)
        
    def forward(self, depth_features, visual_features):
        for layer in self.layers:
            depth_features = layer(depth_features, visual_features)
        output_features = self.mlp(depth_features)
        return output_features

class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim
        
    def forward(self, x1, x2):
        return torch.cat((x1, x2), dim=self.dim)


def build_share_projector(config, delay_load=False, **kwargs):
    concat = Concat(dim=-1)
    projector_type = getattr(config, 'mm_share_projector_type', 'linear')

    if projector_type == 'linear':
        modules = [concat, nn.Linear(config.hidden_size*2, config.hidden_size)]
        return nn.Sequential(*modules)
    
    if projector_type == 'transformer':
        return MultiLayerTransformer(d_model=config.hidden_size, \
                                         num_heads=2, hidden_dim=896, \
                                         num_layers=1, output_dim=config.hidden_size)
        
    # if projector_type == 'transformer':
    #     return MultiLayerTransformer(d_model=config.hidden_size, \
    #                                      num_heads=2, hidden_dim=config.hidden_size, \
    #                                      num_layers=1, output_dim=config.hidden_size)
        # return nn.Sequential(*modules)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        # modules = [nn.Linear(config.hidden_size*2, config.hidden_size)]
        modules = [concat, nn.Linear(config.mm_hidden_size*2, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    raise ValueError(f'Unknown projector type: {projector_type}')

# # 假设输入特征
# batch_size, seq_len, d_model = 16, 729, 1152
# num_heads = 8
# hidden_dim = 2048  # 通常在Transformer中，hidden_dim比d_model大，例如2048
# output_dim = 862

# depth_features = torch.randn(batch_size, seq_len, d_model)
# visual_features = torch.randn(batch_size, seq_len, d_model)

# # 初始化MultiHeadCrossAttention模块
# multi_head_cross_attention = MultiHeadCrossAttention(d_model, num_heads)

# # 初始化FeedForwardNetwork模块
# ffn = FeedForwardNetwork(d_model, hidden_dim, output_dim)

# # 前向传播
# attention_output = multi_head_cross_attention(depth_features, visual_features)
# output_features = ffn(attention_output)

# print(output_features.shape)  # 应该是 [16, 729, 862]