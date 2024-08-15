
import numpy as np
import pytest
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import graph_pooling as gp
from scipy.spatial import cKDTree  # Import KD-tree from scipy
import torch
import torch.optim as optim

@pytest.mark.torch
def knn(x, k):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    idx = []
    for i in range(batch_size):
        points = x[i].transpose(0, 1).detach().cpu().numpy()
 # Convert to NumPy array
        tree = cKDTree(points)
        distances, neighbors = tree.query(points, k=k + 1)  # k+1 to exclude the point itself
        neighbors = neighbors[:, 1:]  # Exclude the first neighbor (the point itself)
        idx.append(neighbors)

    idx = torch.tensor(idx, dtype=torch.long, device=x.device)

    return idx

class StarTransformer(nn.Module):
    r"""
    **Star-Transformer** 的 encoder 部分。输入 3d 的文本输入，返回相同长度的文本编码。
    基于论文 `Star-Transformer <https://arxiv.org/abs/1902.09113>`_

    :param hidden_size: 输入维度的大小，同时也是输出维度的大小。
    :param num_layers: **Star-Transformer** 的层数
    :param num_head: **多头注意力** head 的数目，需要能被 ``d_model`` 整除
    :param head_dim: 每个 ``head`` 的维度大小。
    :param dropout: dropout 概率
    :param max_len: 如果为 :class:`int` 表示输入序列的最大长度，模型会为输入序列加上 ``position embedding``；
        若为 ``None`` 则会跳过此步骤。
    """

    def __init__(self, hidden_size: int, num_layers: int, num_head: int, head_dim: int, dropout: float=0.05, max_len: int=None):
        super(StarTransformer, self).__init__()
        self.iters = num_layers

        self.norm = nn.ModuleList([nn.LayerNorm(hidden_size, eps=1e-6) for _ in range(self.iters)])
        # self.emb_fc = nn.Conv2d(hidden_size, hidden_size, 1)
        self.emb_drop = nn.Dropout(dropout)
        self.ring_att = nn.ModuleList(
            [_MSA1(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.05)
             for _ in range(self.iters)])
        self.star_att = nn.ModuleList(
            [_MSA2(hidden_size, nhead=num_head, head_dim=head_dim, dropout=0.05)
             for _ in range(self.iters)])



    def forward(self, data: torch.FloatTensor, relay: torch.FloatTensor, mask: torch.ByteTensor):
        r"""
        :param data: 输入序列，形状为 ``[batch_size, length, hidden]``
        :param mask: 输入序列的 padding mask， 形状为 ``[batch_size, length]`` , 为 **0** 的地方为 padding
        :return: 返回一个元组，第一个元素形状为 ``[batch_size, length, hidden]`` ，代表编码后的输出序列；
            第二个元素形状为 ``[batch_size, hidden]``，表示全局 relay 节点, 详见论文。
        """
        def norm_func(f, x):
            # B, H, L, 1
            return f(x.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)

        B, N, L, H = data.size()
        if mask is not None:
            mask = (mask.eq(False))  # flip the mask for masked_fill_
            smask = torch.cat([torch.zeros(B, 1, ).byte().to(mask), mask], 1)

        embs = data.permute(0, 3, 1, 2)[:, :, :, :, None]  # B H L N 1

        embs = norm_func(self.emb_drop, embs)
        nodes = embs
        relay = relay.permute(0,3,1,2)[:, :, :, :, None]
        if mask is not None:
            ex_mask = mask[:, None, :, None].expand(B, H, L, 1)
        r_embs = embs.reshape(B, H, N, 1, L)
        for i in range(self.iters):
            ax = torch.cat([r_embs, relay.expand(B, H, N, 1, L)], 3)
            nodes = F.leaky_relu(self.ring_att[i](norm_func(self.norm[i], nodes), ax=ax))
            if mask is not None:
                relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 3), smask))
            else:
                relay = F.leaky_relu(self.star_att[i](relay, torch.cat([relay, nodes], 3), None))
            if mask is not None:
                nodes = nodes.masked_fill_(ex_mask, 0)

        nodes = nodes.view(B, H, N, L).permute(0, 2, 3, 1)
        relay = relay.view(B, H, N, 1).permute(0, 2, 3, 1)

        return nodes, relay


class _MSA1(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        super(_MSA1, self).__init__()
        # Multi-head Self Attention Case 1, doing self-attention for small regions
        # Due to the architecture of GPU, using hadamard production and summation are faster than dot production when unfold_size is very small
        self.WQ = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv3d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, ax=None):
        # x: B, H, L, 1, ax : B, H, X, L append features
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, N, L, _ = x.shape

        q, k, v = self.WQ(x), self.WK(x), self.WV(x)  # x: (B,H,L,1)

        if ax is not None:
            aL = ax.shape[3]
            ak = self.WK(ax).view(B, nhead, head_dim, N, aL, L)
            av = self.WV(ax).view(B, nhead, head_dim, N, aL, L)
        q = q.view(B, nhead, head_dim, N, 1, L)
        #k = k.view(B, nhead, head_dim, N, 1, L)
        #v = v.view(B, nhead, head_dim, N, 1, L)
        k = F.unfold(k.view(B, nhead * head_dim, N, L), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, N, unfold_size, L)
        v = F.unfold(v.view(B, nhead * head_dim, N, L), (unfold_size, 1), padding=(unfold_size // 2, 0)) \
            .view(B, nhead, head_dim, N, unfold_size, L)
        if ax is not None:
            k = torch.cat([k, ak], 4)
            v = torch.cat([v, av], 4)

        alphas = self.drop(F.softmax((q * k).sum(2, keepdim=True) / np.sqrt(head_dim), 4))  # B N L 1 U
        att = (alphas * v).sum(4).view(B, nhead * head_dim, N, L, 1)

        ret = self.WO(att)

        return ret


class _MSA2(nn.Module):
    def __init__(self, nhid, nhead=10, head_dim=10, dropout=0.1):
        # Multi-head Self Attention Case 2, a broadcastable query for a sequence key and value
        super(_MSA2, self).__init__()
        self.WQ = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WK = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WV = nn.Conv3d(nhid, nhead * head_dim, 1)
        self.WO = nn.Conv3d(nhead * head_dim, nhid, 1)

        self.drop = nn.Dropout(dropout)

        # print('NUM_HEAD', nhead, 'DIM_HEAD', head_dim)
        self.nhid, self.nhead, self.head_dim, self.unfold_size = nhid, nhead, head_dim, 3

    def forward(self, x, y, mask=None):
        # x: B, H, 1, 1, 1 y: B H L 1
        nhid, nhead, head_dim, unfold_size = self.nhid, self.nhead, self.head_dim, self.unfold_size
        B, H, N, L, _ = y.shape

        q, k, v = self.WQ(x), self.WK(y), self.WV(y)

        q = q.view(B, nhead, N, 1, head_dim)  # B, H, 1, 1 -> B, N, 1, h
        k = k.view(B, nhead, N, head_dim, L)  # B, H, L, 1 -> B, N, h, L
        v = v.view(B, nhead, N, head_dim, L).permute(0, 1, 2, 4, 3)  # B, H, L, 1 -> B, N, L, h
        pre_a = torch.matmul(q, k) / np.sqrt(head_dim)
        if mask is not None:
            pre_a = pre_a.masked_fill(mask[:, None, None, :], -float('inf'))
        alphas = self.drop(F.softmax(pre_a, 3))  # B, N, 1, L
        att = torch.matmul(alphas, v).view(B, nhead*head_dim, N, 1, 1)  # B, N, 1, h -> B, N*h, 1, 1
        return self.WO(att)





def get_graph_feature(x, k=20, idx=None, dim6=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim6 == False:
            idx = knn(x, k=k)   # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 0:3], k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


def get_shuang_feature(x, k=20, idx=None, dim6=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim6 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 0:3], k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    norm_x = torch.norm(x, p=2, dim=1, keepdim=True)
    x_tanh_norm = torch.tanh(norm_x) / norm_x
    x1= x_tanh_norm * x  #x1

    norm_f = torch.norm(feature, p=2, dim=1, keepdim=True)
    f_tanh_norm = torch.tanh(norm_f) / norm_f
    feature1 = f_tanh_norm * feature  # feature1

    feature1 = torch.cat((feature1 - x1, x1), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature1,idx


class AdaptiveConv(nn.Module):
    def __init__(self, k, in_channels, feat_channels, nhiddens, out_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.nhiddens = nhiddens
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.k = k

        self.conv0 = nn.Conv2d(feat_channels, nhiddens, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(nhiddens, nhiddens*in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(nhiddens)
        self.bn1 = nn.BatchNorm2d(nhiddens)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.residual_layer = nn.Sequential(nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            )
        self.linear = nn.Sequential(nn.Conv2d(nhiddens, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, points, feat, idx):
        # points: (bs, in_channels, num_points), feat: (bs, feat_channels/2, num_points)
        batch_size, _, num_points = points.size()

        x, _ = get_graph_feature(points, k=self.k, idx=idx) # (bs, in_channels, num_points, k)
        y, _ = get_graph_feature(feat, k=self.k, idx=idx) # (bs, feat_channels, num_points, k)
        x1, _ = get_shuang_feature(points, k=self.k, idx=idx)  # (bs, in_channels, num_points, k)
        y1, _ = get_shuang_feature(feat, k=self.k, idx=idx)  # (bs, feat_channels, num_points, k)

        x1 = x.permute(0, 2, 3, 1)  # (bs, num_points, k, in_channels)
        y1 = x.permute(0, 2, 3, 1)  # (bs, num_points, k, feat_channels)
        relay = x1.unsqueeze(2)
        model2 = StarTransformer(num_layers=6, hidden_size=100, num_head=8, head_dim=20, max_len=None)

        x_emb, x_relay= model2(x1, relay, mask=None) # (bs, num_points, k, in_channels) (bs, num_points, 1, in_channels)
        x_relay=x_relay.squeeze(2)  # (bs, num_points, in_channels)
        x_relay = x_relay.permute(0, 2, 1)   # (bs, in_channels,,num_points)


        kernel = self.conv0(y) # (bs, nhiddens, num_points, k)
        kernel = self.leaky_relu(self.bn0(kernel))
        kernel = self.conv1(kernel) # (bs, in*nhiddens, num_points, k)
        kernel = kernel.permute(0, 2, 3, 1).view(batch_size, num_points, self.k, self.nhiddens, self.in_channels) # (bs, num_points, k, nhiddens, in)



        x = x.permute(0, 2, 3, 1).unsqueeze(4) # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(kernel, x).squeeze(4) # (bs, num_points, k, nhiddens)
        x = x.permute(0, 3, 1, 2).contiguous() # (bs, nhiddens, num_points, k)

        # nhiddens -> out_channels
        x = self.leaky_relu(self.bn1(x))
        x = self.linear(x) # (bs, out_channels, num_points, k)
        # residual: feat_channels -> out_channels
        residual = self.residual_layer(y)
        x += residual
        x = self.leaky_relu(x)

        x = x.max(dim=-1, keepdim=False)[0] # (bs, out_channels, num_points)
        B, O, N = x.size()
        x2= x_relay.view(B,O,N)
        combined_x = torch.cat((x, x2), dim=3)  # 沿着第4维度拼接

        # 将拼接后的数据乘以门控的转换矩阵，并经过softmax函数得到门控权重
        combined_x_reshaped = combined_x.view(batch_size * num_points * self.k, -1)
        gated_output = torch.matmul(combined_x_reshaped, self.gated_transform)
        gated_output = gated_output.view(batch_size, num_points, self.k, -1)

        # 通过softmax函数获取门控权重
        gated_weights = nn.functional.softmax(gated_output, dim=-1)

        norm_x1 = torch.norm(x1, p=2, dim=1, keepdim=True)
        x_h =  x1 /torch.tanh(norm_x1)
        x1 =x_h /norm_x1
        gate = x * gated_weights + x1 * (1 - gated_weights)

        return gate








class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, k):
        super(GraphConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x, idx):
        # x: (bs, in_channels, num_points)
        x, _ = get_graph_feature(x, k=self.k, idx=idx) # (bs, in_channels, num_points, k)
        x = self.conv(x) # (bs, out_channels, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0] # (bs, out_channels, num_points)

        return x

class ConvLayer(nn.Module):
    def __init__(self, para, k, in_channels, feat_channels):
        super(ConvLayer, self).__init__()
        self.type = para[0]
        self.out_channels = para[1]
        self.k = k
        if self.type == 'adapt':
            self.layer = AdaptiveConv(k, in_channels, feat_channels, nhiddens=para[2], out_channels=para[1])
        elif self.type == 'graph':
            self.layer = GraphConv(feat_channels, self.out_channels, k)
        elif self.type == 'conv1d':
            self.layer = nn.Sequential(nn.Conv1d(int(feat_channels/2), self.out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm1d(self.out_channels),
                                    nn.LeakyReLU(negative_slope=0.2))
        else:
            raise ValueError('Unknown convolution layer: {}'.format(self.type))

    def forward(self, points, x, idx):
        # points: (bs, 3, num_points), x: (bs, feat_channels/2, num_points)
        if self.type == 'conv1d':
            x = self.layer(x)
            x = x.max(dim=-1, keepdim=False)[0] # (bs, num_dims)
        elif self.type == 'adapt':
            x = self.layer(points, x, idx)
        elif self.type == 'graph':
            x = self.layer(x, idx)

        return x


class Transform_Net(nn.Module):
    def __init__(self, in_channels=6, out_channels=3):
        super(Transform_Net, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, out_channels*out_channels)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(out_channels, out_channels))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, self.out_channels, self.out_channels)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class Net(nn.Module):
    def __init__(self, args, class_num, cat_num, use_stn=True):
        super(Net, self).__init__()
        self.args = args
        self.k = args.k
        self.class_num = class_num
        self.cat_num = cat_num
        self.use_stn = use_stn

        # architecture
        self.in_channels = 6
        self.forward_para = [['adapt', 64, 64], 
                            ['adapt', 64, 64], 
                            ['pool', 4], 
                            ['adapt', 128, 64],
                            ['pool', 4], 
                            ['adapt', 256, 64], 
                            ['pool', 2], 
                            ['graph', 512],
                            ['conv1d', 1024]]
        self.agg_channels = 0

        # layers
        self.forward_layers = nn.ModuleList()
        feat_channels = 12
        for i, para in enumerate(self.forward_para):
            if para[0] == 'pool':
                self.forward_layers.append(gp.Pooling_fps(pooling_rate=para[1], neighbor_num=self.k))
            else:
                self.forward_layers.append(ConvLayer(para, self.k, self.in_channels, feat_channels))
                self.agg_channels += para[1]
                feat_channels = para[1]*2
        
        self.agg_channels += 64

        self.conv_onehot = nn.Sequential(nn.Conv1d(cat_num, 64, kernel_size=1, bias=False),
                                   nn.BatchNorm1d(64),
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv1d = nn.Sequential(
            nn.Conv1d(self.agg_channels, 512, kernel_size=1),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Conv1d(512, 256, kernel_size=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Conv1d(256, class_num, kernel_size=1),
            )

        if self.use_stn:
            self.stn = Transform_Net(in_channels=12, out_channels=3)


    def forward(self, x, onehot):
        # x: (bs, num_points, 6), onehot: (bs, cat_num)
        x = x.permute(0, 2, 1).contiguous() # (bs, 6, num_points)
        batch_size = x.size(0)
        num_points = x.size(2)

        if self.use_stn:
            x0, _ = get_graph_feature(x, k=self.k)
            t = self.stn(x0)
            p1 = torch.bmm(x[:,0:3,:].transpose(2, 1), t) # (bs, num_points, 3)
            p2 = torch.bmm(x[:,3:6,:].transpose(2, 1), t)
            x = torch.cat((p1, p2), dim=2).transpose(2, 1).contiguous() # (bs, 6, num_points)
        points = x[:,0:3,:] # (bs, 3, num_points)
        
        # forward
        feat_forward = []
        points_forward = [points]
        _, idx = get_graph_feature(points, k=self.k)
        for i, block in enumerate(self.forward_layers):
            if self.forward_para[i][0] == 'pool':
                points, x = block(points, x, idx)
                points_forward.append(points)
                _, idx = get_graph_feature(points, k=self.k)
            elif self.forward_para[i][0] == 'conv1d':
                x = block(points, x, idx)
                x = x.unsqueeze(2).repeat(1, 1, num_points)
                feat_forward.append(x)
            else:
                x = block(points, x, idx)
                feat_forward.append(x)

        # onehot
        onehot = onehot.unsqueeze(2)
        onehot_expand = self.conv_onehot(onehot)
        onehot_expand = onehot_expand.repeat(1, 1, num_points)

        # aggregating features from all layers
        x_agg = []
        points0 = points_forward.pop(0)
        points = None
        for i, para in enumerate(self.forward_para):
            if para[0] == 'pool':
                points = points_forward.pop(0)
            else:
                x = feat_forward.pop(0)
                if x.size(2) == points0.size(2):
                    x_agg.append(x)
                    continue
                idx = gp.get_nearest_index(points0, points)
                x_upsample = gp.indexing_neighbor(x, idx).squeeze(3)
                x_agg.append(x_upsample)
        x = torch.cat(x_agg, dim=1)
        x = torch.cat((x, onehot_expand), dim=1)
        x = self.conv1d(x)
        x = x.permute(0, 2, 1).contiguous() # (bs, num_points, class_num)

        return x


