from torch import nn
import torch
import torch.nn.functional as F

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

class DWConv(nn.Module):
    def __init__(self, dim, H, W):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.H = H
        self.W = W
    def forward(self, x):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, self.H, self.W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class endecoder2(nn.Module):
    def __init__(self, d_model, heads, dropout, activation, flag):
        super(endecoder2, self).__init__()
        self.activition = _get_activation_fn(activation)
        self.flag = flag

        self.linear_q1 = nn.Linear(d_model, d_model)
        self.linear_k1 = nn.Linear(d_model, d_model)
        self.linear_v1 = nn.Linear(d_model, d_model)
        self.linear1_1 = nn.Linear(d_model, 2 * d_model)
        self.linear1_2 = nn.Linear(2 * d_model, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout1_1 = nn.Dropout(dropout)
        self.dropout1_2 = nn.Dropout(dropout)
        self.multihead_attn1 = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm1_1 = nn.LayerNorm(d_model)
        self.norm1_2 = nn.LayerNorm(d_model)
        self.norm1_3 = nn.LayerNorm(d_model)

        self.linear_q2 = nn.Linear(d_model, d_model)
        self.linear_k2 = nn.Linear(d_model, d_model)
        self.linear_v2 = nn.Linear(d_model, d_model)
        self.linear2_1 = nn.Linear(d_model, 2 * d_model)
        self.linear2_2 = nn.Linear(2 * d_model, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout2_1 = nn.Dropout(dropout)
        self.dropout2_2 = nn.Dropout(dropout)
        self.multihead_attn2 = nn.MultiheadAttention(d_model, heads, batch_first=True)
        self.norm2_1 = nn.LayerNorm(d_model)
        self.norm2_2 = nn.LayerNorm(d_model)
        self.norm2_3 = nn.LayerNorm(d_model)

    def forward(self, x):
        rgb1, depth1 = x
        rediual = rgb1
        rediual_d = depth1
        if self.flag == 1:
            rgb1 = self.norm1_1(rgb1)
            depth1 = self.norm2_1(depth1)
        q = self.linear_q1(rgb1)
        k = self.linear_k1(depth1)
        v = self.linear_v1(depth1)

        q2 = self.linear_q2(depth1)
        k2 = self.linear_k2(rgb1)
        v2 = self.linear_v2(rgb1)

        k = torch.cat((k, k2), dim=1)
        v = torch.cat((v, v2), dim=1)

        src1, src1_1 = self.multihead_attn1(q, k, v)
        res = rediual + self.dropout1_1(src1)
        res1 = self.norm1_2(res)
        res1 = self.linear1_2(self.dropout1(self.activition(self.linear1_1(res1))))
        res2 = res + self.dropout1_2(res1)

        src2, src2_2 = self.multihead_attn2(q2, k, v)
        res3 = rediual_d + self.dropout2_1(src2)
        res4 = self.norm2_2(res3)
        res4 = self.linear2_2(self.dropout2(self.activition(self.linear2_1(res4))))
        res5 = res3 + self.dropout2_2(res4)

        return res2, res5

class interactive(nn.Module):
    def __init__(self, n, d_model, heads, dropout, activation, pos_feats, num_pos_feats, ratio):
        super(interactive, self).__init__()
        self.trans = []
        self.conv1 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.conv2 = nn.Conv2d(d_model, d_model//ratio, kernel_size=(1, 1), stride=(1, 1))
        self.conv3 = nn.Conv2d(d_model//ratio, d_model, kernel_size=(1, 1), stride=(1, 1))
        self.conv4 = nn.Conv2d(d_model//ratio, d_model, kernel_size=(1, 1), stride=(1, 1))
        flag1 = 0
        for i in range(n):
            if flag1 == 0:
                self.trans.append(endecoder2(d_model//ratio, heads, dropout, activation, 0).to(device=1))
                flag1 += 1
            elif flag1 > 0:
                self.trans.append(endecoder2(d_model//ratio, heads, dropout, activation, 0).to(device=1))

        self.transall = nn.Sequential(*self.trans)
        total_params1 = sum(p.numel() for p in self.transall.parameters())
        print('总参数量：{}'.format(total_params1))

    def forward(self, rgb, depth):
        n, c, h, w = rgb.size()
        rgb = self.conv1(rgb)
        depth = self.conv2(depth)
        rgb1 = torch.flatten(rgb, start_dim=2, end_dim=3).permute(0, 2, 1)
        depth1 = torch.flatten(depth, start_dim=2, end_dim=3).permute(0, 2, 1)

        x = self.transall((rgb1, depth1))
        rgb1, depth1 = x
        res = rgb1.permute(0, 2, 1)
        res1 = depth1.permute(0, 2, 1)
        output = res.reshape(n, c//2, h, w)
        output1 = res1.reshape(n, c//2, h, w)
        output = self.conv3(output)
        output1 = self.conv4(output1)

        return output, output1

class interactive1_2(nn.Module):
    def __init__(self, d_model, ratio):
        super(interactive1_2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(d_model, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(d_model//ratio, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(d_model//ratio, d_model//ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
                                   nn.BatchNorm2d(d_model//ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(d_model, d_model // ratio, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), dilation=(3, 3)),
            nn.BatchNorm2d(d_model // ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // ratio, d_model // ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model // ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(d_model // ratio, d_model // ratio, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(d_model // ratio, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True))

    def forward(self, rgb, depth):
        inp = torch.cat((rgb, depth), dim=1)
        output1 = self.conv1(inp)
        output2 = self.conv2(inp)

        return output1, output2
