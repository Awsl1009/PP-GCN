import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import warnings

# 忽略所有警告
warnings.filterwarnings("ignore")


class TextCLIP(nn.Module):
    def __init__(self, model):
        super(TextCLIP, self).__init__()
        self.model = model

    def forward(self, text):
        return self.model.encode_text(text)


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(
        2. / (n * k1 * k2 * branches)))  # 使用正态分布初始化权重，一种专为ReLU激活函数设计的权重初始化方法，He初始化通过调整权重的初始值来保持激活值的方差，从而避免梯度消失问题
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1, 2, 3, 4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            )
            for ks, dilation in zip(kernel_size, dilations)
        ])

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels)
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)

        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class CTRGC(nn.Module):
    def __init__(self, in_channels, out_channels, rel_reduction=8, mid_reduction=1):
        super(CTRGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if in_channels == 2 or in_channels == 3 or in_channels == 9:  # change to 2, because the origin data is 2D-Pose.
            self.rel_channels = 8
            self.mid_channels = 16
        else:
            self.rel_channels = in_channels // rel_reduction
            self.mid_channels = in_channels // mid_reduction
        self.conv1 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv2 = nn.Conv2d(self.in_channels, self.rel_channels, kernel_size=1)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)
        self.conv4 = nn.Conv2d(self.rel_channels, self.out_channels, kernel_size=1)
        self.tanh = nn.Tanh()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def forward(self, x, A=None, alpha=1):
        x1, x2, x3 = self.conv1(x).mean(-2), self.conv2(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x2.unsqueeze(-2))
        x1 = self.conv4(x1) * alpha + (A.unsqueeze(0).unsqueeze(0) if A is not None else 0)  # N,C,V,V
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        return x1


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=9, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1))

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, adaptive=True, residual=True):
        super(unit_gcn, self).__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.out_c = out_channels
        self.in_c = in_channels
        self.adaptive = adaptive
        self.num_subset = A.shape[0]
        self.convs = nn.ModuleList()
        for i in range(self.num_subset):
            self.convs.append(CTRGC(in_channels, out_channels))

        if residual:
            if in_channels != out_channels:
                self.down = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                self.down = lambda x: x
        else:
            self.down = lambda x: 0
        if self.adaptive:
            self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)))
        else:
            self.A = Variable(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.bn = nn.BatchNorm2d(out_channels)
        self.soft = nn.Softmax(-2)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    def forward(self, x):
        y = None
        if self.adaptive:
            A = self.PA
        else:
            A = self.A.cuda(x.get_device())
        for i in range(self.num_subset):
            z = self.convs[i](x, A[i], self.alpha)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        y = self.relu(self.tcn1(self.gcn1(x)) + self.residual(x))
        return y


class Model(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=3,
                 drop_out=0, adaptive=True):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)  #2,3,6,9
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N是批次数量，C是坐标数，T是帧数，V是骨架数，M是人数
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)  # BM C T N
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # 64
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)  # 128
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # 256

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x)


class Model_4part(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 drop_out=0, adaptive=True, head=['ViT-L/14']):
        super(Model_4part, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)  # 2,3,6,9
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))

        # self.part_list = nn.ModuleList()

        # for i in range(4):
        #     self.part_list.append(nn.Linear(256, 512))

        self.part_list = nn.ModuleList([nn.Linear(256, 768) for _ in range(4)])

        self.head = head

        for head in self.head:
            if head == 'ViT-L/14':
                self.linear_head[head] = nn.Linear(256, 768)
                conv_init(self.linear_head[head])

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N是批次数量，C是坐标数，T是帧数，V是骨架数，M是人数
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # 64
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)  # 128
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # 256

        # N*M,C,T,V
        c_new = x.size(1)

        feature = x.view(N, M, c_new, T // 4, V)
        head_list = torch.Tensor([0, 1, 2, 3, 4]).long()
        hand_arm_list = torch.Tensor([5, 6, 7, 8, 9, 10, 11]).long()
        hip_list = torch.Tensor([11, 12]).long()
        leg_foot_list = torch.Tensor([13, 14, 15, 16]).long()
        head_feature = self.part_list[0](feature[:, :, :, :, head_list].mean(4).mean(3).mean(1))
        hand_arm_feature = self.part_list[1](feature[:, :, :, :, hand_arm_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:, :, :, :, hip_list].mean(4).mean(3).mean(1))
        leg_foot_feature = self.part_list[3](feature[:, :, :, :, leg_foot_list].mean(4).mean(3).mean(1))

        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()

        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_arm_feature, hip_feature,
                                                            leg_foot_feature]


class Model_4part_Angle(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 num_angle_features=1, drop_out=0, adaptive=True, head=['ViT-L/14']):
        super(Model_4part_Angle, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # 用于计算角度特征
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

        # 初始化邻接矩阵
        A = self.graph.A

        # 输入通道动态计算
        total_in_channels = in_channels + num_angle_features

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 增加的通道压缩层
        # self.channel_compress = nn.Linear(total_in_channels, in_channels)  # 将3压缩回2
        self.channel_compress = nn.Sequential(
            nn.Conv1d(in_channels=6, out_channels=2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)  # 2,3,6,9
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))

        # self.part_list = nn.ModuleList()

        # for i in range(4):
        #     self.part_list.append(nn.Linear(256, 512))

        self.part_list = nn.ModuleList([nn.Linear(256, 768) for _ in range(4)])

        self.head = head
        for head in self.head:
            if head == 'ViT-L/14':
                self.linear_head[head] = nn.Linear(256, 768)
                conv_init(self.linear_head[head])

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def preprocessing(self, x):
        """
        计算二维骨架的角度特征并拼接到原始输入，同时压缩通道数。
        """
        N, C, T, V, M = x.size()

        # 骨架连接点对，需根据实际数据集进行调整
        dgait_bone_angle_pairs = {
            1: (2, 3),
            2: (1, 4),
            3: (1, 5),
            4: (2, 6),
            5: (3, 7),
            6: (8, 12),
            7: (9, 13),
            8: (6, 10),
            9: (7, 11),
            10: (10, 10),
            11: (11, 11),
            12: (6, 14),
            13: (7, 15),
            14: (12, 16),
            15: (13, 17),
            16: (16, 16),
            17: (17, 17)
        }

        # 初始化不同角度特征的列表
        joint_list_bone_angle = []
        left_hand_angle = []
        right_hand_angle = []
        two_hand_angle = []
        two_elbow_angle = []
        two_knee_angle = []
        two_feet_angle = []

        # 存放所有角度特征
        angle_features = [joint_list_bone_angle, left_hand_angle, right_hand_angle, two_hand_angle, two_elbow_angle,
                          two_knee_angle, two_feet_angle]  # 初始化角度特征列表

        # 计算所有角度特征
        for joint, (v1, v2) in dgait_bone_angle_pairs.items():

            # 计算所有相邻骨骼角度
            vec1 = x[:, :2, :, v1 - 1, :] - x[:, :2, :, joint - 1, :]  # 向量 1 (选择 x, y)
            vec2 = x[:, :2, :, v2 - 1, :] - x[:, :2, :, joint - 1, :]  # 向量 2 (选择 x, y)
            angle_feature = (1.0 - self.cos(vec1, vec2))  # 计算角度特征
            angle_feature[angle_feature != angle_feature] = 0  # 去除 NaN 值
            joint_list_bone_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))  # 保持通道维度

            # 计算左手角度
            vec1 = x[:, :2, :, 9 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 11 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            left_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算右手角度
            vec1 = x[:, :2, :, 8 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 10 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            right_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两手角度
            vec1 = x[:, :2, :, 11 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 10 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两肘角度
            vec1 = x[:, :2, :, 9 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 8 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_elbow_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两膝角度
            vec1 = x[:, :2, :, 15 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 14 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_knee_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两脚角度
            vec1 = x[:, :2, :, 17 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 16 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_feet_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

        # 将每个角度特征列表拼接到一起
        for angle_features_id in range(len(angle_features)):
            angle_features[angle_features_id] = torch.cat(angle_features[angle_features_id], dim=3)

        # 沿通道维度拼接所有角度特征
        angle_features = torch.cat(angle_features, dim=1)

        # 确保与 x 的设备一致
        angle_features = angle_features.to(x.device)

        # 将角度特征拼接到原始输入数据 (通道维度增加)
        x = torch.cat((x, angle_features), dim=1)  # 通道数变为9

        # # Joint + All_Bone_Angle
        # x = x[:, :3, :, :, :]  # 只保留角度特征 (N, 3, T, V, M)

        # Joint + All_Pairs_Angle
        x = torch.cat((x[:, :2, :, :, :], x[:, 5:9, :, :, :]), dim=1)  # 只保留角度特征 (N, 3, T, V, M)

        # 通道压缩
        x = x.permute(0, 4, 3, 2, 1).reshape(N * M * V, T, 6)  # (N*M*V, T, 3)
        x = x.permute(0, 2, 1)  # 重新排列维度，确保通道数在第二维，时间步数在第三维
        x = self.channel_compress(x)  # 压缩 (N*M*V, T, 2)
        # x = x.permute(0, 2, 1).view(N, M, V, 2, T).permute(0, 3, 4, 2, 1)  # 恢复形状 (N, 2, T, V, M)
        x = x.permute(0, 2, 1).contiguous().view(N, M, V, 2, T).permute(0, 3, 4, 2, 1)  # 恢复形状 (N, 2, T, V, M)

        return x

    def forward(self, x):
        # 添加角度特征并压缩通道
        x = self.preprocessing(x)

        N, C, T, V, M = x.size()

        # 数据归一化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N是批次数量，C是坐标数，T是帧数，V是骨架数，M是人数
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 网络层
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # 64
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)  # 128
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # 256

        # 特征池化
        c_new = x.size(1)
        feature = x.view(N, M, c_new, T // 4, V)

        # 部分特征计算
        head_list = torch.Tensor([0, 1, 2, 3, 4]).long()
        hand_arm_list = torch.Tensor([5, 6, 7, 8, 9, 10, 11]).long()
        hip_list = torch.Tensor([11, 12]).long()
        leg_foot_list = torch.Tensor([13, 14, 15, 16]).long()

        head_feature = self.part_list[0](feature[:, :, :, :, head_list].mean(4).mean(3).mean(1))
        hand_arm_feature = self.part_list[1](feature[:, :, :, :, hand_arm_list].mean(4).mean(3).mean(1))
        hip_feature = self.part_list[2](feature[:, :, :, :, hip_list].mean(4).mean(3).mean(1))
        leg_foot_feature = self.part_list[3](feature[:, :, :, :, leg_foot_list].mean(4).mean(3).mean(1))

        # 全局池化
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        feature_dict = dict()
        for name in self.head:
            feature_dict[name] = self.linear_head[name](x)

        x = self.drop_out(x)

        return self.fc(x), feature_dict, self.logit_scale, [head_feature, hand_arm_feature, hip_feature,
                                                            leg_foot_feature]

class Model_Only_Angle(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 num_angle_features=1, drop_out=0, adaptive=True):
        super(Model_Only_Angle, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        # 用于计算角度特征
        self.cos = nn.CosineSimilarity(dim=1, eps=0)

        # 初始化邻接矩阵
        A = self.graph.A

        # 输入通道动态计算
        total_in_channels = in_channels + num_angle_features

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        # 增加的通道压缩层
        self.channel_compress = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=2, kernel_size=1, bias=False),
            nn.BatchNorm1d(2),
            nn.ReLU()
        )

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)  #2,3,6,9
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def preprocessing(self, x):
        """
        计算二维骨架的角度特征并拼接到原始输入，同时压缩通道数。
        """
        N, C, T, V, M = x.size()

        # 骨架连接点对，需根据实际数据集进行调整
        dgait_bone_angle_pairs = {
            1: (2, 3),
            2: (1, 4),
            3: (1, 5),
            4: (2, 6),
            5: (3, 7),
            6: (8, 12),
            7: (9, 13),
            8: (6, 10),
            9: (7, 11),
            10: (10, 10),
            11: (11, 11),
            12: (6, 14),
            13: (7, 15),
            14: (12, 16),
            15: (13, 17),
            16: (16, 16),
            17: (17, 17)
        }

        # 初始化不同角度特征的列表
        joint_list_bone_angle = []
        left_hand_angle = []
        right_hand_angle = []
        two_hand_angle = []
        two_elbow_angle = []
        two_knee_angle = []
        two_feet_angle = []

        # 存放所有角度特征
        angle_features = [joint_list_bone_angle, left_hand_angle, right_hand_angle, two_hand_angle, two_elbow_angle,
                          two_knee_angle, two_feet_angle]  # 初始化角度特征列表

        # 计算所有角度特征
        for joint, (v1, v2) in dgait_bone_angle_pairs.items():

            # 计算所有相邻骨骼角度
            vec1 = x[:, :2, :, v1 - 1, :] - x[:, :2, :, joint - 1, :]  # 向量 1 (选择 x, y)
            vec2 = x[:, :2, :, v2 - 1, :] - x[:, :2, :, joint - 1, :]  # 向量 2 (选择 x, y)
            angle_feature = (1.0 - self.cos(vec1, vec2))  # 计算角度特征
            angle_feature[angle_feature != angle_feature] = 0  # 去除 NaN 值
            joint_list_bone_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))  # 保持通道维度

            # 计算左手角度
            vec1 = x[:, :2, :, 9 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 11 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            left_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算右手角度
            vec1 = x[:, :2, :, 8 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 10 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            right_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两手角度
            vec1 = x[:, :2, :, 11 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 10 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_hand_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两肘角度
            vec1 = x[:, :2, :, 9 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 8 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_elbow_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两膝角度
            vec1 = x[:, :2, :, 15 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 14 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_knee_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

            # 计算两脚角度
            vec1 = x[:, :2, :, 17 - 1, :] - x[:, :2, :, joint - 1, :]
            vec2 = x[:, :2, :, 16 - 1, :] - x[:, :2, :, joint - 1, :]
            angle_feature = (1.0 - self.cos(vec1, vec2))
            angle_feature[angle_feature != angle_feature] = 0
            two_feet_angle.append(angle_feature.unsqueeze(2).unsqueeze(1))

        # 将每个角度特征列表拼接到一起
        for angle_features_id in range(len(angle_features)):
            angle_features[angle_features_id] = torch.cat(angle_features[angle_features_id], dim=3)

        # 沿通道维度拼接所有角度特征
        angle_features = torch.cat(angle_features, dim=1)

        # 确保与 x 的设备一致
        angle_features = angle_features.to(x.device)

        # 将角度特征拼接到原始输入数据 (通道维度增加)
        x = torch.cat((x, angle_features), dim=1)  # 通道数变为9

        # # Joint + All_Bone_Angle
        # x = x[:, :3, :, :, :]  # 只保留角度特征 (N, 3, T, V, M)

        # Joint + Two_Knee_Angle
        x = torch.cat((x[:, :2, :, :, :], x[:, 7:8, :, :, :]), dim=1)  # 只保留角度特征 (N, 3, T, V, M)

        # 通道压缩
        x = x.permute(0, 4, 3, 2, 1).reshape(N * M * V, T, 3)  # (N*M*V, T, 3)
        x = x.permute(0, 2, 1)  # 重新排列维度，确保通道数在第二维，时间步数在第三维
        x = self.channel_compress(x)  # 压缩 (N*M*V, T, 2)
        x = x.permute(0, 2, 1).contiguous().view(N, M, V, 2, T).permute(0, 3, 4, 2, 1)  # 恢复形状 (N, 2, T, V, M)

        return x

    def forward(self, x):
        # 添加角度特征并压缩通道
        x = self.preprocessing(x)

        N, C, T, V, M = x.size()

        # 数据归一化
        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)  # N是批次数量，C是坐标数，T是帧数，V是骨架数，M是人数
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        # 网络层
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)  # 64
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)  # 128
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # 256

        # 特征池化
        c_new = x.size(1)

        # 全局池化
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)

        x = self.drop_out(x)

        return self.fc(x)
    


class Model_4part_ForFusion(nn.Module):
    def __init__(self, num_class=2, num_point=17, num_person=1, graph=None, graph_args=dict(), in_channels=2,
                 drop_out=0, adaptive=True, head=['ViT-L/14']):
        super(Model_4part_ForFusion, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(**graph_args)

        A = self.graph.A

        self.num_class = num_class
        self.num_point = num_point
        self.data_bn = nn.BatchNorm1d(num_person * in_channels * num_point)

        base_channel = 64
        self.l1 = TCN_GCN_unit(in_channels, base_channel, A, residual=False, adaptive=adaptive)
        self.l2 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)  # 2,3,6,9
        self.l3 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l4 = TCN_GCN_unit(base_channel, base_channel, A, adaptive=adaptive)
        self.l5 = TCN_GCN_unit(base_channel, base_channel * 2, A, stride=2, adaptive=adaptive)
        self.l6 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l7 = TCN_GCN_unit(base_channel * 2, base_channel * 2, A, adaptive=adaptive)
        self.l8 = TCN_GCN_unit(base_channel * 2, base_channel * 4, A, stride=2, adaptive=adaptive)
        self.l9 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)
        self.l10 = TCN_GCN_unit(base_channel * 4, base_channel * 4, A, adaptive=adaptive)

        self.linear_head = nn.ModuleDict()
        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))

        self.part_list = nn.ModuleList([nn.Linear(256, 768) for _ in range(4)])

        self.head = head

        for head in self.head:
            if head == 'ViT-L/14':
                self.linear_head[head] = nn.Linear(256, 768)
                conv_init(self.linear_head[head])

        self.fc = nn.Linear(base_channel * 4, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x):
        N, C, T, V, M = x.size()

        x = x.permute(0, 4, 3, 1, 2).contiguous().view(N, M * V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T).permute(0, 1, 3, 4, 2).contiguous().view(N * M, C, T, V)

        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x = self.l10(x)  # 输出形状: (N*M, 256, T', V')

        # 全局池化得到最终骨架特征
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)  # (N, 256)

        return x  # 只返回最终的骨架特征
