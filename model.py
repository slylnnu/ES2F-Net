import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class PAM_Module(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x, x1):
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x1).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class CAM_Module(Module):
    """ Channel attention module"""

    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)

    def forward(self, x, x2):
        m_batchsize, C, height, width = x.size()
        proj_query = x2.view(m_batchsize, C, -1)
        proj_key = x2.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)
        out = self.gamma * out + x
        return out


class GraphConvolutionLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # 权重矩阵 W
        self.bias = nn.Parameter(torch.randn(out_features))  # 偏置

    def forward(self, A, H):
        H_prime = torch.bmm(A, H.transpose(1, 2))  # (batch_size, nodes, features)
        H_prime = torch.matmul(H_prime, self.weight) + self.bias  # (batch_size, nodes, out_features)
        H_prime = F.relu(H_prime)

        return H_prime


class GraphConvolutionLayer2(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolutionLayer2, self).__init__()
        self.weight = nn.Parameter(torch.randn(in_features, out_features))  # 权重矩阵 W
        self.bias = nn.Parameter(torch.randn(out_features))  # 偏置

    def forward(self, A, H):
        H_prime = torch.bmm(A, H.transpose(1, 2))  # (batch_size, nodes, features)
        H_prime = torch.matmul(H_prime, self.weight) + self.bias  # (batch_size, nodes, out_features)
        H_prime = F.relu(H_prime)

        return H_prime


class SE(nn.Module):

    def __init__(self, in_channels, para_tune=False, inter_channels=None, mode='embedded', dimension=2, bn_layer=True):
        super(SE, self).__init__()
        assert dimension in [1, 2, 3]
        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
        self.mode = mode
        self.dimension = dimension
        self.para_tune = para_tune
        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):
        # x_thisBranch for g and theta
        # x_otherBranch for phi
        """
        args
            x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
        """
        # print(x_thisBranch.shape)

        batch_size = x_thisBranch.size(0)

        # (N, C, THW)
        # this reshaping and permutation is from the spacetime_nonlocal function in the original Caffe2 implementation
        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        # g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        # elif self.mode == "concatenate":
        else:  # default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            # a = 1
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])
        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch
        if self.para_tune:
            z = x_thisBranch
        return z


class SA(nn.Module):

    def __init__(self, in_channels, inter_channels=None, mode='embedded',
                 dimension=2, bn_layer=True):
        super(SA, self).__init__()

        assert dimension in [1, 2, 3]

        if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
            raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')

        self.mode = mode
        self.dimension = dimension

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        # the channel size is reduced to half inside the block
        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # assign appropriate convolutional, max pool, and batch norm layers for different dimensions
        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        # function g in the paper which goes through conv. with kernel size 1
        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        # add BatchNorm layer after the last conv layer
        if bn_layer:
            self.W_z = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
                bn(self.in_channels)
            )
            # from section 4.1 of the paper, initializing params of BN ensures that the initial state of non-local block is identity mapping
            nn.init.constant_(self.W_z[1].weight, 0)
            nn.init.constant_(self.W_z[1].bias, 0)
        else:
            self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

            # from section 3.3 of the paper by initializing Wz to 0, this block can be inserted to any existing architecture
            nn.init.constant_(self.W_z.weight, 0)
            nn.init.constant_(self.W_z.bias, 0)

        # define theta and phi for all operations except gaussian
        if self.mode == "embedded" or self.mode == "dot" or self.mode == "concatenate":
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

        if self.mode == "concatenate":
            self.W_f = nn.Sequential(
                nn.Conv2d(in_channels=self.inter_channels * 2, out_channels=1, kernel_size=1),
                nn.ReLU()
            )

    def forward(self, x_thisBranch, x_otherBranch):

        batch_size = x_thisBranch.size(0)

        g_x = self.g(x_thisBranch).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        if self.mode == "gaussian":
            theta_x = x_thisBranch.view(batch_size, self.in_channels, -1)
            phi_x = x_otherBranch.view(batch_size, self.in_channels, -1)
            theta_x = theta_x.permute(0, 2, 1)
            f = torch.matmul(theta_x, phi_x)

        elif self.mode == "embedded" or self.mode == "dot":
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, -1)
            # theta_x = theta_x.permute(0, 2, 1)
            phi_x = phi_x.permute(0, 2, 1)
            f = torch.matmul(phi_x, theta_x)

        else:  # default as concatenate
            theta_x = self.theta(x_thisBranch).view(batch_size, self.inter_channels, -1, 1)
            phi_x = self.phi(x_otherBranch).view(batch_size, self.inter_channels, 1, -1)

            h = theta_x.size(2)
            w = phi_x.size(3)
            theta_x = theta_x.repeat(1, 1, 1, w)
            phi_x = phi_x.repeat(1, 1, h, 1)

            concat = torch.cat([theta_x, phi_x], dim=1)
            f = self.W_f(concat)
            f = f.view(f.size(0), f.size(2), f.size(3))

        if self.mode == "gaussian" or self.mode == "embedded":
            f_div_C = F.softmax(f, dim=-1)
        elif self.mode == "dot" or self.mode == "concatenate":
            N = f.size(-1)  # number of position in x
            f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)

        # contiguous here just allocates contiguous chunk of memory
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x_thisBranch.size()[2:])

        W_y = self.W_z(y)
        # residual connection
        z = W_y + x_thisBranch

        return z


class Exchange(nn.Module):
    def __init__(self):
        super(Exchange, self).__init__()

    def forward(self, x, bn, bn_threshold):
        bn1, bn2 = bn[0].weight.abs(), bn[1].weight.abs()
        x1, x2 = torch.zeros_like(x[0]), torch.zeros_like(x[1])
        x1[:, bn1 >= bn_threshold] = x[0][:, bn1 >= bn_threshold]
        x1[:, bn1 < bn_threshold] = x[1][:, bn1 < bn_threshold]
        x2[:, bn2 >= bn_threshold] = x[1][:, bn2 >= bn_threshold]
        x2[:, bn2 < bn_threshold] = x[0][:, bn2 < bn_threshold]
        return [x1, x2]



class ESF2Net(nn.Module):
    def __init__(self, FM, NC, Classes):
        super(ESF2Net, self).__init__()
        self.bn_threshold1 = 0.95
        self.bn_threshold2 = 0.95
        self.conv1 = nn.Conv2d(
            in_channels=NC,
            out_channels=FM,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn1 = nn.BatchNorm2d(FM)
        self.conv2 = nn.Conv2d(
            in_channels=1,
            out_channels=FM,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn2 = nn.BatchNorm2d(FM)

        self.conv3 = nn.Conv2d(
                in_channels=FM,
                out_channels=FM * 2,
                kernel_size=3,
                stride=1,
                padding=1
            )
        self.conv7 = nn.Conv2d(
            in_channels=FM,
            out_channels=FM * 2,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.bn3 = nn.BatchNorm2d(FM * 2)
        self.bn4 = nn.BatchNorm2d(FM * 2)

        self.convhl = nn.Sequential(
            nn.Conv2d(FM * 4, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.convlh = nn.Sequential(
            nn.Conv2d(FM * 4, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(FM * 4, FM * 4, 3, 1, 1),
            nn.BatchNorm2d(FM * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),
        )

        self.conv_mask1 = nn.Conv2d(FM * 2, FM * 2, kernel_size=1)  # context Modeling
        self.conv_mask2 = nn.Conv2d(FM * 2, FM * 2, kernel_size=1)  # context Modeling
        self.conv_mask3 = nn.Conv2d(FM * 2, FM * 2, kernel_size=1)  # context Modeling
        self.conv_mask4 = nn.Conv2d(FM * 2, FM * 2, kernel_size=1)  # context Modeling
        self.gcn_layer = GraphConvolutionLayer(64, 64)
        self.gcn_layer2 = GraphConvolutionLayer2(64, 64)
        self.sa = SA(in_channels=FM * 2)
        self.se = SE(in_channels=FM * 2, para_tune=True)
        self.exchange = Exchange()
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(0.5)
        self.out1 = nn.Linear(FM * 4, Classes)
        self.out2 = nn.Linear(FM * 4, Classes)
        self.out3 = nn.Linear(FM * 4, Classes)

    def forward(self, x1, x2):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        xx1 = [x1, x2]
        bn1_list = [self.bn1, self.bn2]
        xx1 = self.exchange(xx1, bn1_list, self.bn_threshold1)
        x1 = xx1[0]
        x2 = xx1[1]
        x1 = self.relu(x1)
        x1_1 = self.pool(x1)
        x2 = self.relu(x2)
        x2_1 = self.pool(x2)

        x1_1 = self.conv3(x1_1)
        x1_1 = self.bn3(x1_1)
        x2_1 = self.conv3(x2_1)
        x2_1 = self.bn4(x2_1)
        xx2 = [x1_1, x2_1]
        bn2_list = [self.bn3, self.bn4]
        xx2 = self.exchange(xx2, bn2_list, self.bn_threshold2)
        x1_1 = xx2[0]
        x2_1 = xx2[1]
        x1_1 = self.relu(x1_1)
        x1_1 = self.pool(x1_1)
        x1 = self.dropout(x1_1)
        x2_1 = self.relu(x2_1)
        x2_1 = self.pool(x2_1)
        x2 = self.dropout(x2_1)


        B = x1.size(0)
        C = x1.size(1)
        H = x1.size(2)
        W = x1.size(3)
        query = self.conv_mask1(x1)
        key = self.conv_mask2(x1)
        query = query.reshape(B, C, H * W).permute(0, 2, 1)
        key = key.reshape(B, C, H * W)
        att_map = query @ key
        att_map =att_map.softmax(dim=-1)
        x1_reshape = x1.reshape(B, C, -1)
        output_gcn = self.gcn_layer(att_map, x1_reshape)
        output_gcn = output_gcn.view(B, C, H, W)
        out5 = torch.cat((x1, output_gcn), 1)
        x_hl = self.convhl(out5)


        B = x2.size(0)
        C = x2.size(1)
        H = x2.size(2)
        W = x2.size(3)
        query2 = self.conv_mask3(x2)
        key2 = self.conv_mask4(x2)
        query2 = query2.reshape(B, C, H * W).permute(0, 2, 1)
        key2 = key2.reshape(B, C, H * W)
        att_map2 = query2 @ key2
        att_map2 = att_map2.softmax(dim=-1)
        x2_reshape = x2.reshape(B, C, -1)
        output_gcn2 = self.gcn_layer2(att_map2, x2_reshape)
        output_gcn2 = output_gcn2.view(B, C, H, W)
        out4 = torch.cat((x2, output_gcn2), 1)
        x_lh = self.convlh(out4)

        x1 = self.sa(x1, x2)
        x2 = self.se(x2, x1)

        x_fusion = torch.cat((x1, x2), 1)
        x_fusion = self.conv5(x_fusion)

        x_hl = x_hl.view(x_hl.size(0), -1)
        out1 = self.out1(x_hl)

        x_lh = x_lh.view(x_lh.size(0), -1)
        out2 = self.out2(x_lh)

        x_fusion = x_fusion.view(x_fusion.size(0), -1)
        out3 = self.out3(x_fusion)

        return out1, out2, out3