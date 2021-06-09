
import torch
import torch.nn as nn
import torch.distributions as D
from torch.nn import functional as F
from torch.distributions.utils import logits_to_probs, probs_to_logits
import torch.nn.utils.weight_norm as wn
import math
import numpy as np
from sklearn.mixture import GaussianMixture

from utils import build_fc_network, softplus_inverse, softplus_inverse_numpy, build_cnn_network
from typing import List


class ResidualConvBlock(nn.Module):
    def __init__(self, n_channels: int, activation: str = 'relu'):
        super(ResidualConvBlock, self).__init__()

        net = []
        net.append(nn.Conv2d(n_channels, n_channels, 3, 1, 1, bias=True))  # Conv2D(in_channels, out_channels, kernel_size, stride=1, padding=0,...)
        if activation == 'relu':
            net.append(nn.ReLU())
        elif activation == 'leaky_relu':
            net.append(nn.LeakyReLU())
        elif activation == 'elu':
            net.append(nn.ELU())
        net.append(nn.BatchNorm2d(n_channels))
        net.append(nn.Conv2d(n_channels, n_channels, 1, bias=True))
        if activation == 'relu':
            net.append(nn.ReLU())
        elif activation == 'leaky_relu':
            net.append(nn.LeakyReLU())
        elif activation == 'elu':
            net.append(nn.ELU())
        net.append(nn.BatchNorm2d(n_channels))

        self.block = nn.Sequential(*net)

    def forward(self, x):
        return x + self.block(x)


class CONVvlaeEncoderCelebA(nn.Module):
    """
    See implementation in https://github.com/Zhiyuan1991/proVLAE/blob/master/model_ladder_pro_celbA.py for inspiration of architecture.
    """
    def __init__(self, in_dim: int, J_n_mixtures: int = 2, activation: str = "relu", add_resid_backbone: bool = False,
                 n_blocks_resid: int = 2, dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.in_dim = in_dim
        self.conv_backbone_0 = nn.ModuleList()  # "enc"
        self.conv_backbone_1 = nn.ModuleList()  # "enc"
        if J_n_mixtures == 3:
            self.conv_backbone_2 = nn.ModuleList()  # "enc"
        self.conv_rung_0 = nn.ModuleList()  # "qladder"
        self.conv_rung_1 = nn.ModuleList()  # "qladder"
        if J_n_mixtures == 3:
            self.conv_rung_2 = nn.ModuleList()  # "qladder"
        # self.mlp_rung = nn.ModuleList()  # "qladder"
        self.add_resid_backbone = add_resid_backbone
        self.n_blocks_resid = n_blocks_resid
        if add_resid_backbone:
            self.resid_conv_interm_0 = nn.ModuleList()
            self.resid_conv_interm_1 = nn.ModuleList()
            if J_n_mixtures == 3:
                self.resid_conv_interm_2 = nn.ModuleList()

        assert J_n_mixtures == 2 or J_n_mixtures == 3

        # construct network
        b_lower_dim = in_dim

        self.conv_backbone_out_sizes = [16, 8, 4, 2, 1]
        self.conv_rung_out_sizes = [8, 4, 2, 1, 1]

        if in_dim == 784:
            # MNIST/Fashion-MNIST
            in_channels = 1
        elif in_dim == 3072:
            # C10/SVHN:
            in_channels = 3

        self.conv_backbone_0.append(build_cnn_network(in_channels=in_channels,
                                                        out_channels=64,
                                                        transpose_conv=False,
                                                        kernel_size=4,
                                                        stride=2,
                                                        activation=activation))

        self.conv_backbone_1.append(build_cnn_network(in_channels=64,
                                                        out_channels=128,
                                                        transpose_conv=False,
                                                        kernel_size=4,
                                                        stride=2,
                                                        activation=activation))

        if J_n_mixtures == 3:
            self.conv_backbone_2.append(build_cnn_network(in_channels=128,
                                                        out_channels=256,
                                                        transpose_conv=False,
                                                        kernel_size=4,
                                                        stride=2,
                                                        activation=activation))


        self.conv_rung_0.append(build_cnn_network(in_channels=64,
                                                out_channels=64,
                                                transpose_conv=False,
                                                kernel_size=4,
                                                stride=2,
                                                activation=activation))

        self.conv_rung_0.append(build_cnn_network(in_channels=64,
                                                out_channels=64,
                                                transpose_conv=False,
                                                kernel_size=4,
                                                stride=1,
                                                activation=activation))


        self.conv_rung_1.append(build_cnn_network(in_channels=128,
                                                out_channels=128,
                                                transpose_conv=False,
                                                kernel_size=4,
                                                stride=2,
                                                activation=activation))

        self.conv_rung_1.append(build_cnn_network(in_channels=128,
                                                out_channels=256,
                                                transpose_conv=False,
                                                kernel_size=4,
                                                stride=2,
                                                activation=activation))


        if J_n_mixtures == 3:
            self.conv_rung_2.append(build_cnn_network(in_channels=256,
                                                out_channels=256,
                                                transpose_conv=False,
                                                kernel_size=4,
                                                stride=2,
                                                activation=activation))

            self.conv_rung_2.append(build_cnn_network(in_channels=256,
                                                    out_channels=512,
                                                    transpose_conv=False,
                                                    kernel_size=4,
                                                    stride=2,
                                                    activation=activation))

        self.encoder_output_dims = [3136, 1024, 512]

        if add_resid_backbone:
            for i in range(n_blocks_resid):
                self.resid_conv_interm_0.append(ResidualConvBlock(n_channels=64, activation=activation))
                self.resid_conv_interm_1.append(ResidualConvBlock(n_channels=128, activation=activation))
                if J_n_mixtures == 3:
                    self.resid_conv_interm_2.append(ResidualConvBlock(n_channels=256, activation=activation))


    def forward(self, x):
        # print("forward start ---")
        rung_list = []
        b = x
        # backbone 0
        b = self.conv_backbone_0[0](b)
        if self.add_resid_backbone:
            for i in range(self.n_blocks_resid):
                b = self.resid_conv_interm_0[i](b)
        if self.do_progressive_training:
            b_aux = b * self.alpha_enc_fade_in_list[0]
        else:
            b_aux = b
        # rung 0
        r = self.conv_rung_0[0](b_aux)
        r = self.conv_rung_0[1](r)
        r = r.view(r.shape[0], -1)
        rung_list.append(r)
        # backbone 1
        b = self.conv_backbone_1[0](b)
        if self.add_resid_backbone:
            for i in range(self.n_blocks_resid):
                b = self.resid_conv_interm_1[i](b)
        if self.do_progressive_training:
            b_aux = b * self.alpha_enc_fade_in_list[1]
        else:
            b_aux = b
        # rung 1
        r = self.conv_rung_1[0](b_aux)
        r = self.conv_rung_1[1](r)
        r = r.view(r.shape[0], -1)
        rung_list.append(r)
        if self.J_n_mixtures == 3:
            # backbone 2
            b = self.conv_backbone_2[0](b)
            if self.add_resid_backbone:
                for i in range(self.n_blocks_resid):
                    b = self.resid_conv_interm_2[i](b)
            if self.do_progressive_training:
                b_aux = b * self.alpha_enc_fade_in_list[2]
            else:
                b_aux = b
            # rung 1
            r = self.conv_rung_2[0](b_aux)
            r = self.conv_rung_2[1](r)
            r = r.view(r.shape[0], -1)
            rung_list.append(r)

        return rung_list



class CONVvlaeDecoderCelebA(nn.Module):
    """
    See implementation in https://github.com/Zhiyuan1991/proVLAE/blob/master/model_ladder_pro_celbA.py for inspiration of architecture.
    """
    def __init__(self, J_n_mixtures: int, in_dim: int, z_j_dim_list: List[int], activation: str = "relu",
                 add_resid_backbone: bool = False, n_blocks_resid: int = 2,
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.z_j_dim_list = z_j_dim_list
        self.conv_backbone_0 = nn.ModuleList()  # "dec"
        self.conv_backbone_1 = nn.ModuleList()  # "dec"
        if J_n_mixtures == 3:
            self.conv_backbone_2 = nn.ModuleList()  # "dec"
        self.mlp_rung_0 = nn.ModuleList()  # "pladder"
        self.mlp_rung_1 = nn.ModuleList()  # "pladder"
        if J_n_mixtures == 3:
            self.mlp_rung_2 = nn.ModuleList()  # "pladder"
        self.add_resid_backbone = add_resid_backbone
        self.n_blocks_resid = n_blocks_resid
        if add_resid_backbone:
            self.resid_conv_interm_0 = nn.ModuleList()
            self.resid_conv_interm_1 = nn.ModuleList()
            if J_n_mixtures == 3:
                self.resid_conv_interm_2 = nn.ModuleList()

        self.conv_backbone_out_sizes = [16, 8, 4, 2, 1]
        self.conv_rung_out_sizes = [8, 4, 2, 1, 1]

        if in_dim == 784:
            # MNIST/Fashion-MNIST
            out_channels = 1
        elif in_dim == 3072:
            # C10/SVHN:
            out_channels = 3

        # yes, this large!
        if J_n_mixtures == 3:
            self.mlp_rung_2.append(build_fc_network(layer_dims=[z_j_dim_list[2], 4 * 4 * 512], activation=activation, batch_norm=do_fc_batch_norm))

            self.conv_backbone_2.append(build_cnn_network(in_channels=512,
                                                        out_channels=512,  # different to below
                                                        transpose_conv=True,
                                                        kernel_size=4,
                                                        stride=2,
                                                        activation=activation))

            self.conv_backbone_2.append(build_cnn_network(in_channels=512,
                                                        out_channels=256,
                                                        transpose_conv=True,
                                                        kernel_size=3,  # changed kernel size to ensure dimensions match
                                                        stride=1,
                                                        activation=activation))

        # yes, this large!
        self.mlp_rung_1.append(build_fc_network(layer_dims=[z_j_dim_list[1], 8 * 8 * 256], activation=activation, batch_norm=do_fc_batch_norm))

        self.conv_backbone_1.append(build_cnn_network(in_channels=256 if self.J_n_mixtures == 2 else 512,
                                                    out_channels=128,
                                                    transpose_conv=True,
                                                    kernel_size=4,
                                                    stride=2,
                                                    activation=activation))

        self.conv_backbone_1.append(build_cnn_network(in_channels=128,
                                                    out_channels=64,
                                                    transpose_conv=True,
                                                    kernel_size=3,  # changed kernel size to ensure dimensions match
                                                    stride=1,
                                                    activation=activation))

        self.mlp_rung_0.append(build_fc_network(layer_dims=[z_j_dim_list[0], 16 * 16 * 64], activation=activation, batch_norm=do_fc_batch_norm))

        # option 1: last layer with regular building block
        self.conv_backbone_0.append(build_cnn_network(in_channels=128,  # since merged
                                                    out_channels=out_channels,
                                                    transpose_conv=True,
                                                    kernel_size=4,
                                                    stride=2,
                                                    activation=activation))

        # option 2: last layer unbounded with plain transpose convolution
        # is without weight norm
        # self.conv_backbone_0.append(nn.ConvTranspose2d(in_channels=128, out_channels=out_channels, kernel_size=4, stride=2, padding=1))

        # DO NOT COMMENT IN !!!!!!!!!!!!!!!!
        # self.conv_backbone_0.append(build_cnn_network(in_channels=64,
        #                                             out_channels=out_channels,
        #                                             transpose_conv=True,
        #                                             kernel_size=4,
        #                                             stride=2,
        #                                             activation=activation))
        # DO NOT COMMENT IN !!!!!!!!!!!!!!!!

        if add_resid_backbone:
            for i in range(n_blocks_resid):
                self.resid_conv_interm_0.append(ResidualConvBlock(n_channels=128, activation=activation))
                self.resid_conv_interm_1.append(ResidualConvBlock(n_channels=256 if self.J_n_mixtures == 2 else 512, activation=activation))
                if J_n_mixtures == 3:
                    self.resid_conv_interm_2.append(ResidualConvBlock(n_channels=512, activation=activation))


    def merge(self, r, upper_b, merge_type='cat', const=0.1):
        if merge_type == 'gated_add':
            m = const * r + upper_b
        elif merge_type == 'cat':
            m = torch.cat((r, upper_b), dim=1)
        return m


    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        # j = 2
        if self.J_n_mixtures == 3:  # TODO capital J and lower-case j are inconsistent (one-off) -> are they?
            r = self.mlp_rung_2[0](z_sample_q_z_j_x_list[2]).view(-1, 512, 4, 4)
            if self.do_progressive_training:
                r_aux = r * self.alpha_dec_fade_in_list[2]
            else:
                r_aux = r
            b = r_aux
            if self.add_resid_backbone:
                for i in range(self.n_blocks_resid):
                    b = self.resid_conv_interm_2[i](b)
            b = self.conv_backbone_2[0](b)
            b = self.conv_backbone_2[1](b)
        # j = 1
        r = self.mlp_rung_1[0](z_sample_q_z_j_x_list[1]).view(-1, 256, 8, 8)
        if self.do_progressive_training:
            r_aux = r * self.alpha_dec_fade_in_list[1]
        else:
            r_aux = r
        if self.J_n_mixtures == 2:
            b = r_aux
        else:
            b = self.merge(r_aux, b)
        if self.add_resid_backbone:
            for i in range(self.n_blocks_resid):
                b = self.resid_conv_interm_1[i](b)
        b = self.conv_backbone_1[0](b)
        b = self.conv_backbone_1[1](b)
        # j = 0
        r = self.mlp_rung_0[0](z_sample_q_z_j_x_list[0]).view(-1, 64, 16, 16)
        if self.do_progressive_training:
            r_aux = r * self.alpha_dec_fade_in_list[0]
        else:
            r_aux = r
        b = self.merge(r_aux, b)
        if self.add_resid_backbone:
            for i in range(self.n_blocks_resid):
                b = self.resid_conv_interm_0[i](b)
        b = self.conv_backbone_0[0](b)
        # b = self.conv_backbone_0[1](b)

        return b