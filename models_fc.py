
import torch
import torch.nn as nn
from utils import build_fc_network
from typing import List


class FCsharedEncoder(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net = build_fc_network(layer_dims=layer_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm)

    def forward(self, x):
        h = self.net(x)
        h_list = [h for _ in range(self.J_n_mixtures)]

        return h_list


class FCSharedDecoder(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net = build_fc_network(layer_dims=layer_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm)

    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        z_sample_q_z_x = torch.cat(z_sample_q_z_j_x_list, dim=1)

        return self.net(z_sample_q_z_x)


class FCseparateEncoders(nn.Module):
    def __init__(self, layer_dims: List[int], J_n_mixtures: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = J_n_mixtures
        self.net_list = nn.ModuleList()
        for j in range(self.J_n_mixtures):
            self.net_list.append(build_fc_network(layer_dims=layer_dims[j], activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))


    def forward(self, x):
        h_list = []
        for j in range(self.J_n_mixtures):
            h = self.net_list[j](x)
            h_list.append(h)

        return h_list


class FCvlaeEncoder(nn.Module):
    def __init__(self, layer_dims: List[int], in_dim: int, activation: str = "relu",
                 dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = len(layer_dims)
        self.fc_backbone = nn.ModuleList()  # "enc"
        self.fc_rung = nn.ModuleList()  # "qladder" / "Sprosse"
        self.encoder_output_dims = []

        # construct network
        b_lower_dim = in_dim
        for j in range(self.J_n_mixtures):
            branch_index = layer_dims[j].index("branch")  # find branch
            # print(b_lower_dim)
            backbone_dims = [b_lower_dim] + layer_dims[j][:branch_index]
            # update b_lower_dim here already
            b_lower_dim = layer_dims[j][branch_index - 1] if branch_index-1 >= 0 else b_lower_dim  # lower branch index; used in upper layer
            rung_dims = [b_lower_dim] + layer_dims[j][branch_index + 1:]

            # print(backbone_dims)
            # print(rung_dims)

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(nn.Identity())
            else:
                self.fc_backbone.append(build_fc_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(nn.Identity())
            else:
                self.fc_rung.append(build_fc_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            self.encoder_output_dims.append(rung_dims[-1])



    def forward(self, x):
        # print("forward start ---")
        rung_list = []
        b = x
        for j in range(self.J_n_mixtures):
            # print(b.size())
            b = self.fc_backbone[j](b)
            if self.do_progressive_training:
                b_aux = b * self.alpha_enc_fade_in_list[j]
            else:
                b_aux = b
            r = self.fc_rung[j](b_aux)
            rung_list.append(r)

        return rung_list




class FCvlaeDecoder(nn.Module):
    def __init__(self, layer_dims: List[int], z_j_dim_list: List[int], merge_type: str = 'gated_add',
                 activation: str = "relu", dropout_prob: float = 0., do_fc_batch_norm: bool = False):
        super(self.__class__, self).__init__()
        self.J_n_mixtures = len(layer_dims)
        self.z_dim_list = z_j_dim_list
        self.merge_type = merge_type
        self.fc_backbone = nn.ModuleList()  # "dec"
        self.fc_rung = nn.ModuleList()  # "pladder" / "Sprosse"

        # construct network
        for j in range(self.J_n_mixtures):
            if j == self.J_n_mixtures - 1:
                # edge case: no 'merge' here
                # whether it's rung or backbone is arbitrary here
                rung_dims = []
                backbone_dims = [self.z_dim_list[j]] + layer_dims[j]
            else:
                merge_index = layer_dims[j].index("merge")  # find branch
                # print(merge_index)
                # note the reversed order!
                rung_dims = [self.z_dim_list[j]] + layer_dims[j][:merge_index]
                if self.merge_type == 'gated_add':
                    backbone_dims = [layer_dims[j][merge_index - 1]] + layer_dims[j][merge_index + 1:]
                elif self.merge_type == 'cat':
                    backbone_dims = [layer_dims[j][merge_index - 1] + layer_dims[j + 1][-1]] + layer_dims[j][merge_index + 1:]
            # print(backbone_dims)
            # print(rung_dims)
            if len(rung_dims) == 1:  # only input dimension
                self.fc_rung.append(nn.Identity())
            else:
                self.fc_rung.append(build_fc_network(layer_dims=rung_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

            if len(backbone_dims) == 1:  # only input dimension
                self.fc_backbone.append(nn.Identity())
            else:
                self.fc_backbone.append(build_fc_network(layer_dims=backbone_dims, activation=activation, dropout_prob=dropout_prob, batch_norm=do_fc_batch_norm))

    def merge(self, r, upper_b, merge_type='gated_add', const=0.1):
        if merge_type == 'gated_add':
            m = const * r + upper_b
        elif merge_type == 'cat':
            m = torch.cat((r, upper_b), dim=1)

        return m


    def forward(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        b = z_sample_q_z_j_x_list[self.J_n_mixtures - 1]
        b = self.fc_backbone[self.J_n_mixtures - 1](b)  # rung is empty here
        for j in reversed(range(self.J_n_mixtures - 1)):  # last one already processed
            r = self.fc_rung[j](z_sample_q_z_j_x_list[j])
            if self.do_progressive_training:
                r_aux = r * self.alpha_dec_fade_in_list[j]
            else:
                r_aux = r
            b = self.merge(r_aux, b, merge_type=self.merge_type)
            b = self.fc_backbone[j](b)

        return b



if __name__ == "__main__":

    enc = FCvlaeEncoder(layer_dims=[[100, 101, 'branch', 102, 103], [104, 105, 'branch'], ['branch'], ['branch', 106, 107]], in_dim = 50)
    x = torch.randn((10, 50))
    h_list = enc(x)

    dec = FCvlaeDecoder(layer_dims=[[100, 107, 'merge', 102, 103], [104, 107, 'merge'], [107, 'merge'], [109, 'merge', 106, 107], [108, 109]], z_j_dim_list= [3, 4, 5, 6, 7], merge_type='gated_add')
    z_sample_list = [torch.randn((10, 3)), torch.randn((10, 4)), torch.randn((10, 5)), torch.randn((10, 6)), torch.randn((10, 7))]
    h = dec(z_sample_list)
