from scipy.optimize import linear_sum_assignment as linear_assignment
import numpy as np
# import collections
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as wn

import yaml
import argparse


def cluster_acc_old(y_true, y_pred):
    """
    Compute clustering accuracy via the Kuhn-Munkres algorithm, also called the Hungarian matching algorithm.
    This algorithm provides a 1 to 1 matching between VaDE clusters and ground truth classes.
    Therefore, it is only valid when n_clusters is equal to the number of ground truth classes.

    y_pred and y_true contain integers, each indicating the cluster number a sample belongs to.
    y_pred therefore induces the predicted partition of all samples.
    However, the integers in y_pred are arbitrary and do not have to match the integers chosen for the true partition.
    We align the integers of y_true and y_pred through the Kuhn-Munkres algorithm and subsequently compute
    accuracy as usual.

    Code as modified from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch.

    Args:
        y_true: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.
        y_pred: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.

    Returns:
        A scalar indicating the clustering accuracy.
    """
    assert y_pred.size == y_true.size  # Arguments y_true and y_pred must be of equal shape.
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    # perform the Kuhn-Munkres algorithm to obtain the pairs
    ind = linear_assignment(w.max() - w)  # ind is a list of 2 numpy arrays where ind[0][k] and ind[1][k] form a pair for k=0,...,n_clusters-1
    # add the corresponding
    acc_and_w = sum([w[i, j] for (i, j) in zip(ind[0].tolist(), ind[1].tolist())]) * 1.0 / y_pred.size, w
    return acc_and_w


def cluster_acc_and_conf_mat(y_true, y_pred, conf_mat_option="absolute"):
    """
    Compute clustering accuracy.
    In this version, each cluster is assigned to the class with the largest number of observations in the cluster.
    Different from the cluster_acc_old function, this function allows multiple clusters to the same class.
    Therefore, n_cluster can be larger than the number of ground truth classes.

    As a by-product, the square confusion matrix is also computed.

    Args:
        y_true: 1-D numpy array containing integers between 0 and n_clusters-1, where n_clusters indicates the number of clusters.
        y_pred: 1-D numpy array containing integers between 0 and N-1, where N indicates the number of ground truth classes.

    Code as modified from https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch.

    Returns:
        A scalar indicating the clustering accuracy.
    """
    assert y_pred.size == y_true.size  # Arguments y_true and y_pred must be of equal shape.
    D_pred = y_pred.max() + 1
    D_true = y_true.max() + 1
    w = np.zeros((D_pred, D_true), dtype=np.int64)
    conf_mat = np.zeros((D_true, D_true), dtype=np.int64)
    # w[i, j] is the count of data points that lie in both VaDE cluster i and true class j
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind_pred = np.arange(D_pred)
    ind_true = np.zeros(D_pred, dtype=np.int64)
    # for each VaDE cluster, find the class with the largest number of observations in the cluster and record in ind_true
    for i in range(D_pred):
        ind_max = np.argmax(w[i, :])
        ind_true[i] = ind_max
        # add the count into the corresponding row of the confusion matrix
        conf_mat[ind_max, :] += w[i, :]
    ind = (ind_pred, ind_true)
    acc = sum([w[i, j] for (i, j) in zip(ind[0].tolist(), ind[1].tolist())]) * 1.0 / y_pred.size
    return acc, conf_mat, w


def cluster_acc_weighted(conf_mat):
    """
    Compute the weighted clustering accuracy.
    For each label, we compute the accuracy as the proportion of correctly predicted images within all images with the label.
    We then average the accuracies computed for each label, so that each label contributes to the same amount in the weighted accuracy.

    Args:
        conf_mat: confusion matrix

    Returns:
        A scalar indicating the weighted clustering accuracy.
    """
    label_counts = np.sum(conf_mat, axis=0)
    acc_for_each_label = np.diagonal(conf_mat) / label_counts
    acc_weighted = np.sum(acc_for_each_label) / len(label_counts)
    return acc_weighted


def build_fc_network(layer_dims, activation="relu", dropout_prob=0., batch_norm=False):
    """
    Stacks multiple fully-connected layers with an activation function and a dropout layer in between.

    - Source used as orientation: https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py

    Args: 
        layer_dims: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the input
                    and output dimension of the ith layer, respectively.
        activation: Activation function to choose. "relu" or "sigmoid".
        dropout_prob: Dropout probability between every fully connected layer with activation.

    Returns: 
        An nn.Sequential object of the layers.
    """
    # Note: possible alternative: OrderedDictionary
    net = []
    for i in range(1, len(layer_dims)):
        net.append(nn.Linear(layer_dims[i-1], layer_dims[i]))
        if activation == "relu":
            net.append(nn.ReLU())
        elif activation == 'leaky_relu':
            net.append(nn.LeakyReLU())
        elif activation == 'elu':
            net.append(nn.ELU())
        elif activation == "sigmoid":
            net.append(nn.Sigmoid())

        if batch_norm:
            net.append(nn.BatchNorm1d(layer_dims[i]))

        net.append(nn.Dropout(dropout_prob))
    net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

    return net


def build_cnn_network(in_channels, out_channels, transpose_conv, kernel_size, stride, output_padding=None,
                      activation='leaky_relu', dropout_prob=0., weight_norm=False, batch_norm=True):
    """
    Stuck multiple 2D-convolution layers with an action.

    Args:
        channels_list: A list of integers, where (starting from 1) the (i-1)th and ith entry indicates the number of input and output channels
                        of the ith (transposed) convolution operation, respectively.
        transpose_conv: Whether to use Conv2d or ConvTranspose2d
        kernel_size_list: list of kernel sizes of (transposed) convolutional layers.
        stride_list: list of strides of (transposed) convlutional layers.
        output_padding_list: List of integers indicating output padding after each conv layer.
        activation: Activation function to choose. "relu" currently implemented.
        dropout_prob: Dropout probability between every fully connected layer with activation.

    Returns:
        A sequential module of stacked convolution or transposed convolution layers.
    """
    net = []

    if not transpose_conv:
        if weight_norm:
            net.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1))
        else:
            net.append(wn(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)))
    else:
        if weight_norm:
            net.append(wn(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1)))
        else:
            net.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=1))
    if activation == 'relu':
        net.append(nn.ReLU())
    elif activation == 'leaky_relu':
        net.append(nn.LeakyReLU())
    elif activation == 'elu':
        net.append(nn.ELU())

    if batch_norm:
        net.append(nn.BatchNorm2d(out_channels))

    if dropout_prob > 0.:
        net.append(nn.Dropout(dropout_prob))
    net = nn.Sequential(*net)  # unpacks list as separate arguments to be passed to function

    return net


def str2bool(v):
    """
    Source code copied from: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def softplus_inverse(x, beta):
    """
    Inverse of torch softplus function https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html.
    """
    return (1/beta) * torch.log(torch.exp(beta*x) - 1.)


def softplus_inverse_numpy(x, beta):
    """
    Inverse of torch softplus function https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html ,
    but with numpy
    """
    return (1 / beta) * np.log(np.exp(beta * x) - 1.)









# below here: copied from https://github.com/addtt/boiler-pytorch/blob/master/boilr/nn/init.py
# -------------------------------------------------------

from typing import Optional

import torch
from torch import nn

# from boilr.nn.utils import is_conv, is_linear
def is_conv(module: nn.Module) -> bool:
    """Returns whether the module is a convolutional layer."""
    return isinstance(module, torch.nn.modules.conv._ConvNd)


def is_linear(module: nn.Module) -> bool:
    """Returns whether the module is a linear layer."""
    return isinstance(module, torch.nn.Linear)


from boilr.utils import to_np

debug = False


def _get_data_dep_hook(init_scale):
    """Creates forward hook for data-dependent initialization.
    The hook computes output statistics of the layer, corrects weights and
    bias, and corrects the output accordingly in-place, so the forward pass
    can continue.
    Args:
        init_scale (float): Desired scale (standard deviation) of each
            layer's output at initialization.
    Returns:
        Forward hook for data-dependent initialization
    """

    def hook(module, inp, out):
        inp = inp[0]

        out_size = out.size()

        if is_conv(module):
            separation_dim = 1
        elif is_linear(module):
            separation_dim = -1
        dims = tuple([i for i in range(out.dim()) if i != separation_dim])
        mean = out.mean(dims, keepdim=True)
        var = out.var(dims, keepdim=True)

        if debug:
            print("Shapes:\n   input:  {}\n   output: {}\n   weight: {}".format(
                inp.size(), out_size, module.weight.size()))
            print("Dims to compute stats over:", dims)
            print("Input statistics:\n   mean: {}\n   var: {}".format(
                to_np(inp.mean(dims)), to_np(inp.var(dims))))
            print("Output statistics:\n   mean: {}\n   var: {}".format(
                to_np(out.mean(dims)), to_np(out.var(dims))))
            print("Weight statistics:   mean: {}   var: {}".format(
                to_np(module.weight.mean()), to_np(module.weight.var())))

        # Given channel y[i] we want to get
        #   y'[i] = (y[i]-mu[i]) * is/s[i]
        #         = (b[i]-mu[i]) * is/s[i] + sum_k (w[i, k] * is / s[i] * x[k])
        # where * is 2D convolution, k denotes input channels, mu[i] is the
        # sample mean of channel i, s[i] the sample variance, b[i] the current
        # bias, 'is' the initial scale, and w[i, k] the weight kernel for input
        # k and output i.
        # Therefore the correct bias and weights are:
        #   b'[i] = is * (b[i] - mu[i]) / s[i]
        #   w'[i, k] = w[i, k] * is / s[i]
        # And finally we can modify in place the output to get y'.

        scale = torch.sqrt(var + 1e-5)

        # Fix bias
        module.bias.data = ((module.bias.data - mean.flatten()) * init_scale /
                            scale.flatten())

        # Get correct dimension if transposed conv
        transp_conv = getattr(module, 'transposed', False)
        ch_out_dim = 1 if transp_conv else 0

        # Fix weight
        size = tuple(-1 if i == ch_out_dim else 1 for i in range(out.dim()))
        weight_size = module.weight.size()
        module.weight.data *= init_scale / scale.view(size)
        assert module.weight.size() == weight_size

        # Fix output in-place so we can continue forward pass
        out.data -= mean
        out.data *= init_scale / scale

        assert out.size() == out_size

    return hook


def data_dependent_init(model: nn.Module,
                        model_input_dict: dict,
                        init_scale: Optional[float] = .1) -> None:
    """Performs data-dependent initialization on a model.
    Updates each layer's weights such that its outputs, computed on a batch
    of actual data, have mean 0 and the same standard deviation. See the code
    for more details.
    Args:
        model (torch.nn.Module):
        model_input_dict (dict): Dictionary of inputs to the model.
        init_scale (float, optional): Desired scale (standard deviation) of
            each layer's output at initialization. Default: 0.1.
    """

    hook_handles = []
    modules = filter(lambda m: is_conv(m) or is_linear(m), model.modules())
    for module in modules:
        # Init module parameters before forward pass
        nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.zero_()

        # Forward hook: data-dependent initialization
        hook_handle = module.register_forward_hook(
            _get_data_dep_hook(init_scale))
        hook_handles.append(hook_handle)

    # Forward pass one minibatch
    model.forward_new(**model_input_dict)  # dry-run   ; before: without ".forward_new"

    # Remove forward hooks
    for hook_handle in hook_handles:
        hook_handle.remove()


def load_args_from_yaml(file_path):
    """
    Load args from .yml file.

    Args:
        file_path:

    Returns:

    """
    with open(file_path) as file:
        config = yaml.safe_load(file)

    # create argsparse object
    parser = argparse.ArgumentParser(description='MFCVAE training')
    # parser.add_argument('--dummy', type=int, default=-1, metavar='N', help='placeholder')
    args, unknown = parser.parse_known_args()
    for key, value in config.items():
        setattr(args, key, value)

    # print(args)
    return args




#### Test #####

if __name__ == '__main__':
    debug = True

    # Test simple data-dependent init


    def do_test(x, layer):
        layer.bias.data.zero_()
        print("Output stats before:",
              layer(x).mean().item(),
              layer(x).var().item())
        handle = layer.register_forward_hook(_get_data_dep_hook(init_scale=0.5))
        y = layer(x)
        print("Output stats after:", y.mean().item(), y.var().item())
        handle.remove()

    # shape 64, 3, 5, 5
    x__ = (torch.rand(64, 3, 5, 5) - 0.2) * 20

    # Test Conv2d
    print("\n\n *** TEST Conv2d\n")
    do_test(x__, nn.Conv2d(3, 4, 3, padding=1))

    # Test ConvTranspose2d
    print("\n\n *** TEST ConvTranspose2d\n")
    do_test(x__, nn.ConvTranspose2d(3, 4, 3, padding=1))

    # Test Linear
    print("\n\n *** TEST Linear\n")
    x__ = x__.view(64, 25 * 3)  # flatten
    do_test(x__, nn.Linear(25 * 3, 8))


class Namespace_helper:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def merge_two_dicts(x, y):
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z