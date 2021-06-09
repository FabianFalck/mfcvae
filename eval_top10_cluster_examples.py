import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import argparse

from datasets import Fast_MNIST, Fast_SVHN, Fast_3DShapes
from load_model import load_model_from_save_dict
from plotting import plot_cluster_examples_torch_grid


def eval_top10_cluster_examples():
    """
    Run this function to perform post-training cluster input example plot of a model.
    For more information on the plot, see Section 4.1 and Appendix E.3 of the paper.
    """
    parser = argparse.ArgumentParser(description='Evaluation parsing.')
    parser.add_argument('--model_path', type=str, default="pretrained_models/mnist.pt", metavar='N', help="Path to a model file of type .pt .")
    parser.add_argument('--results_dir', type=str, default="results/mnist", metavar='N', help="Path to a directory where results will be stored.")
    parser.add_argument('--device', type=str, default='cpu', metavar='N', help="device to use for all heavy tensor operations, e.g. 'cuda:0', 'cpu', ...")
    eval_args, unknown = parser.parse_known_args()

    # configs
    model_path = eval_args.model_path
    results_dir = eval_args.results_dir
    device_string = eval_args.device  # define device and load model

    mfcvae, args = load_model_from_save_dict(model_path, map_location=device_string)

    mfcvae.eval()  # changes model to evaluation mode (e.g. dropout, batch norm affected)

    # transfer model to device
    args.device = device_string
    device = torch.device(args.device)
    mfcvae.device = device
    mfcvae = mfcvae.to(device)


    # make deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'fast_mnist':
        print("Initialize MNIST data and data loaders...")
        # initialize dataset
        test_data = Fast_MNIST("./data", train=False, device=args.device)  # before: torchvision.datasets.MNIST
    elif args.dataset == 'fast_svhn':
        test_data = Fast_SVHN("./data", split='test', download=True, device=args.device)
        # move data and labels back to cpu for these testing purposes, otherwise quickly out of memory
        test_data.data = test_data.data.to('cpu')
        test_data.labels = test_data.labels.to('cpu')
    elif args.dataset == 'fast_3dshapes':
        test_data = Fast_3DShapes(train=False, device=args.device, train_frac = args.threedshapes_train_frac,
                                                         factors_variation_dict=args.factors_variation_dict,
                                                         factors_label_list=args.factors_label_list,
                                                         seed=args.seed)
        # move data and labels back to cpu for these testing purposes, otherwise quickly out of memory
        test_data.data = test_data.data.to('cpu')
        test_data.labels = test_data.labels.to('cpu')
    else:
        exit("Dataset not implemented.")

    # make dataset compatible with multi-label datasets
    # train_data.targets = train_data.targets.unsqueeze(1)
    # test_data.targets = test_data.targets.unsqueeze(1)

    # limit to subset - otherwise, memory explodes
    # test_data = torch.utils.data.Subset(test_data, range(1000))

    if args.do_progressive_training:
        epoch = int(sum(args.n_epochs_per_progressive_step)) - 1
    else:
        epoch = args.n_epochs - 1
    fig_list, row_indices, column_indices, num_nonempty_clusters = plot_top10_cluster_examples(test_data,
                                    mfcvae, args, epoch, results_dir, show_plot=True)


def plot_top10_cluster_examples(data, mfcvae, args, epoch, results_dir=None, show_plot=False):
    device = torch.device(args.device)
    """
    Args:
       data: The set of data where input examples are chosen from.
       mfcvae: The trained MFCVAE model.
       args: The arguments associated with the training procedure.
       epoch: The number of current epoch, which influences the value of alpha and gamma in progressive training.
              If the function is called post-training, then epoch can be set to be the last epoch.
       results_dir: Path to save the output plots.
       show_plot: Whether to show the plots by plt.show().
    """
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.eval_batch_size, shuffle=False,  #
                                              num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390

    if args.dataset == 'fast_mnist':
        in_channels = 1
        width, height = 28, 28
    elif args.dataset in ['fast_svhn', 'fast_3dshapes']:
        in_channels = 3
        width, height = 32, 32
    else:
        exit("dataset not known")

    vis_examples_per_cluster_logged = [{} for j in range(args.J_n_mixtures)]
    vis_count_examples_per_cluster = [{} for j in range(args.J_n_mixtures)]
    vis_z_j_per_cluster = [{} for j in range(args.J_n_mixtures)]
    index_to_y_j_cluster = {}
    index_to_prob_p_c_j_z_j = {}

    for j in range(args.J_n_mixtures):
        for k in range(mfcvae.n_clusters_j_list[j]):
            vis_examples_per_cluster_logged[j][k] = []
            vis_z_j_per_cluster[j][k] = []
            vis_count_examples_per_cluster[j][k] = 0
    for n in range(len(data)):
        index_to_y_j_cluster[n] = []

    for batch_idx, (x, y_true) in enumerate(data_loader):
        x, y_true = x.to(device), y_true.to(device)
        if args.dataset == 'fast_mnist':
            x = x.view(x.size(0), -1).float()
        global_indices = list(range(batch_idx*args.eval_batch_size, (batch_idx+1)*args.eval_batch_size))

        x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, epoch, batch_idx)
        prob_p_c_j_z_j_list = mfcvae.compute_q_c_j_x(z_sample_q_z_j_x_list)

        for h in range(z_sample_q_z_j_x_list[0].shape[0]):  # is probably == batch size
            g = global_indices[h]
            index_to_prob_p_c_j_z_j[g] = [prob_p_c_j_z_j_list[j][h].detach().cpu() for j in range(args.J_n_mixtures)]

        y_pred_j_list = []
        for j in range(mfcvae.J_n_mixtures):
            prob_p_c_j_z_j_list[j] = prob_p_c_j_z_j_list[j].data.cpu().numpy()
            y_pred_j = np.argmax(prob_p_c_j_z_j_list[j], axis=1)
            y_pred_j_list.append(y_pred_j)

        for j in range(mfcvae.J_n_mixtures):
            for k in range(mfcvae.n_clusters_j_list[j]):
                y_pred = y_pred_j_list[j]
                indices = (np.where(y_pred == k)[0])
                count_indices = indices.shape[0]
                indices = indices.tolist()

                for h in indices:
                    index_to_y_j_cluster[global_indices[h]].append(k)

                vis_count_examples_per_cluster[j][k] += count_indices

    # print("looped through data.")

    # build a useful data structure to handle the clustering probabilities
    j_to_cluster_to_index_prob = {}
    # create empty things
    for j in range(args.J_n_mixtures):
        j_to_cluster_to_index_prob[j] = {}
        for c in range(args.n_clusters_j_list[j]):
            j_to_cluster_to_index_prob[j][c] = []

    for (index, prob_list) in index_to_prob_p_c_j_z_j.items():
        for j in range(args.J_n_mixtures):
            cluster_j = torch.argmax(prob_list[j])
            cluster_j = cluster_j.item()
            j_to_cluster_to_index_prob[j][cluster_j].append((index, prob_list[j][cluster_j]))

    # Sort clusters s.t. cluster with the largest "average confidence" is 0, second largest 1 etc.
    row_indices = []
    cluster_average_confidence = {}
    num_nonempty_clusters = []
    for j in range(args.J_n_mixtures):
        cluster_average_confidence[j] = {}
        for c in range(args.n_clusters_j_list[j]):
            cluster_average_confidence[j][c] = np.nan_to_num(np.mean([j_to_cluster_to_index_prob[j][c][k][1] for k in range(len(j_to_cluster_to_index_prob[j][c]))]))
        # sort
        cluster_index_average_confidence_list = [(cluster_j, score) for (cluster_j, score) in cluster_average_confidence[j].items()]
        cluster_index_average_confidence_list = sorted(cluster_index_average_confidence_list, key=lambda tuple: tuple[1], reverse=True)
        # print(cluster_index_average_confidence_list)
        cluster_j_sorted = [cluster_j for (cluster_j, score) in cluster_index_average_confidence_list]
        # print('sorted index for facet', j, ': ', cluster_j_sorted)
        row_indices.append(cluster_j_sorted)
        num_nonempty_clusters.append(len(np.argwhere(np.array([cluster_index_average_confidence_list[i][1] for i in range(args.n_clusters_j_list[j])]))))
        fromto_mapping = {cluster_j: i for i, cluster_j in enumerate(cluster_j_sorted)}
        # remap the dictionary - https://gist.github.com/pszaflarski/b139736415abbf8d344d77524baaece8
        j_to_cluster_to_index_prob[j] = {fromto_mapping.get(k, k): v for k, v in j_to_cluster_to_index_prob[j].items() if k in fromto_mapping}
    # print(cluster_average_confidence)

    # 2) sort the list within each cluster by most probable to least probable
    column_indices = []
    for j in range(args.J_n_mixtures):
        column_indices.append([])
        for c in range(args.n_clusters_j_list[j]):
            j_to_cluster_to_index_prob[j][c] = sorted(j_to_cluster_to_index_prob[j][c], key=lambda tuple: tuple[1], reverse=True)
            column_indices[j] = j_to_cluster_to_index_prob[j][c]

    j_to_cluster_to_input = {}
    j_to_cluster_to_count = {}
    # create empty things
    for j in range(args.J_n_mixtures):
        j_to_cluster_to_input[j] = {}
        j_to_cluster_to_count[j] = {}
        for c in range(args.n_clusters_j_list[j]):
            j_to_cluster_to_input[j][c] = []

    # select only max of n_examples_per_cluster_to_show images per cluster
    n_examples_per_cluster_to_show_max = 10
    for j in range(args.J_n_mixtures):
        for c in range(args.n_clusters_j_list[j]):
            n_examples = min(len(j_to_cluster_to_index_prob[j][c]), n_examples_per_cluster_to_show_max)
            j_to_cluster_to_index_prob[j][c] = j_to_cluster_to_index_prob[j][c][:n_examples]
            j_to_cluster_to_count[j][c] = n_examples

    # actually make the output dict of cluster examples
    # populate
    for j in range(args.J_n_mixtures):
        for c in range(args.n_clusters_j_list[j]):
            for tup in j_to_cluster_to_index_prob[j][c]:
                # print(test_data[tup[0]][0])
                j_to_cluster_to_input[j][c].append(((data[tup[0]][0]).cpu().detach(), 'sthelse'))  # .permute(1, 2, 0)

    # print("Checkpoint 2.")

    # PLOTTING OUT CLUSTER EXAMPLES ------------
    fig_list = []
    for j in range(mfcvae.J_n_mixtures):
        fig = plot_cluster_examples_torch_grid(inputs_dict=j_to_cluster_to_input[j],
                                               count_dict=j_to_cluster_to_count[j],
                                               n_clusters=mfcvae.n_clusters_j_list[j],
                                               n_examples_per_cluster=n_examples_per_cluster_to_show_max)
        fig_list.append(fig)
        if results_dir is not None:
            plt.savefig(os.path.join(results_dir, 'examples_top_10_facet_%d_full.pdf'%(j)), format='pdf')  # , dpi=3000
        if show_plot:
            plt.show()
        plt.close()

    print("cluster examples done.")
    return fig_list, row_indices, column_indices, num_nonempty_clusters


if __name__ == '__main__':
    eval_top10_cluster_examples()
