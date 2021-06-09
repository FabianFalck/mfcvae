"""
Run this function to perform post-training compositionality plots of a model.
For more information on the plots, see Section 4.2 and Appendix E.4 of the paper.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import os
import argparse

from datasets import Fast_MNIST, Fast_SVHN, Fast_3DShapes

from load_model import load_model_from_save_dict
from plotting import plot_cluster_examples_torch_grid


parser = argparse.ArgumentParser(description='Evaluation parsing.')
parser.add_argument('--model_path', type=str, default="pretrained_models/3dshapes_1.pt", metavar='N', help="Path to a model file of type .pt .")
parser.add_argument('--results_dir', type=str, default="results/3dshapes_1", metavar='N', help="Path to a directory where results will be stored.")
parser.add_argument('--device', type=str, default='cpu', metavar='N', help="device to use for all heavy tensor operations, e.g. 'cuda:0', 'cpu', ...")
parser.add_argument('--swapped_facet', type=int, default=1, metavar='N', help='Facet number (j) of the facet that is swapped between two clusters.')
parser.add_argument('--first_n_clusters_swapped_1', type=int, default=5, metavar='N', help='Number of clusters swapped (starting from lowest index) in facet 1 (max. is number of clusters; scales quadratically with computational time to produce plots).')
parser.add_argument('--first_n_clusters_swapped_2', type=int, default=5, metavar='N', help='Number of clusters swapped (starting from lowest index) in facet 2 (max. is number of clusters; scales quadratically with computational time to produce plots)')
eval_args, unknown = parser.parse_known_args()


# Note: CONTROLS!!!!
# y_1_hat_chosen = 15  # cluster in facet 1 chosen

# Controls if done automatically (START) ---
model_path = eval_args.model_path
results_dir = eval_args.results_dir
swapped_facet = eval_args.swapped_facet  # must be 1 in SVHN and 0 in MNIST (for style)
device_string = eval_args.device
first_n_clusters_swapped_1 = eval_args.first_n_clusters_swapped_1
first_n_clusters_swapped_2 = eval_args.first_n_clusters_swapped_2
# Controls if done automatically (END) ---


# define device and load model
mfcvae, run_args = load_model_from_save_dict(model_path, map_location=device_string)

mfcvae.eval()  # changes model to evaluation mode (e.g. dropout, batch norm affected)

# transfer model to device
run_args.device = device_string
device = torch.device(device_string)
mfcvae.device = device
mfcvae = mfcvae.to(device)

if run_args.dataset == 'fast_mnist':
    print("Initialize MNIST data and data loaders...")
    # initialize dataset
    train_data = Fast_MNIST('./data', train=True, download=True,
                            device=run_args.device)  # before: torchvision.datasets.MNIST
    test_data = Fast_MNIST("./data", train=False, device=run_args.device)  # before: torchvision.datasets.MNIST
    # move data and labels back to cpu for these testing purposes, otherwise quickly out of memory
    test_data.data = test_data.data.to('cpu')
    test_data.targets = test_data.targets.to('cpu')
elif run_args.dataset == 'fast_svhn':
    # train_data = Fast_SVHN('./data', split='train', download=True, device=args.device)
    test_data = Fast_SVHN("./data", split='test', download=True, device=run_args.device)
    # move data and labels back to cpu for these testing purposes, otherwise quickly out of memory
    test_data.data = test_data.data.to('cpu')
    test_data.labels = test_data.labels.to('cpu')
elif run_args.dataset == 'fast_3dshapes':
    test_data = Fast_3DShapes(train=False, device=run_args.device, train_frac = run_args.threedshapes_train_frac,
                              factors_variation_dict=run_args.factors_variation_dict,
                              factors_label_list=run_args.factors_label_list,
                              seed=run_args.seed)
    # move data and labels back to cpu for these testing purposes, otherwise quickly out of memory
    test_data.data = test_data.data.to('cpu')
    test_data.labels = test_data.labels.to('cpu')
else:
    exit("Dataset not implemented.")


# Note: limit to subset - otherwise, memory explodes
# test_data = torch.utils.data.Subset(test_data, range(1000))

# initialize data loaders
test_loader = torch.utils.data.DataLoader(test_data, batch_size=run_args.eval_batch_size, shuffle=False,  #
                                          num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390

# print('Checkpoint 1.')

#  # plotting
def plot_image(img):
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if run_args.dataset == 'fast_mnist':
        ax.imshow(img, cmap='gray')
    elif run_args.dataset in ['fast_svhn', 'fast_3dshapes']:
        ax.imshow(img)
    else:
        exit("dataset not known")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

if run_args.dataset == 'fast_mnist':
    in_channels = 1
    width, height = 28, 28
elif run_args.dataset in ['fast_svhn', 'fast_3dshapes']:
    in_channels = 3
    width, height = 32, 32
else:
    exit("dataset not known")

y_true_list = []

vis_examples_per_cluster_logged = [{} for j in range(run_args.J_n_mixtures)]
vis_count_examples_per_cluster = [{} for j in range(run_args.J_n_mixtures)]
vis_z_j_per_cluster = [{} for j in range(run_args.J_n_mixtures)]
index_to_y_j_cluster = {}
index_to_z_j_embed = {}
index_to_x_hat = {}
index_to_prob_p_c_j_z_j = {}

for j in range(run_args.J_n_mixtures):
    for k in range(mfcvae.n_clusters_j_list[j]):
        vis_examples_per_cluster_logged[j][k] = []
        vis_z_j_per_cluster[j][k] = []
        vis_count_examples_per_cluster[j][k] = 0
for n in range(len(test_data)):
    index_to_y_j_cluster[n] = []



for batch_idx, (x, y_true) in enumerate(test_loader):
    x, y_true = x.to(device), y_true.to(device)
    if run_args.dataset == 'fast_mnist':
        x = x.view(x.size(0), -1).float()
    global_indices = list(range(batch_idx * run_args.eval_batch_size, (batch_idx + 1) * run_args.eval_batch_size))

    x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, int(sum(run_args.n_epochs_per_progressive_step)) - 1, 0)
    prob_p_c_j_z_j_list = mfcvae.compute_q_c_j_x(z_sample_q_z_j_x_list)

    for h in range(z_sample_q_z_j_x_list[0].shape[0]):  # is probably == batch size
        g = global_indices[h]
        index_to_z_j_embed[g] = [z_sample_q_z_j_x_list[j][h].unsqueeze(0).detach().cpu() for j in range(run_args.J_n_mixtures)]
        index_to_prob_p_c_j_z_j[g] = [prob_p_c_j_z_j_list[j][h].detach().cpu() for j in range(run_args.J_n_mixtures)]


    if run_args.dataset == 'fast_mnist':
        x = x.view(-1, in_channels, height, width).permute(0, 2, 3, 1)
    x_hat = x_hat.view(-1, in_channels, height, width).permute(0, 2, 3, 1)

    y_true = y_true.cpu().numpy()
    # y_true_list.append(y_true)

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

print("looped through test data.")

# build a useful data structure to handle the clustering probabilities
j_to_cluster_to_index_prob = {}
# create empty things
for j in range(run_args.J_n_mixtures):
    j_to_cluster_to_index_prob[j] = {}
    for c in range(run_args.n_clusters_j_list[j]):
        j_to_cluster_to_index_prob[j][c] = []

for (index, prob_list) in index_to_prob_p_c_j_z_j.items():
    for j in range(run_args.J_n_mixtures):
        cluster_j = torch.argmax(prob_list[j])
        cluster_j = cluster_j.item()
        j_to_cluster_to_index_prob[j][cluster_j].append((index, prob_list[j][cluster_j]))

# Sort clusters s.t. cluster with the largest "average confidence" is 0, second largest 1 etc.
cluster_average_confidence = {}
for j in range(run_args.J_n_mixtures):
    cluster_average_confidence[j] = {}
    for c in range(run_args.n_clusters_j_list[j]):
        cluster_average_confidence[j][c] = np.nan_to_num(
            np.mean([j_to_cluster_to_index_prob[j][c][k][1] for k in range(len(j_to_cluster_to_index_prob[j][c]))]))
    # sort
    cluster_index_average_confidence_list = [(cluster_j, score) for (cluster_j, score) in cluster_average_confidence[j].items()]
    cluster_index_average_confidence_list = sorted(cluster_index_average_confidence_list, key=lambda tuple: tuple[1], reverse=True)
    # print(cluster_index_count_list)
    cluster_j_sorted = [cluster_j for (cluster_j, score) in cluster_index_average_confidence_list]
    fromto_mapping = {cluster_j: i for i, cluster_j in enumerate(cluster_j_sorted)}
    # remap the dictionary - https://gist.github.com/pszaflarski/b139736415abbf8d344d77524baaece8
    j_to_cluster_to_index_prob[j] = {fromto_mapping.get(k, k): v for k, v in j_to_cluster_to_index_prob[j].items() if k in fromto_mapping}


# 2) sort the list within each cluster by most probable to least probable
for j in range(run_args.J_n_mixtures):
    for c in range(run_args.n_clusters_j_list[j]):
        j_to_cluster_to_index_prob[j][c] = sorted(j_to_cluster_to_index_prob[j][c], key=lambda tuple: tuple[1], reverse=True)

j_to_cluster_to_input = {}
j_to_cluster_to_count = {}
# create empty things
for j in range(run_args.J_n_mixtures):
    j_to_cluster_to_input[j] = {}
    j_to_cluster_to_count[j] = {}
    for c in range(run_args.n_clusters_j_list[j]):
        j_to_cluster_to_input[j][c] = []

# select only max of n_examples_per_cluster_to_show images per cluster
n_examples_per_cluster_to_show_max = 10
for j in range(run_args.J_n_mixtures):
    for c in range(run_args.n_clusters_j_list[j]):
        n_examples = min(len(j_to_cluster_to_index_prob[j][c]), n_examples_per_cluster_to_show_max)
        j_to_cluster_to_index_prob[j][c] = j_to_cluster_to_index_prob[j][c][:n_examples]
        j_to_cluster_to_count[j][c] = n_examples

# actually make the output dict of cluster examples
# populate
for j in range(run_args.J_n_mixtures):
    # print(j)
    for c in range(run_args.n_clusters_j_list[j]):
        # print(c)
        for tup in j_to_cluster_to_index_prob[j][c]:
            # print(test_data[tup[0]][0])
            j_to_cluster_to_input[j][c].append(((test_data[tup[0]][0]).cpu().detach(), 'sthelse'))  # .permute(1, 2, 0)

# print("Checkpoint 2.")


# PLOTTING OUT CLUSTER EXAMPLES ------------
for j in range(mfcvae.J_n_mixtures):
    fig = plot_cluster_examples_torch_grid(inputs_dict=j_to_cluster_to_input[j],
                                           count_dict=j_to_cluster_to_count[j],
                                           n_clusters=mfcvae.n_clusters_j_list[j],
                                           n_examples_per_cluster=n_examples_per_cluster_to_show_max)

    # plt.savefig('results/examples_top10/examples_top_10_facet_%d_full.pdf'%(j), format='pdf')  # , dpi=3000
    plt.show()
    plt.close()


# Analysis 1: same y_1_hat, swap z_0, keep z_1 -- SVHN
for swapped_cluster_0 in range(first_n_clusters_swapped_1):
    for swapped_cluster_1 in range(swapped_cluster_0, first_n_clusters_swapped_2):
# for swapped_cluster_0 in range(args.n_clusters_j_list[0 if swapped_facet == 1 else 1]):
#     for swapped_cluster_1 in range(swapped_cluster_0, args.n_clusters_j_list[swapped_facet]):
        print("cluster combo: %d_%d"%(swapped_cluster_0, swapped_cluster_1))
        comp_result_dir = os.path.join(results_dir, 'swap_clusters_%d_%d' % (swapped_cluster_0, swapped_cluster_1))
        if not os.path.exists(comp_result_dir):
            os.mkdir(comp_result_dir)


        # Controls if done manually (START) ---
        # swapped_facet = 1  # must be 1 in SVHN and 0 in MNIST (for style)
        # swapped_cluster_0 = 4  # style cluster 0 in facet 0 chosen
        # swapped_cluster_1 = 6  # style cluster 1 in facet 0 chosen
        # comp_result_dir = "results/comp_final/svhn/%d_%d" % (swapped_cluster_0, swapped_cluster_1)
        # Controls if done manually (END) ---

        # if not os.path.exists(comp_result_dir):
        #     os.mkdir(comp_result_dir)



        # find all indices with y_1_hat

        # Analysis 1 (strict) -> doesn't seem to work, since reconstructions are not reflecting style well
        # indices_style_0_chosen = [n for n in range(len(test_data)) if index_to_y_j_cluster[n][1] == y_1_hat_chosen and index_to_y_j_cluster[n][0] == style_0_cluster_y_0_hat_chosen]
        # indices_style_1_chosen = [n for n in range(len(test_data)) if index_to_y_j_cluster[n][1] == y_1_hat_chosen and index_to_y_j_cluster[n][0] == style_1_cluster_y_0_hat_chosen]
        # Analysis 2 (less strict)
        # way 1 to do it:
        # indices_swapped_cluster_0 = [n for n in range(len(test_data)) if index_to_y_j_cluster[n][0] == swapped_cluster_0]
        # indices_swapped_cluster_1 = [n for n in range(len(test_data)) if index_to_y_j_cluster[n][0] == swapped_cluster_1]
        # way 2 (using new data structure)
        indices_swapped_cluster_0 = [tup[0] for tup in j_to_cluster_to_index_prob[swapped_facet][swapped_cluster_0]]  # must be 0 for MNIST
        indices_swapped_cluster_1 = [tup[0] for tup in j_to_cluster_to_index_prob[swapped_facet][swapped_cluster_1]]

        # take the minimum of their two
        n_examples_swapped = min(len(indices_swapped_cluster_0), len(indices_swapped_cluster_1))
        if n_examples_swapped == 0:
            continue

        # print("examples in swapped cluster 0: %d"%len(indices_swapped_cluster_0))
        # print("examples in swapped cluster 1: %d"%len(indices_swapped_cluster_1))

        # only take the ones chosen
        indices_swapped_cluster_0 = indices_swapped_cluster_0[:n_examples_swapped]
        indices_swapped_cluster_1 = indices_swapped_cluster_1[:n_examples_swapped]

        # BEFORE ---

        # x - swapped cluster 0
        x_list_swapped_cluster_0 = [test_data[n][0] for n in indices_swapped_cluster_0]
        # x - swapped cluster 1
        x_list_swapped_cluster_1 = [test_data[n][0] for n in indices_swapped_cluster_1]

        # produce index_to_x_hat plot for chosen indices
        indices_to_reconstruct = indices_swapped_cluster_0 + indices_swapped_cluster_1
        for i in indices_to_reconstruct:
            x, y_true = test_data[i]
            x, y_true = x.to(device), y_true.to(device)
            x = x.unsqueeze(0)  # insert batch dimension, since just one sample taken  # .view(1, in_channels, height, width)
            if run_args.dataset == 'fast_mnist':
                x = x.view(x.size(0), -1).float()
            x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, int(sum(run_args.n_epochs_per_progressive_step)) - 1, 0)
            x_hat = x_hat.to('cpu')
            # if args.dataset == 'fast_mnist':
            x_hat = x_hat.view(in_channels, height, width).float()
            index_to_x_hat[i] = x_hat

        # just x hat
        x_hat_list_swapped_cluster_0 = [index_to_x_hat[n] for n in indices_swapped_cluster_0]
        x_hat_list_swapped_cluster_1 = [index_to_x_hat[n] for n in indices_swapped_cluster_1]

        # x and xhat
        x_and_x_hat_swapped_cluster_0 = x_list_swapped_cluster_0 + x_hat_list_swapped_cluster_0
        x_and_x_hat_swapped_cluster_1 = x_list_swapped_cluster_1 + x_hat_list_swapped_cluster_1



        # print(len(x_and_x_hat_swapped_cluster_1))


        def plot_torch_grid(img_list, nrow, title=None):
            grid_img = make_grid(img_list, nrow=nrow, pad_value=0, padding=0)

            # grid_img can be outside of allowed range [0, 1] -> clamp before imshow(...) call to avoid warning
            grid_img = torch.clamp(grid_img, min=0.0, max=1.0)

            # do plotting
            # undo permute and make numpy for imshow
            grid_img = grid_img.permute(1, 2, 0).cpu().detach().numpy()

            # plotting
            plt.clf()
            fig = plt.figure(figsize=(nrow, 2))
            ax = fig.add_subplot(1, 1, 1)

            if run_args.dataset == 'fast_mnist':
                ax.imshow(grid_img, cmap='gray')
            elif run_args.dataset in ['fast_svhn', 'fast_3dshapes']:
                ax.imshow(grid_img)
            else:
                exit("dataset not known.")

            if title is not None:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])

            fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

            return fig


        fig = plot_torch_grid(x_and_x_hat_swapped_cluster_0, n_examples_swapped, "Before swapping: Examples and reconstructions - Cluster 0")
        plt.savefig(os.path.join(comp_result_dir, 'x_x_hat_before_0.pdf'))  # , dpi=800
        plt.show()
        plt.close(fig=fig)

        fig = plot_torch_grid(x_and_x_hat_swapped_cluster_1, n_examples_swapped, "Before swapping: Examples and reconstructions - Cluster 1")
        plt.savefig(os.path.join(comp_result_dir, 'x_x_hat_before_1.pdf'))  # dpi=800
        plt.show()
        plt.close(fig=fig)

        # MNIST
        # n = 5748

        # SVHN
        # !!! For plotting just a single image!
        # n = 1
        # plot_image(test_data[n][0].permute(1, 2, 0).cpu().squeeze())  # + 0.5 only because of weird shifting
        # plot_image(index_to_x_hat[n].permute(1, 2, 0).cpu().detach().squeeze())

        # AFTER ---

        # we keep the digit facet in these list and swap the style
        z_sample_list_swapped_cluster_0 = [index_to_z_j_embed[n] for n in indices_swapped_cluster_0]
        z_sample_list_swapped_cluster_1 = [index_to_z_j_embed[n] for n in indices_swapped_cluster_1]

        # do swapping of style facet
        swap_done_z_sample_list_swapped_cluster_0, swap_done_z_sample_list_swapped_cluster_1 = [], []
        for i in range(n_examples_swapped):
            # !!! how the swapping is done
            swap_done_z_sample_list_swapped_cluster_0.append([z_sample_list_swapped_cluster_1[i][0].to(device), z_sample_list_swapped_cluster_0[i][1].to(device)])
            swap_done_z_sample_list_swapped_cluster_1.append([z_sample_list_swapped_cluster_0[i][0].to(device), z_sample_list_swapped_cluster_1[i][1].to(device)])

        x_hat_swapped_list_0 = [mfcvae.decode(swap_done_z_sample_list_swapped_cluster_0[i]).view(in_channels, height, width).detach().cpu() for i in range(n_examples_swapped)]
        x_hat_swapped_list_1 = [mfcvae.decode(swap_done_z_sample_list_swapped_cluster_1[i]).view(in_channels, height, width).detach().cpu() for i in range(n_examples_swapped)]

        if swapped_facet == 0:
            swapped_plot_list_0 = x_list_swapped_cluster_0 + x_hat_swapped_list_0
            swapped_plot_list_1 = x_list_swapped_cluster_1 + x_hat_swapped_list_1
        elif swapped_facet == 1:
            swapped_plot_list_0 = x_list_swapped_cluster_0 + x_hat_swapped_list_1
            swapped_plot_list_1 = x_list_swapped_cluster_1 + x_hat_swapped_list_0

        fig = plot_torch_grid(swapped_plot_list_0, n_examples_swapped, "After swapping: Examples and reconstructions - Combination 1")
        plt.savefig(os.path.join(comp_result_dir, 'x_x_hat_after_1.pdf'))  # , dpi=800
        plt.show()
        plt.close(fig=fig)

        fig = plot_torch_grid(swapped_plot_list_1, n_examples_swapped, "After swapping: Examples and reconstructions - Combination 0")
        plt.savefig(os.path.join(comp_result_dir, 'x_x_hat_after_0.pdf'))  # , dpi=800
        plt.show()
        plt.close(fig=fig)


print("compositionality done.")

