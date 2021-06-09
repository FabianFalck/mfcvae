
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torchvision.utils import make_grid


def plot_inputs_and_recons(inputs_dict, recon_dict, count_dict, n_clusters, n_pairs_per_cluster):
    plt.clf()
    fig = plt.figure(figsize=(2.5*n_pairs_per_cluster, 1.5*n_clusters))
    for i in range(n_clusters):
        for j in range(count_dict[i]):
            plt.subplot(n_clusters, n_pairs_per_cluster*2, i * n_pairs_per_cluster * 2 + j * 2 + 1)
            if j == 0:
                plt.ylabel("Cluster %d"%(i))
            x = inputs_dict[i][j][0]
            if len(x.shape) == 2:
                plt.imshow(x, cmap='gray')
            else:
                plt.imshow(x)
            plt.xticks([])
            plt.yticks([])
            # .axis('off')
            # axes[i, j * 2].set_title(inputs_dict[i][j][1])
            plt.subplot(n_clusters, n_pairs_per_cluster*2, i*n_pairs_per_cluster * 2 + j * 2 + 1 + 1)
            x_hat = recon_dict[i][j][0]
            if len(x_hat.shape) == 2:
                plt.imshow(x_hat, cmap='gray')
            else:
                plt.imshow(x_hat)
            plt.axis('off')
            # axes[i, j * 2 + 1].set_title(inputs_dict[i][j][1])
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    # fig.suptitle("Input vs. Reconstruction", x=0.5, y=0.995)
    # fig.text(0.5, 0.99, 'Input vs. Reconstruction', ha='center')  # title
    fig.text(0.5, 0.99, 'Pairs of inputs and reconstructions for each predicted cluster', ha='center')  # common X label
    fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig



def plot_inputs_and_recons_torch_grid(inputs_dict, recon_dict, count_dict, n_clusters, n_pairs_per_cluster, show_empty_clusters=False):
    # find dimensions of one input
    for c in range(n_clusters):
        if count_dict[c] > 0:
            img_dims = inputs_dict[c][0][0].size()
            break

    if show_empty_clusters:
        n_clusters_plotted = n_clusters
        # include white images into lists
        for c in range(n_clusters):
            n_blank = n_pairs_per_cluster - count_dict[c]
            for j in range(n_blank):
                inputs_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
                recon_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
        # flattened list of images
        img_list = []
        for c in range(n_clusters):
            for j in range(n_pairs_per_cluster):
                img_list.append(inputs_dict[c][j][0])
                img_list.append(recon_dict[c][j][0])
    else:
        # include white images into lists
        for c in range(n_clusters):
            n_blank = n_pairs_per_cluster - count_dict[c]
            # do not fill up if all blank
            if not n_blank == n_pairs_per_cluster:
                for j in range(n_blank):
                    inputs_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
                    recon_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
        # flattened list of images
        img_list = []
        n_clusters_plotted = 0
        for c in range(n_clusters):
            # if any non-blank images in list: actually append the image
            if len(inputs_dict[c]) > 0:
                n_clusters_plotted += 1
                for j in range(n_pairs_per_cluster):
                    img_list.append(inputs_dict[c][j][0])
                    img_list.append(recon_dict[c][j][0])

    # make grid of images
    grid_img = make_grid(img_list, nrow=n_pairs_per_cluster*2, pad_value=0,  # black padding
                         padding=0)  # no padding
    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(2 * n_pairs_per_cluster, n_clusters_plotted))
    ax = fig.add_subplot(1, 1, 1)
    if len(img_dims) == 2:
        ax.imshow(grid_img, cmap='gray')
    else:
        ax.imshow(grid_img)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    # fig.text(0.5, 0.99, 'Pairs of inputs and reconstructions for each predicted cluster', ha='center')  # common X label
    # fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_cluster_examples(inputs_dict, count_dict, n_clusters, n_examples_per_cluster):
    plt.clf()
    fig = plt.figure(figsize=(1.5*n_examples_per_cluster, 1.5*n_clusters))
    for i in range(n_clusters):
        for j in range(count_dict[i]):
            plt.subplot(n_clusters, n_examples_per_cluster, i*n_examples_per_cluster + j + 1)
            if j == 0:
                plt.ylabel("Cluster %d"%(i))
            x = inputs_dict[i][j][0]
            if len(x.shape) == 2:
                plt.imshow(x, cmap='gray')
            else:
                plt.imshow(x)
            plt.xticks([])
            plt.yticks([])
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    fig.text(0.5, 0.99, 'Input examples for each cluster', ha='center')  # common X label
    fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_cluster_examples_torch_grid(inputs_dict, count_dict, n_clusters, n_examples_per_cluster, show_empty_clusters=False):
    # find dimensions of one input
    for c in range(n_clusters):
        if count_dict[c] > 0:
            img_dims = inputs_dict[c][0][0].size()
            break

    if show_empty_clusters:
        n_clusters_plotted = n_clusters
        # include white images into lists
        for c in range(n_clusters):
            n_blank = n_examples_per_cluster - count_dict[c]
            for j in range(n_blank):
                inputs_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
        # flattened list of images
        img_list = []
        for c in range(n_clusters):
            for j in range(n_examples_per_cluster):
                img_list.append(inputs_dict[c][j][0])
    else:
        # include white images into lists, but only if not entirely empty
        for c in range(n_clusters):
            n_blank = n_examples_per_cluster - count_dict[c]
            if not n_blank == n_examples_per_cluster:
                for j in range(n_blank):
                    inputs_dict[c].append((torch.ones(img_dims), "blank_image"))  # white
        # flattened list of images
        img_list = []
        n_clusters_plotted = 0
        for c in range(n_clusters):
            # if any non-blank images in list: actually append the image
            if len(inputs_dict[c]) > 0:
                n_clusters_plotted += 1
                for j in range(n_examples_per_cluster):
                    img_list.append(inputs_dict[c][j][0])

    # make grid of images
    grid_img = make_grid(img_list, nrow=n_examples_per_cluster, pad_value=0,  # black padding
                         padding=0)  # no padding
    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(n_examples_per_cluster, n_clusters_plotted))
    ax = fig.add_subplot(1, 1, 1)
    if len(img_dims) == 2:
        ax.imshow(grid_img, cmap='gray')
    else:
        ax.imshow(grid_img)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    # fig.text(0.5, 0.99, 'Input examples for each cluster', ha='center')  # common X label
    # fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_sample_generations_from_each_cluster(sample_dict, n_clusters, n_examples_per_cluster):
    plt.clf()
    fig = plt.figure(figsize=(1.5*n_examples_per_cluster, 1.5*n_clusters))
    for i in range(n_clusters):
        for j in range(n_examples_per_cluster):
            plt.subplot(n_clusters, n_examples_per_cluster, i*n_examples_per_cluster + j + 1)
            if j == 0:
                plt.ylabel("Cluster %d"%(i))
            x = sample_dict[i][j][0]
            if len(x.shape) == 2:
                plt.imshow(x, cmap='gray')
            else:
                plt.imshow(x)
            plt.xticks([])
            plt.yticks([])
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    fig.text(0.5, 0.99, 'Samples generated from each cluster', ha='center')  # common X label
    fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_sample_generations_from_each_cluster_torch_grid(sample_dict, n_clusters, n_examples_per_cluster):
    # find dimensions of one input
    img_dims = sample_dict[0][0][0].size()
    # flattened list of images
    img_list = []
    for c in range(n_clusters):
        for j in range(n_examples_per_cluster):
            img_list.append(sample_dict[c][j][0])
    # make grid of images
    grid_img = make_grid(img_list, nrow=n_examples_per_cluster, pad_value=0,  # black padding
                         padding=0)  # no padding
    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(n_examples_per_cluster, n_clusters))
    ax = fig.add_subplot(1, 1, 1)
    if len(img_dims) == 2:
        ax.imshow(grid_img, cmap='gray')
    else:
        ax.imshow(grid_img)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])

    return fig


def plot_cluster_mean_one_facet_recons(recon_dict, n_clusters):
    """
    Used in VaDE.
    """
    plt.clf()
    fig = plt.figure(figsize=(3, 1.5*n_clusters))
    for i in range(n_clusters):
        plt.subplot(n_clusters, 1, i + 1)
        x_hat = recon_dict[i][0]
        if len(x_hat.shape) == 2:
            plt.imshow(x_hat, cmap='gray')
        else:
            plt.imshow(x_hat)
        plt.xticks([])
        plt.yticks([])
        plt.ylabel("Cluster %d"%(i))
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    fig.text(0.5, 0.99, 'Reconstruction of means of p(z | c)', ha='center')  # common X label
    fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_cluster_mean_two_facets_recons(recon_dict, n_clusters_0, n_clusters_1):
    plt.clf()
    fig = plt.figure(figsize=(1.5 * n_clusters_1, 1.5 * n_clusters_0))   # width, height
    for m in range(n_clusters_0):
        for n in range(n_clusters_1):
            plt.subplot(n_clusters_0, n_clusters_1, m * n_clusters_1 + n + 1)
            if n == 0:
                plt.ylabel("Cluster %d" % (m), fontsize=10)
            if m == 0:
                plt.title("Cluster %d" % (n), fontsize=10)
            x_hat = recon_dict[(m, n)][0]
            if len(x_hat.shape) == 2:
                plt.imshow(x_hat, cmap='gray')
            else:
                plt.imshow(x_hat)
            plt.xticks([])
            plt.yticks([])
    fig.tight_layout(rect=[0.02, 0.01, 0.99, 0.98])
    # fig.suptitle("Reconstruction of means of p(z_0 | c_0) and p(z_1 | c_1) combinations", x=0.5, y=0.995)
    fig.text(0.5, 0.985, 'p(z_1 | c_1) (Facet 1)', ha='center')  # common X label
    fig.text(0.005, 0.5, 'p(z_0 | c_0) (Facet 0)', va='center', rotation='vertical')  # common Y label

    return fig


def plot_cluster_mean_two_facets_recons_torch_grid(recon_dict, n_clusters_0, n_clusters_1):
    # find dimensions of one input
    img_dims = (recon_dict[0, 0][0]).size()
    # flattened list of images
    img_list = []
    for c in range(n_clusters_0):
        for j in range(n_clusters_1):
            img_list.append(recon_dict[c, j][0])
    # make grid of images
    grid_img = make_grid(img_list, nrow=n_clusters_1, pad_value=0,  # black padding
                         padding=0)  # no padding
    # undo permute and make numpy for imshow
    grid_img = grid_img.permute(1, 2, 0).numpy()

    # plotting
    plt.clf()
    fig = plt.figure(figsize=(n_clusters_1, n_clusters_0))
    ax = fig.add_subplot(1, 1, 1)
    if len(img_dims) == 2:
        ax.imshow(grid_img, cmap='gray')
    else:
        ax.imshow(grid_img)
    ax.set_xticks([])
    ax.set_yticks([])

    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    # fig.text(0.5, 0.99, 'Pairs of inputs and reconstructions for each predicted cluster', ha='center')  # common X label
    # fig.text(0.005, 0.5, 'Predicted clusters', va='center', rotation='vertical')  # common Y label

    return fig


def plot_confusion_matrix(conf_mat, n_true_classes):
    plt.clf()
    fig, ax = plt.subplots(figsize=(n_true_classes * .5, n_true_classes * .5))
    ax = sns.heatmap(conf_mat, linewidth=0.5, annot=True, annot_kws={"size": 5}, ax=ax)
    ax.set_ylabel('Predicted classes')
    ax.set_xlabel('True classes')

    return fig


def plot_n_inputs_per_cluster(y_pred_count, n_clusters):
    samples_per_cluster = np.concatenate((y_pred_count, np.zeros(n_clusters - len(y_pred_count))))
    plt.clf()
    fig, ax = plt.subplots()
    sns.ecdfplot(x=samples_per_cluster)
    sns.rugplot(x=samples_per_cluster)
    ax.set_xlabel('Number of inputs per predicted cluster')
    ax.set_ylabel('Cumulative proportion')

    return fig

def plot_latent_code_traversal(recon_dict, n_dim, traversal_range=[-3, 3], n_traversal = 10, n_random_samples=5):
    plt.clf()
    fig = plt.figure(figsize=(1.5 * (n_dim * n_random_samples), 1.5 * n_traversal))   # width, height
    traversal_steps = np.linspace(traversal_range[0], traversal_range[1], n_traversal)
    for nz in range(n_dim):
        for N in range(n_random_samples):
            for nt in range(n_traversal):
                plt.subplot(n_dim * n_random_samples, n_traversal, nz * n_random_samples * n_traversal + N * n_traversal + nt + 1)
                x_hat = recon_dict[nz][N][nt]
                if len(x_hat.shape) == 2:
                    plt.imshow(x_hat, cmap='gray')
                else:
                    plt.imshow(x_hat)
                plt.xticks([])
                plt.yticks([])
    fig.tight_layout(rect=[0.01, 0.01, 0.99, 0.98])
    return fig

def plot_pi(pi_p_c_i):
    plt.clf()
    pi_p_c_i = pi_p_c_i / np.sum(pi_p_c_i)
    fig = plt.figure()
    plt.bar(range(len(pi_p_c_i)), pi_p_c_i)
    plt.xlabel('Cluster index')
    plt.ylabel('Parameter value')
    return fig


def plot_dict(plot_dict):
    plt.clf()
    fig = plt.figure()
    plt.title("Facet to label mapping")
    for i, (k, v) in enumerate(plot_dict.items()):
        string = str(k) + " --> " + v
        plt.text(0.3, 0.9 - i * 0.04, string)
    plt.axis('off')

    return fig

if __name__ == '__main__':
    facet_to_label = {0: "A", 1: "B", 2: "C"}
    plot_dict(facet_to_label)