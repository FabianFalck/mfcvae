
from mfcvae import MFCVAE
import torch
from torchvision.datasets import MNIST, SVHN
from datasets import Fast_3DShapes
import torchvision
from torchvision import transforms


def load_model_from_save_dict(save_dict_path, map_location='cuda:0'):

    save_dict = torch.load(save_dict_path, map_location)
    state_dict = save_dict['state_dict']
    args = save_dict['args']
    args.device = map_location

    device = torch.device(args.device)

     # dataset-specific parameters
    if args.dataset in ['fast_mnist']:
        height, width = 28, 28
        n_true_classes = 10
        cmap = 'grey'
        in_channels = 1
        in_dim = 28 * 28 * in_channels
        likelihood_model = 'Bernoulli'
        activation_x_hat_z = "sigmoid"
        n_labels = 1
    elif args.dataset in ['fast_svhn']:
        height, width = 32, 32
        n_true_classes = 10
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = "sigmoid"    # BE CAREFUL!!!
        n_labels = 1
    elif args.dataset == 'fast_3dshapes':
        n_labels = len(args.factors_label_list)
        chosen_attr_list = args.factors_label_list
        n_true_classes = []
        for j in range(n_labels):
            n_true_classes.append(len(args.factors_variation_dict[args.factors_label_list[j]]))
            assert len(args.factors_variation_dict[args.factors_label_list[j]]) == n_true_classes[j]
        # print('number of true classes across labels: ', n_true_classes)
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = None  # 'sigmoid'  # 'sigmoid'  #  None  # 'sigmoid'
        n_labels = len(args.factors_label_list)


    # other dataset-specific hyperparams
    if args.dataset == 'fast_mnist':
        train_data = torchvision.datasets.MNIST('./data', train=True, download=True, transform=transforms.ToTensor())
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_svhn':
        train_data = torchvision.datasets.SVHN('./data', split='train', download=True, transform=transforms.ToTensor())
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    elif args.dataset == 'fast_3dshapes':
        train_data = Fast_3DShapes(train=True, device=args.device, train_frac = args.threedshapes_train_frac,
                                                     factors_variation_dict=args.factors_variation_dict,
                                                     factors_label_list=args.factors_label_list)
        n_train_batches_per_epoch = int(len(train_data) / args.batch_size)
    # print("n_train_batches_per_epoch: ", n_train_batches_per_epoch)


    mfcvae = MFCVAE(in_dim=in_dim, J_n_mixtures=args.J_n_mixtures, z_j_dim_list=args.z_j_dim_list,
                    n_clusters_j_list=args.n_clusters_j_list,
                    device=device, encode_layer_dims=args.encode_layer_dims,
                    decode_layer_dims=args.decode_layer_dims,
                    activation_x_hat_z=activation_x_hat_z, likelihood_model=likelihood_model,
                    sigma_multiplier_p_x_z=args.sigma_multiplier_p_x_z,
                    cov_type_p_z_c=args.cov_type_p_z_c,
                    init_type_p_z_c=args.init_type_p_z_c,
                    init_off_diag_cov_p_z_c=args.init_off_diag_cov_p_z_c,
                    model_type=args.model_type,
                    fix_pi_p_c=args.fix_pi_p_c,
                    facet_to_fix_pi_p_c=args.facet_to_fix_pi_p_c if 'facet_to_fix_pi_p_c' in (vars(args)).keys() else 'all',  # does nothing if 'fix_pi_p_c' is False
                    n_train_batches_per_epoch=n_train_batches_per_epoch,
                    n_epochs_per_progressive_step=args.n_epochs_per_progressive_step, n_batches_fade_in=args.n_batches_fade_in, gamma_kl_z_pretrain=args.gamma_kl_z_pretrain, gamma_kl_c_pretrain=args.gamma_kl_c_pretrain,
                    do_progressive_training=args.do_progressive_training,
                    activation=args.activation if 'activation' in (vars(args)).keys() else 'relu',
                    do_fc_batch_norm=args.do_fc_batch_norm)

    mfcvae.load_state_dict(state_dict)

    return mfcvae, args