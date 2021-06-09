"""
- Initial source code was oriented on and inspired from the following script (and repository): https://github.com/eelxpeng/UnsupervisedDeepLearning-Pytorch/blob/master/udlp/clustering/vade.py
"""


def train():


    # fix for "RuntimeError: received 0 items of ancdata"
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))

    import argparse
    import torch
    import torchvision
    from torchvision import transforms
    import torch.distributions as D
    import torch.optim as optim
    from torch.autograd import Variable
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    # import datetime
    import pickle
    import boilr
    if int(boilr.__version__[2]) == 5 or int(boilr.__version__[2]) == 6 and int(boilr.__version__[4]) < 4:
        from boilr.nn_init import data_dependent_init
    else:
        from boilr.nn.init import data_dependent_init
    from mfcvae import MFCVAE

    torch.backends.cudnn.benchmark = True  # for potential speedup, see https://www.reddit.com/r/MachineLearning/comments/kvs1ex/d_here_are_17_ways_of_making_pytorch_training/ (6.)

    from utils import cluster_acc_and_conf_mat, cluster_acc_weighted, str2bool, load_args_from_yaml
    from datasets import Fast_MNIST, Fast_SVHN, Fast_3DShapes

    from plotting import plot_confusion_matrix, plot_n_inputs_per_cluster, plot_pi, plot_dict, \
        plot_cluster_examples_torch_grid, plot_inputs_and_recons_torch_grid, \
        plot_sample_generations_from_each_cluster_torch_grid
    from eval_top10_cluster_examples import plot_top10_cluster_examples
    from eval_sample_generation import plot_sample_generation

    # avoid weird error
    torch.set_num_threads(1)


    parser = argparse.ArgumentParser(description='MFCVAE training')

    # important configs
    parser.add_argument('--device', type=str, default="cuda:0", metavar='N',
                    help="device to use for all heavy tensor operations, e.g. 'cuda:0', 'cpu', ...")
    parser.add_argument('--wandb_mode', type=str, default="online", metavar='N',
                    help="mode of wandb run tracking, either no tracking ('disabled') or with tracking ('online')")
    parser.add_argument('--user', type=str, default='user1', metavar='N',
                        help="which user to login as ('user1' or 'user2')")
    parser.add_argument('--config_args_path', type=str, default="", metavar='N',  # e.g. "configs/svhn.yml"
                        help="the path to the args config namespace to be loaded. If a path is provided, all specifications of hyperparameters above are ignored. \
                            File can either end in '.pkl' or '.yaml', and depending on the file type, different ways of loading the file will be used. \
                            If the argument is an empty string, the hyperparameter specifications above are used as usual.")

    parser.add_argument('--model_type', type=str, default='fc_vlae', metavar='N',
                        help="model type to use, EITHER 'fc_shared' (fully-connected, one shared encoder) OR 'fc_per_facet_enc_shared_dec' (fully-connected, J non-shared encoders, one per facet, shared decoder) \
                              OR 'fc_vlae' (fully-connected, VLAE type encoder) OR VLAE as in progressive training paper ('conv_vlae')")
    parser.add_argument('--dataset', type=str, default='fast_mnist', metavar='N',
                        help="dataset used during training, one in ['fast_mnist', 'fast_svhn', 'fast_3dshapes']")

    # model
    parser.add_argument('--J_n_mixtures', type=int, default=2, metavar='N', help='J - number of clusterings/facets/mixtures')
    parser.add_argument('--z_j_dim_list', type=list, default=[5, 5], metavar='N',
                        help='dimension of each z_j variable')
    parser.add_argument('--n_clusters_j_list', type=list, default=[25, 25], metavar='N',
                        help='number of possible values of each c_j')
    parser.add_argument('--save_model', type=str2bool, nargs='?', dest='save_model', const=True, default=True,
                        help='whether to save the model or not')

    # progressive training
    parser.add_argument('--do_progressive_training', type=str2bool, nargs='?', dest='do_progressive_training', const=True, default=True,
                        help='whether to do progressive training or not')
    parser.add_argument('--n_epochs_per_progressive_step', type=list, default=[201, 201], metavar='N', help="number of epochs during each progressive training step")
    parser.add_argument('--n_batches_fade_in', type=int, default=15000, metavar='N', help='number of batches during which to fade in at the start of a progressive step')
    parser.add_argument('--gamma_kl_z_pretrain', type=float, default=0.0, metavar='N', help='gamma factor used on z KL-terms during the pretraining phase')
    parser.add_argument('--gamma_kl_c_pretrain', type=float, default=0.0, metavar='N', help='gamma factor used on c KL-terms during the pretraining phase')

    # 3DShapes configs
    threedshapes_configs = {
        '1': ({'floor_hue': list(range(10)), 'wall_hue': [5], 'object_hue': [6], 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))}, ['shape', 'floor_hue']),
        '2': ({'floor_hue': [0], 'wall_hue': list(range(10)), 'object_hue': [6], 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))}, ['shape', 'wall_hue']),
    }
    parser.add_argument('--factors_variation_dict', type=dict, default=threedshapes_configs['2'][0], metavar='N', help="A dictionary indicating the factors used as variation. Keys of this dictionary must be ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'] Values are lists of integers (which must contain at least one item). The following dictionary indicates the 'maximum chosen dictionary', of which sublists can be chosen in the values: {'floor_hue': list(range(10)), 'wall_hue': list(range(10)), 'object_hue': list(range(10)), 'scale': list(range(8)), 'shape': list(range(4)), 'orientation': list(range(15))}")
    parser.add_argument('--factors_label_list', type=list, default=threedshapes_configs['2'][1], metavar='N', help="A list indicating the factors used for labels. The list must be a sublist of ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'], containing at least 1 element.")
    parser.add_argument('--threedshapes_train_frac', type=float, default=0.8, metavar='N', help='fraction of samples used for training, a float in [0, 1]')

    # layer configs
    parser.add_argument('--activation', type=str, default='relu', metavar='N', help="which activation to choose, one from ['relu', 'leaky_relu', 'elu']")
    parser.add_argument('--do_fc_batch_norm', type=str2bool, nargs='?', dest='do_fc_batch_norm',
                        const=True, default=False, help="whether to have a batch norm layer in the build_fc_network(...) network function or not")

    # 'fc_shared' and 'fc_per_facet_enc_shared_dec' configs
    parser.add_argument('--encode_layer_dims', type=list, default=[500, 2000],
                        metavar='N', help="if args.encode_type is 'fc_shared': one list of integers, each indicating the output dimensions of each hidden layer. \
                              if args.encode_type is 'fc_per_facet_enc_shared_dec': J lists of integers, each list representing one facet, and the integers indicating output dims of each hidden layer.")
    parser.add_argument('--decode_layer_dims', type=list, default=[2000, 500], metavar='N',
                        help="if args.decode_type is 'fc_shared': one list of integers, each indicating the output dimensions of each hidden layer. \
                              if args.decode_type is 'fc_per_facet_enc_shared_dec': J lists of integers, each list representing one facet, and the integers indicating output dims of each hidden layer.")

    # 'fc_vlae' configs
    parser.add_argument('--enc_backbone_dims', type=list, default=[500, 2000], metavar='N', help="list of integers indicating output dims in the j-th encoder backbone")
    parser.add_argument('--enc_backbone_n_hidden', type=list, default=[1, 1], metavar='N', help="list of integers indicating number of hidden layers in the j-th encoder backbone")
    parser.add_argument('--enc_rung_dims', type=list, default=[-1, -1], metavar='N', help="list of integers indicating output dims in the j-th encoder rung")
    parser.add_argument('--enc_rung_n_hidden', type=list, default=[-1, -1], metavar='N', help="list of integers indicating number of hidden layers in the j-th encoder rung")
    # -
    parser.add_argument('--dec_backbone_dims', type=list, default=[500, 500], metavar='N', help="list of integers indicating output dims in the j-th decoder backbone")
    parser.add_argument('--dec_backbone_n_hidden', type=list, default=[1, 1], metavar='N', help="list of integers indicating number of hidden layers in the j-th decoder backbone")
    parser.add_argument('--dec_rung_dims', type=list, default=[500, 2000], metavar='N', help="list of integers indicating output dims in the j-th decoder rung")
    parser.add_argument('--dec_rung_n_hidden', type=list, default=[1, 1], metavar='N', help="list of integers indicating number of hidden layers in the j-th decoder rung")

    # configs for p(x | z), p(z | c) and p(c)
    parser.add_argument('--sigma_multiplier_p_x_z', type=float, default=0.6, metavar='N',
                        help='in the case of a Gaussian likelihood model p(x | z), the mulitplier of the diagonal elements of the covariance matrix (standard deviations)')
    parser.add_argument('--cov_type_p_z_c', type=str, default="diag", metavar='N',
                        help="covariance type of p(z_0 | c_0) and p(z_1 | c_1), with options diagonal ('diag') and full ('full')")
    parser.add_argument('--init_type_p_z_c', type=str, default="gmm", metavar='N',
                        help="initialisation method for all p(z_j | c_j), with options glorot initialisation ('glorot'), gmm initialisation ('gmm').")
    parser.add_argument('--fixed_var_init', type=float, default=0.01, metavar='N',
                        help='the fixed value used for initialising variance (in "diag" case) or diagonal values in covariance (in "full" case) of p(z|c)')
    parser.add_argument('--init_off_diag_cov_p_z_c', type=str2bool, nargs='?', dest='init_off_diag_cov_p_z_c', const=True, default=False,
                        help="whether to initialize off-diagonal entries in the covariance matrix of p(z | c) to be non-zeros, if cov_type_p_z_c='full' and init_type_p_z_c='glorot'")
    parser.add_argument('--fix_pi_p_c', type=str2bool, nargs='?', dest='fix_pi_p_c',
                        const=True, default=False, help="whether to fix parameter pi in p(c_j). which facet(s) are fixed is chosen in args argument --facet_to_fix_pi_p_c (see below)")
    parser.add_argument('--facet_to_fix_pi_p_c', type=str, default="all", metavar='N',
                        help="the facet to fix parameter pi in p(c), if fix_pi_p_c=True; choose from ['all', 'facet_0', 'facet_1']")

    # further configs
    parser.add_argument('--init_lr', type=float, default=0.0005, metavar='N',
                        help='initial learning rate for training')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='input batch size for training')
    parser.add_argument('--eval_batch_size', type=int, default=1000, metavar='N',
                        help='input batch size for evaluation')
    parser.add_argument('--n_test_batches', type=int, default=-1, metavar='N',
                        help='number of test batches to use per evaluation (-1 uses all available batches)')

    parser.add_argument('--n_epochs', type=int, default=301,  # 301
                        metavar='N', help='number of epochs to train. overwritten by sum(args.n_epochs_per_progressive_step), if args.do_progressive_training == True')
    parser.add_argument('--seed', type=int, default=6000, metavar='N',
                        help='seed for pseudo-randomness')

    # visualisation configs
    parser.add_argument('--do_test_during_training', type=str2bool, nargs='?', dest='do_test_during_training', const=True, default=True,
                    help='whether to perform evaluation on the test set throughout training, if True, do it for every epoch, otherwise only do it at the end of the training')
    parser.add_argument('--do_vis_train', type=str2bool, nargs='?', dest='do_vis_train', const=True, default=True,
                        help='whether to perform image logging on training set')
    parser.add_argument('--do_vis_test', type=str2bool, nargs='?', dest='do_vis_test', const=True, default=True,
                        help='whether to perform image logging on test set')
    parser.add_argument('--do_vis_recon', type=str2bool, nargs='?', dest='do_vis_recon', const=True, default=True,
                        help='whether to perform image logging of inputs and reconstructions')
    parser.add_argument('--do_vis_examples_per_cluster', type=str2bool, nargs='?', dest='do_vis_examples_per_cluster',
                        const=True, default=True,
                        help='whether to perform image logging of example images per cluster')
    parser.add_argument('--do_vis_sample_generations_per_cluster', type=str2bool, nargs='?', dest='do_vis_sample_generations_per_cluster', const=True, default=True,
                        help='whether to perform image logging of sample generations for each cluster')
    parser.add_argument('--do_vis_conf_mat', type=str2bool, nargs='?', dest='do_vis_conf_mat', const=True, default=True,
                        help='whether to perform image logging of confusion matrices')
    parser.add_argument('--do_vis_n_samples_per_cluster', type=str2bool, nargs='?', dest='do_vis_n_samples_per_cluster',
                        const=True, default=True,
                        help='whether to perform image logging of the number of samples per cluster')
    parser.add_argument('--do_vis_pi', type=str2bool, nargs='?', dest='do_vis_pi', const=True, default=True,
                        help='whether to perform visualisation of pi_p_c_0 and pi_p_c_1')
    parser.add_argument('--vis_every_n_epochs', type=int, default=50,
                          metavar='N', help='every how many epochs to do image logging')
    parser.add_argument('--n_recons_per_cluster', type=int, default=4,
                        metavar='N', help='how many input-recon-pairs per cluster to log')
    parser.add_argument('--n_examples_per_cluster', type=int, default=10,
                        metavar='N', help='how many examples per cluster to log')
    parser.add_argument('--n_sample_generations_per_cluster', type=int, default=10,
                        metavar='N', help='how many sample generations per cluster to log')
    parser.add_argument('--temp_sample_generation', type=float, default=0.5, metavar='N',
                        help='temperature which scales the variance in sample generation plots')

    parser.add_argument('--data_init', type=str2bool, nargs='?', dest='data_init', const=True, default=False,
                        help='whether to perform data-aware init of model (True), or initialise the model randomly (False)')

    args, unknown = parser.parse_known_args()
    print("args not parsed in train: ", unknown)

   # load config dictionary instead
    if args.config_args_path != "":
        if '.pkl' in args.config_args_path:
            with open(args.config_args_path, 'rb') as file:
                print("NOTE: Loading args configuration dictionary from .pkl which overrides any specified hyperparameters in train.py!")
                args = pickle.load(file)
        elif '.yml' in args.config_args_path:
            print("NOTE: Loading args configuration dictionary from .yaml which overrides any specified hyperparameters in train.py!")
            args = load_args_from_yaml(args.config_args_path)
        else:
            exit("No loading method for args config available.")

    import wandb

    # load wandb api key, project name and entity name
    wandb_args = load_args_from_yaml('configs/wandb.yml')

    # set the right user to login
    # The API key can be found under your account (top-right) when logged in -> settings -> API key
    if args.user == 'user1':
        wandb.login(key=wandb_args.login_key_1)
    elif args.user == 'user2':
        wandb.login(key=wandb_args.login_key_2)
    else:
        exit("user incorrect")

    # not passing config, since sweep defined configs are defined for this run in config already
    # team_name is the name of the team shared with others in wandb
    # project_name is a project within the team with team_name
    wandb_run = wandb.init(project=wandb_args.project_name, entity=wandb_args.team_name,
                           mode=args.wandb_mode)
    # for sweep: don't use such values of args above which are defined by sweep
    # set args value to sweep values instead
    for (key, value) in wandb.config.items():
        setattr(args, key, value)  # args is namespace object

    # sweep configs updating args configs
    if 'z_dim' in wandb.config.keys():
        args.z_j_dim_list = [wandb.config['z_dim']] * args.J_n_mixtures
    elif 'z_0_dim' in wandb.config.keys() and 'z_1_dim' in wandb.config.keys() and 'z_2_dim' in wandb.config.keys():
        args.z_j_dim_list = [wandb.config['z_0_dim'], wandb.config['z_1_dim'], wandb.config['z_2_dim']]
    elif 'z_0_dim' in wandb.config.keys() and 'z_1_dim' in wandb.config.keys():
        args.z_j_dim_list = [wandb.config['z_0_dim'], wandb.config['z_1_dim']]
    elif 'z_0_dim' in wandb.config.keys() and args.J_n_mixtures == 2:
        args.z_j_dim_list = [wandb.config['z_0_dim'], args.z_j_dim_list[1]]
    elif 'z_1_dim' in wandb.config.keys() and args.J_n_mixtures == 2:
        args.z_j_dim_list = [args.z_j_dim_list[0], wandb.config['z_1_dim']]

    if 'n_clusters_0' in wandb.config.keys() and 'n_clusters_1' in wandb.config.keys() and 'n_clusters_2' in wandb.config.keys() and 'n_clusters_3' in wandb.config.keys() and 'n_clusters_4' in wandb.config.keys():
        args.n_clusters_j_list = [wandb.config['n_clusters_0'], wandb.config['n_clusters_1'],
                                  wandb.config['n_clusters_2'], wandb.config['n_clusters_3'],
                                  wandb.config['n_clusters_4']]
    elif 'n_clusters_0' in wandb.config.keys() and 'n_clusters_1' in wandb.config.keys() and 'n_clusters_2' in wandb.config.keys() and 'n_clusters_3' in wandb.config.keys():
        args.n_clusters_j_list = [wandb.config['n_clusters_0'], wandb.config['n_clusters_1'],
                                  wandb.config['n_clusters_2'], wandb.config['n_clusters_3']]
    elif 'n_clusters_0' in wandb.config.keys() and 'n_clusters_1' in wandb.config.keys() and 'n_clusters_2' in wandb.config.keys():
        args.n_clusters_j_list = [wandb.config['n_clusters_0'], wandb.config['n_clusters_1'],
                                  wandb.config['n_clusters_2']]
    elif 'n_clusters_0' in wandb.config.keys() and 'n_clusters_1' in wandb.config.keys():
        args.n_clusters_j_list = [wandb.config['n_clusters_0'], wandb.config['n_clusters_1']]
    # vlae sweep configs to args configs
    # NOTE: this does not work with gated_add in decoder -> taken out for now
    # if "backbone_dim" in wandb.config.keys():
    #     args.enc_backbone_dims = [wandb.config['backbone_dim']] * args.J_n_mixtures
    #     args.dec_backbone_dims = [wandb.config['backbone_dim']] * args.J_n_mixtures
    # if "backbone_n_hidden" in wandb.config.keys():
    #     args.enc_backbone_n_hidden = [wandb.config['backbone_n_hidden']] * args.J_n_mixtures
    #     args.dec_backbone_n_hidden = [wandb.config['backbone_n_hidden']] * args.J_n_mixtures
    # if "rung_dim" in wandb.config.keys():
    #     args.enc_rung_dims = [wandb.config['rung_dim']] * args.J_n_mixtures
    #     args.dec_rung_dims = [wandb.config['rung_dim']] * args.J_n_mixtures
    # if "rung_n_hidden" in wandb.config.keys():
    #     args.enc_rung_n_hidden = [wandb.config["rung_n_hidden"]] * args.J_n_mixtures
    #     args.dec_rung_n_hidden = [wandb.config["rung_n_hidden"]] * args.J_n_mixtures
    # -
    if "hidden_dim" in wandb.config.keys():
        args.enc_backbone_dims = [wandb.config['hidden_dim']] * args.J_n_mixtures
        args.dec_backbone_dims = [wandb.config['hidden_dim']] * args.J_n_mixtures
        args.enc_rung_dims = [wandb.config['hidden_dim']] * args.J_n_mixtures
        args.dec_rung_dims = [wandb.config['hidden_dim']] * args.J_n_mixtures
    if "n_hidden" in wandb.config.keys():
        args.enc_backbone_n_hidden = [wandb.config['n_hidden']] * args.J_n_mixtures
        args.dec_backbone_n_hidden = [wandb.config['n_hidden']] * args.J_n_mixtures
        args.enc_rung_n_hidden = [wandb.config["n_hidden"]] * args.J_n_mixtures
        args.dec_rung_n_hidden = [wandb.config["n_hidden"]] * args.J_n_mixtures

    # build encode_layer_dims and decode_layer_dims required as input to 'fc_vlae' type model
    if args.model_type == 'fc_vlae':
        # construct correct encoder and decoder layer dims
        args.encode_layer_dims, args.decode_layer_dims = [], []
        for j in range(args.J_n_mixtures):
            args.encode_layer_dims.append([args.enc_backbone_dims[j]] * args.enc_backbone_n_hidden[j] + ['branch'] + [args.enc_rung_dims[j]] * args.enc_rung_n_hidden[j])
            if j == args.J_n_mixtures - 1:
                args.decode_layer_dims.append(
                    [args.dec_rung_dims[j]] * args.dec_rung_n_hidden[j] + [args.dec_backbone_dims[j]] * args.dec_backbone_n_hidden[j])  # note the order!
            else:
                args.decode_layer_dims.append(
                    [args.dec_rung_dims[j]] * args.dec_rung_n_hidden[j] + ['merge'] + [args.dec_backbone_dims[j]] * args.dec_backbone_n_hidden[j])  # note the order!

    # update configs -> remember hyperparams
    wandb.config.update(args)

    print("args: ")
    print(args)
    print("wandb.run.dir: ", wandb.run.dir)

    # asserts
    if args.do_progressive_training:
        assert args.J_n_mixtures == len(args.z_j_dim_list) == len(args.n_clusters_j_list) == len(args.n_epochs_per_progressive_step)
    else:
        assert args.J_n_mixtures == len(args.z_j_dim_list) == len(args.n_clusters_j_list)
    if args.model_type == 'fc_shared':
        assert type(args.encode_layer_dims[0]) is int
        assert type(args.decode_layer_dims[0]) is int
    elif args.model_type == 'fc_per_facet_enc_shared_dec':
        assert args.J_n_mixtures == len(args.encode_layer_dims)
        assert type(args.decode_layer_dims[0]) is int
    elif args.model_type == 'fc_vlae':
        assert len(args.enc_backbone_dims) == len(args.enc_backbone_n_hidden) == len(args.enc_rung_dims) == len(
            args.enc_rung_n_hidden) == \
               len(args.dec_backbone_dims) == len(args.dec_backbone_n_hidden) == len(args.dec_rung_dims) == len(
            args.dec_rung_n_hidden)

        # check that layer_dims are correctly specified
        for j in range(args.J_n_mixtures):
            assert 'branch' in args.encode_layer_dims[j]
        # J-th list shall not include 'merge' (edge case, since no backbone input)
        assert 'merge' not in args.decode_layer_dims[args.J_n_mixtures - 1]
        for j in range(args.J_n_mixtures - 1):
            assert 'merge' in args.decode_layer_dims[j]

    if args.fix_pi_p_c:
        assert args.J_n_mixtures == 2

    # make device a global variable so that dataset.py can access it
    global device
    # initializing global variable (see above)
    device = torch.device(args.device)

    # make deterministic
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'fast_mnist':
        print("Initialize MNIST dataset and data loaders...")
        # initialize dataset
        train_data = Fast_MNIST('./data', train=True, download=True,
                                device=args.device)  # before: torchvision.datasets.MNIST
        test_data = Fast_MNIST("./data", train=False, device=args.device)  # before: torchvision.datasets.MNIST
        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
    elif args.dataset == 'fast_svhn':
        print("Initialize SVHN dataset and data loaders...")
        train_data = Fast_SVHN('./data', split='train', download=True, device=args.device)
        test_data = Fast_SVHN("./data", split='test', download=True, device=args.device)
        # make dataset compatible with multi-label datasets
        train_data.labels = train_data.labels.unsqueeze(1)
        test_data.labels = test_data.labels.unsqueeze(1)
        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=True,
                                                  num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
    elif args.dataset == 'fast_3dshapes':
        print("Initialize 3DShapes dataset and data loaders...")
        train_data = Fast_3DShapes(train=True, device=args.device, train_frac = args.threedshapes_train_frac,
                                                     factors_variation_dict=args.factors_variation_dict,
                                                     factors_label_list=args.factors_label_list,
                                                     seed=args.seed)
        test_data = Fast_3DShapes(train=False, device=args.device, train_frac = args.threedshapes_train_frac,
                                                     factors_variation_dict=args.factors_variation_dict,
                                                     factors_label_list=args.factors_label_list,
                                                     seed=args.seed)
        # initialize data loaders
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                                   num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=True,
                                                  num_workers=0)  # must be 0 with GPU, good article: https://discuss.pytorch.org/t/cuda-initialization-error-when-dataloader-with-cuda-tensor/43390

    else:
        raise ValueError("args.dataset has not chosen an implemented dataset")

    # dataset-specific parameters
    if args.dataset == 'fast_mnist':
        height, width = 28, 28
        n_true_classes = 10
        cmap = 'grey'
        in_channels = 1
        in_dim = 28 * 28 * in_channels
        likelihood_model = 'Bernoulli'
        activation_x_hat_z = "sigmoid"
        n_labels = 1
    elif args.dataset == 'fast_svhn':
        height, width = 32, 32
        n_true_classes = 10
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = 'sigmoid'
        n_labels = 1
    elif args.dataset == 'fast_3dshapes':
        height, width = 32, 32
        n_labels = len(args.factors_label_list)
        chosen_attr_list = args.factors_label_list
        n_true_classes = []
        for j in range(n_labels):
            n_true_classes.append(len(args.factors_variation_dict[args.factors_label_list[j]]))
            assert len(args.factors_variation_dict[args.factors_label_list[j]]) == n_true_classes[j]
        cmap = 'viridis'
        in_channels = 3
        in_dim = 32 * 32 * in_channels
        likelihood_model = 'Gaussian'
        activation_x_hat_z = None

    # other data-specific hyperparammers
    n_train_batches_per_epoch = len(train_loader)

    assert height == width

    print("Make model object...")
    if args.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae', 'vlae_orig', 'conv_vlae']:
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
                        facet_to_fix_pi_p_c=args.facet_to_fix_pi_p_c,
                        n_train_batches_per_epoch=n_train_batches_per_epoch,
                        do_progressive_training=args.do_progressive_training,
                        n_epochs_per_progressive_step=args.n_epochs_per_progressive_step, n_batches_fade_in=args.n_batches_fade_in, gamma_kl_z_pretrain=args.gamma_kl_z_pretrain, gamma_kl_c_pretrain=args.gamma_kl_c_pretrain,
                        activation=args.activation, do_fc_batch_norm=args.do_fc_batch_norm)

    print("Initialize model randomly (from scratch)...")
    mfcvae.initialize_fc_layers()
    if args.data_init:
        t = [train_data[i] for i in range(args.batch_size)]
        t = torch.stack(tuple(t[i][0] for i in range(len(t)))).to("cpu")
        if args.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae']:
            t = t.view(t.size(0), -1).float()
        elif args.model_type in ['conv_vlae']:
            t = t.float()
        # Use batch for data dependent init
        with torch.no_grad():
            init_do_progressive_training = mfcvae.do_progressive_training
            mfcvae.do_progressive_training = False
            mfcvae.encoder.do_progressive_training = False
            mfcvae.decoder.do_progressive_training = False
            first_forward = mfcvae(t, sum(args.n_epochs_per_progressive_step)-1, 0)
            data_dependent_init(mfcvae, {'x': t,
                                        'epoch': 0,
                                        'batch_idx': 0})
            mfcvae.do_progressive_training = init_do_progressive_training
            mfcvae.do_progressive_training = args.do_progressive_training
            mfcvae.encoder.do_progressive_training = args.do_progressive_training
            mfcvae.decoder.do_progressive_training = args.do_progressive_training

    # print a model summary
    # if args.model_type == 'fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae':
    #     torchsummary.summary(model=mfcvae, input_size=(height*width*in_channels, ), device='cpu')
    # elif args.model_type in ['resnet', 'convnet', 'vlae_orig']:
    #     torchsummary.summary(model=mfcvae, input_size=(in_channels, height, width), device='cpu')

    # copy model to correct device
    if 'cuda' in args.device:
        mfcvae = mfcvae.cuda(device=device)

    # weights&biases tracking (gradients, network topology)
    wandb.watch(mfcvae)

    if args.init_type_p_z_c == 'gmm':
        print("Initialize all p(z_j | c_j) with GMM...")
        mfcvae.initialize_p_z_c_params_with_gmm(train_loader, model_type=args.model_type, epoch=-1, batch_idx=-1)  # epoch and batch_idx are not used here
    elif args.init_type_p_z_c == 'glorot':
        print("Initialize all p(z_j | c_j) with Glorot initialisation")
        mfcvae.initialize_p_z_c_params_with_glorot()
    else:
        exit("not implemented")

    # define optimizer and scheduler
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, mfcvae.parameters()), lr=args.init_lr)

    # training and evaluation loop
    epoch_lr = optimizer.param_groups[0]['lr']
    print("Start training and evaluation loop...")

    # overwrite args.n_epochs in case of progressive pretraining
    if args.do_progressive_training:
        args.n_epochs = torch.sum(torch.tensor(args.n_epochs_per_progressive_step)).item()

    for epoch in range(args.n_epochs):
        # do not train in first epoch (get eval metrics right after initialization)
        if epoch == 0:
            train_loss = 0
            train_log_prob_p_x_z, train_log_prob_E_p_z_c, train_log_prob_E_p_c, train_log_prob_E_q_z_x, train_log_prob_E_q_c_x, train_kl_z, train_kl_c = 0, 0, 0, 0, 0, 0, 0
        else:
            # training
            mfcvae.train()  # changes model to train mode (e.g. dropout, batch norm affected)

            train_loss = 0
            train_log_prob_p_x_z, train_log_prob_E_p_z_c, train_log_prob_E_p_c, train_log_prob_E_q_z_x, train_log_prob_E_q_c_x, train_kl_z, train_kl_c = 0, 0, 0, 0, 0, 0, 0

            y_true_list, y_pred_j_list = [], [[] for j in range(mfcvae.J_n_mixtures)]

            # preparing visualization
            vis_recon_logging_list_done = [False if args.do_vis_recon else True for _ in
                                           range(mfcvae.J_n_mixtures)]
            vis_x_recon_logged = {}
            vis_x_hat_recon_logged = {}
            vis_count_recon_per_cluster = {}

            for i in range(mfcvae.J_n_mixtures):  # J clusterings
                vis_x_recon_logged[i], vis_x_hat_recon_logged[i], vis_count_recon_per_cluster[i] = {}, {}, {}
                for j in range(mfcvae.n_clusters_j_list[i]):
                    vis_count_recon_per_cluster[i][j] = 0
                    vis_x_recon_logged[i][j] = []
                    vis_x_hat_recon_logged[i][j] = []

            for batch_idx, (x, y_true) in enumerate(train_loader):
                if args.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae']:
                    x = x.view(x.size(0), -1).float()
                elif args.model_type in ['conv_vlae']:
                    x = x.float()
                optimizer.zero_grad()
                x = Variable(x)

                x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, epoch, batch_idx)
                if args.model_type in ['conv_vlae']:
                    x = x.view(x.size(0), -1)
                    x_hat = x_hat.view(x_hat.size(0), -1).float()
                loss, mean_log_prob_p_x_z, mean_log_prob_E_p_z_c, mean_log_prob_E_p_c, mean_log_prob_E_q_z_x, mean_log_prob_E_q_c_x, kl_z, kl_c = mfcvae.compute_loss_5terms(x, x_hat, q_z_j_x_list, z_sample_q_z_j_x_list, epoch)

                train_loss += loss.data * len(x)
                train_log_prob_p_x_z += mean_log_prob_p_x_z * len(x)
                train_log_prob_E_p_z_c += mean_log_prob_E_p_z_c * len(x)
                train_log_prob_E_p_c += mean_log_prob_E_p_c * len(x)
                train_log_prob_E_q_z_x += mean_log_prob_E_q_z_x * len(x)
                train_log_prob_E_q_c_x += mean_log_prob_E_q_c_x * len(x)
                train_kl_z += kl_z * len(x)
                train_kl_c += kl_c * len(x)

                loss.backward()

                # reshape back to image shapes
                x_batch = torch.clone(x)
                x = x.view(-1, in_channels, height, width).permute(0, 2, 3, 1)
                x_hat = x_hat.view(-1, in_channels, height, width).permute(0, 2, 3, 1)

                optimizer.step()

                # compute q(c | x)
                prob_p_c_j_z_j_list = mfcvae.compute_q_c_j_x(z_sample_q_z_j_x_list)

                y_true = y_true.cpu().numpy()
                y_true_list.append(y_true)

                for j in range(mfcvae.J_n_mixtures):
                    prob_p_c_j_z_j_list[j] = prob_p_c_j_z_j_list[j].data.cpu().numpy()

                    y_pred_j = np.argmax(prob_p_c_j_z_j_list[j], axis=1)
                    y_pred_j_list[j].append(y_pred_j)

                # visualisation on training set
                if args.do_vis_train and epoch % args.vis_every_n_epochs == 0:
                    # in the 'Gaussian' likelihood model case, x_hat can be outside of [0, 1] -> clamping
                    x_hat = torch.clamp(x_hat, min=1e-10, max=1 - (1e-10))
                    y_pred_list_vis = y_pred_j_list
                    # TODO change i->j; remove usage of j anywhere below
                    for i in range(mfcvae.J_n_mixtures):
                        # only take the last y_pred appended (this is considered to be logged)
                        y_pred = y_pred_list_vis[i][-1]
                        for j in range(mfcvae.n_clusters_j_list[i]):
                            indices_orig = np.where(y_pred == j)[0]
                            count_indices = indices_orig.shape[0]

                            # recon
                            if not vis_recon_logging_list_done[i]:
                                indices = np.copy(indices_orig).tolist()
                                n_taken_recon_and_input = min(count_indices,
                                                              args.n_recons_per_cluster -
                                                              vis_count_recon_per_cluster[i][j])
                                if n_taken_recon_and_input > 0:
                                    indices = indices[:n_taken_recon_and_input]
                                    # logging a single plot
                                    vis_x_recon_logged[i][j] = vis_x_recon_logged[i][j] + [
                                        (x[k].cpu().detach().permute(2, 0, 1),
                                         'input_facet_' + str(i) + '_pred_' + str(j)) for k
                                        in indices]
                                    vis_x_hat_recon_logged[i][j] = vis_x_hat_recon_logged[i][j] + [
                                        (x_hat[k].cpu().detach().permute(2, 0, 1),
                                         'recon_facet_' + str(i) + '_pred_' + str(j))
                                        for k in indices]
                                    vis_count_recon_per_cluster[i][j] += n_taken_recon_and_input

                        # log recon
                        if not vis_recon_logging_list_done[i] and ((np.sum(
                                [vis_count_recon_per_cluster[i][j] for j in
                                 range(mfcvae.n_clusters_j_list[i])]) ==
                                                                    mfcvae.n_clusters_j_list[
                                                                        i] * args.n_recons_per_cluster) or (
                                                                           batch_idx == len(train_loader) - 1)):
                            vis_recon_logging_list_done[i] = True

            # visualise all parameters pi_p_c_j via bar charts on wandb
            if args.do_vis_train and args.do_vis_pi and epoch % args.vis_every_n_epochs == 0:
                pi_p_c_j_list = mfcvae.pi_p_c_j_list
                for j in range(mfcvae.J_n_mixtures):
                    fig = plot_pi(pi_p_c_i=pi_p_c_j_list[j].detach().cpu().numpy())
                    wandb.log(
                        {"train_vis/bar chart for parameter pi of p(c_%d) in facet %d" % (j, j): wandb.Image(plt)},
                        step=epoch)
                    plt.close(fig=fig)

            # visualise cluster examples
            if args.do_vis_train and epoch % args.vis_every_n_epochs == 0 and args.do_vis_examples_per_cluster:
                print("Creating cluster examples plot...")
                fig_list, row_indices, column_indices, num_nonempty_clusters = plot_top10_cluster_examples(
                    train_data, mfcvae, args, epoch)
                for j in range(mfcvae.J_n_mixtures):
                    fig = fig_list[j]
                    wandb.log({"train_vis/cluster examples facet %d (rows sorted by average confidence)" % (j): wandb.Image(fig)}, step=epoch)
                    plt.close(fig=fig)  # close the figure

        if epoch != 0:
            y_true_all = np.concatenate(y_true_list)
            y_pred_j_cat_list = []
            train_acc_j_l_list, conf_mat_j_l_list = [[] for j in range(mfcvae.J_n_mixtures)], [[] for j in range(
                mfcvae.J_n_mixtures)]
            y_pred_j_count_list, y_pred_j_count_dict_list = [], []
            facet_to_label_index = {}
            train_acc_j_max_list = []
            train_acc_l_j_list = [[] for l in range(n_labels)]
            train_acc_l_max_list = []
            for j in range(mfcvae.J_n_mixtures):
                y_pred_j_cat = np.concatenate(y_pred_j_list[j])
                y_pred_j_cat_list.append(y_pred_j_cat)
                y_pred_j_unique, y_pred_j_count = np.unique(y_pred_j_cat, return_counts=True)
                y_pred_j_count_list.append(y_pred_j_count)
                y_pred_j_count_dict = dict(zip(y_pred_j_unique, y_pred_j_count))
                y_pred_j_count_dict_list.append(y_pred_j_count_dict)
                for l in range(n_labels):
                    if n_labels > 1:
                        y_true = y_true_all[:, l].astype(int)
                    else:
                        y_true = y_true_all.astype(int)
                    train_acc_j_l, conf_mat_j_l, _ = cluster_acc_and_conf_mat(y_true, y_pred_j_cat)
                    train_acc_j_l_list[j].append(train_acc_j_l)
                    conf_mat_j_l_list[j].append(conf_mat_j_l)
                    train_acc_l_j_list[l].append(train_acc_j_l)
                j_facet_max_label_index = np.argmax(train_acc_j_l_list[j]).item()
                facet_to_label_index[j] = j_facet_max_label_index
                train_acc_j_max = max(train_acc_j_l_list[j])
                train_acc_j_max_list.append(train_acc_j_max)
            train_facet_acc_max = np.max(train_acc_j_max_list).item()
            train_facet_acc_sum = np.sum(train_acc_j_max_list).item()
            train_facet_acc_mean = np.mean(train_acc_j_max_list).item()
            for l in range(n_labels):
                train_acc_l_max = max(train_acc_l_j_list[l])
                train_acc_l_max_list.append(train_acc_l_max)
            train_label_acc_max = np.max(train_acc_l_max_list).item()
            train_label_acc_sum = np.sum(train_acc_l_max_list).item()
            train_label_acc_mean = np.mean(train_acc_l_max_list).item()

            train_weighted_acc_j_max_list = []
            if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                train_weighted_acc_j_l_list = [[] for j in range(mfcvae.J_n_mixtures)]
                for j in range(mfcvae.J_n_mixtures):
                    for l in range(n_labels):
                        train_weighted_acc_j_l = cluster_acc_weighted(conf_mat_j_l_list[j][l])
                        train_weighted_acc_j_l_list[j].append(train_weighted_acc_j_l)
                    train_weighted_acc_j_max = max(train_weighted_acc_j_l_list[j])
                    train_weighted_acc_j_max_list.append(train_weighted_acc_j_max)
                train_weighted_acc_max = np.max(train_weighted_acc_j_max_list).item()
                train_weighted_acc_sum = np.sum(train_weighted_acc_j_max_list).item()
                train_weighted_acc_mean = np.mean(train_weighted_acc_j_max_list).item()

            # plot facet to label dictionary
            if n_labels > 1:
                # create facet to label dictionary
                facet_to_label = {}
                for j in range(mfcvae.J_n_mixtures):
                    facet_to_label[j] = chosen_attr_list[facet_to_label_index[j]]
                # plot it
                fig = plot_dict(facet_to_label)
                wandb.log({"train_vis/Facet to Label mapping": wandb.Image(plt)}, step=epoch)
                plt.close(fig=fig)  # close the figure
                # log confusion matrix on wandb
                if args.do_vis_train and args.do_vis_conf_mat and epoch % args.vis_every_n_epochs == 0:
                    for j in range(mfcvae.J_n_mixtures):
                        fig = plot_confusion_matrix(conf_mat=conf_mat_j_l_list[j][facet_to_label_index[j]],
                                                    n_true_classes=n_true_classes[facet_to_label_index[j]])
                        wandb.log({"train_vis/confusion matrix facet %d" % (j): wandb.Image(plt)}, step=epoch)
                        plt.close(fig=fig)  # close the figure
            else:
                # log confusion matrix on wandb
                if args.do_vis_train and args.do_vis_conf_mat and epoch % args.vis_every_n_epochs == 0:
                    for j in range(mfcvae.J_n_mixtures):
                        fig = plot_confusion_matrix(conf_mat=conf_mat_j_l_list[j][facet_to_label_index[j]],
                                                    n_true_classes=n_true_classes)
                        wandb.log({"train_vis/confusion matrix facet %d" % (j): wandb.Image(plt)}, step=epoch)
                        plt.close(fig=fig)  # close the figure
            # visualise the number of samples per cluster on wandb
            if args.do_vis_train and args.do_vis_n_samples_per_cluster and epoch % args.vis_every_n_epochs == 0:
                for j in range(mfcvae.J_n_mixtures):
                    fig = plot_n_inputs_per_cluster(y_pred_count=y_pred_j_count_list[j],
                                                    n_clusters=mfcvae.n_clusters_j_list[j])
                    wandb.log({
                                  "train_vis/rug plot and empirical cdf plot of number of inputs per predicted cluster for facet %d" % (
                                      j): wandb.Image(plt)}, step=epoch)
                    plt.close(fig=fig)  # close the figure


            # visualise reconstruction examples
            if args.do_vis_train and epoch % args.vis_every_n_epochs == 0 and args.do_vis_recon:
                vis_x_recon_logged_sorted, vis_x_hat_recon_logged_sorted, vis_count_recon_per_cluster_sorted = {}, {}, {}
                for i in range(mfcvae.J_n_mixtures):
                    vis_x_recon_logged_sorted[i], vis_x_hat_recon_logged_sorted[i], \
                    vis_count_recon_per_cluster_sorted[i] = {}, {}, {}
                    y_pred_j_count_descend_index = np.argsort(-y_pred_j_count_list[i])
                    for j0 in range(len(y_pred_j_count_descend_index)):
                        j = list(y_pred_j_count_dict_list[i].keys())[y_pred_j_count_descend_index[j0]]
                        vis_x_recon_logged_sorted[i][j0] = vis_x_recon_logged[i][j]
                        vis_x_hat_recon_logged_sorted[i][j0] = vis_x_hat_recon_logged[i][j]
                        vis_count_recon_per_cluster_sorted[i][j0] = vis_count_recon_per_cluster[i][j]
                    fig = plot_inputs_and_recons_torch_grid(inputs_dict=vis_x_recon_logged_sorted[i],
                                                                recon_dict=vis_x_hat_recon_logged_sorted[i],
                                                                count_dict=vis_count_recon_per_cluster_sorted[i],
                                                                # n_clusters=mfcvae.n_clusters_j_list[i],
                                                                n_clusters=len(y_pred_j_count_descend_index),
                                                                n_pairs_per_cluster=args.n_recons_per_cluster)
                    wandb.log({"train_vis/input and reconstruction facet %d (sorted, cluster size from large to small)" % (i): wandb.Image(plt)},
                                  step=epoch)
                    plt.close(fig=fig)  # close the figure

            # log and print epoch results
            epoch_lr = optimizer.param_groups[0]['lr']  # current learning rate
            train_loss = train_loss / len(train_loader.dataset)
            train_log_prob_p_x_z = train_log_prob_p_x_z / len(train_loader.dataset)
            train_log_prob_E_p_z_c = train_log_prob_E_p_z_c / len(train_loader.dataset)
            train_log_prob_E_p_c = train_log_prob_E_p_c / len(train_loader.dataset)
            train_log_prob_E_q_z_x = train_log_prob_E_q_z_x / len(train_loader.dataset)
            train_log_prob_E_q_c_x = train_log_prob_E_q_c_x / len(train_loader.dataset)
            train_kl_z = train_kl_z / len(train_loader.dataset)
            train_kl_c = train_kl_c / len(train_loader.dataset)
            print_string = "#Epoch %3d: lr: %.5f, Train Loss: %.4f, " % (epoch, epoch_lr, train_loss)
            for j in range(mfcvae.J_n_mixtures):
                print_string += "Train Acc max facet %d: %.4f, " % (j, train_acc_j_max_list[j])
            for l in range(n_labels):
                print_string += "Train Acc max label %d: %.4f, " % (l, train_acc_l_max_list[l])
            if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                print_string += "|, "
                for j in range(mfcvae.J_n_mixtures):
                    print_string += "Train Weighted Acc max facet %d: %.4f, " % (j, train_weighted_acc_j_max_list[j])
            print(print_string)

            if mfcvae.do_progressive_training:
                print("progr. training - alpha_enc_fade_in_list: ", mfcvae.alpha_enc_fade_in_list, "alpha_dec_fade_in_list: ", mfcvae.alpha_dec_fade_in_list, "gamma_kl_z_list: ", mfcvae.gamma_kl_z_list, "gamma_kl_c_list: ", mfcvae.gamma_kl_c_list)

            log_dict = {'epoch': epoch, 'epoch_lr': epoch_lr}  # train_loss updated below
            log_dict['train_metric/loss'] = train_loss
            log_dict['train_loss/loss'] = train_loss
            log_dict['train_loss/log_prob_p_x_z'] = train_log_prob_p_x_z
            log_dict['train_loss/log_prob_E_p_z_c'] = train_log_prob_E_p_z_c
            log_dict['train_loss/log_prob_E_p_c'] = train_log_prob_E_p_c
            log_dict['train_loss/log_prob_E_q_z_x'] = train_log_prob_E_q_z_x
            log_dict['train_loss/log_prob_E_q_c_x'] = train_log_prob_E_q_c_x
            log_dict['train_loss/kl_z'] = train_kl_z
            log_dict['train_loss/kl_c'] = train_kl_c
            for j in range(mfcvae.J_n_mixtures):
                log_dict['train_metric/facet_%d_acc_max' % j] = train_acc_j_max_list[j]
                if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                    log_dict['train_metric/facet_%d_weighted_acc_max' % j] = train_weighted_acc_j_max_list[j]
                log_dict['train_metric/n_clusters_present_%d' % j] = len(y_pred_j_count_dict_list[j])
            log_dict['train_metric/facet_acc_max'] = train_facet_acc_max
            log_dict['train_metric/facet_acc_sum'] = train_facet_acc_sum
            log_dict['train_metric/facet_acc_mean'] = train_facet_acc_mean
            if n_labels > 1:
                for l in range(n_labels):
                    log_dict['train_metric/label_%d_acc_max' % l] = train_acc_l_max_list[l]
                log_dict['train_metric/label_acc_max'] = train_label_acc_max
                log_dict['train_metric/label_acc_sum'] = train_label_acc_sum
                log_dict['train_metric/label_acc_mean'] = train_label_acc_mean
            if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                log_dict['train_metric/facet_weighted_acc_max'] = train_weighted_acc_max
                log_dict['train_metric/facet_weighted_acc_sum'] = train_weighted_acc_sum
                log_dict['train_metric/facet_weighted_acc_mean'] = train_weighted_acc_mean
            wandb.log(log_dict, step=epoch)
        else:
            epoch_lr = optimizer.param_groups[0]['lr']  # current learning rate
            log_dict = {'epoch': epoch, 'epoch_lr': epoch_lr}
            print("#Epoch %3d: lr: %.5f, " % (epoch, epoch_lr))
            wandb.log(log_dict, step=epoch)

        # evaluation on the test set
        if args.do_test_during_training or (not args.do_test_during_training and epoch == args.n_epochs - 1):
            with torch.no_grad():
                mfcvae.eval()  # changes model to evaluation mode (e.g. dropout, batch norm affected)
                eval_loss = 0.
                y_true_list, y_pred_j_list = [], [[] for j in range(mfcvae.J_n_mixtures)]

                # preparing visualization
                vis_recon_logging_list_done = [False if args.do_vis_recon else True for _ in
                                               range(mfcvae.J_n_mixtures)]
                vis_x_recon_logged = {}
                vis_x_hat_recon_logged = {}
                vis_count_recon_per_cluster = {}

                vis_examples_per_cluster_logging_list_done = [False if args.do_vis_examples_per_cluster else True for _
                                                              in range(mfcvae.J_n_mixtures)]
                vis_examples_per_cluster_logged = {}
                vis_count_examples_per_cluster = {}

                # vis_sample_generations_per_cluster_done = False

                for i in range(mfcvae.J_n_mixtures):  # J clusterings
                    vis_x_recon_logged[i], vis_x_hat_recon_logged[i], vis_count_recon_per_cluster[i], \
                    vis_examples_per_cluster_logged[i], vis_count_examples_per_cluster[i] = {}, {}, {}, {}, {}
                    for j in range(mfcvae.n_clusters_j_list[i]):
                        vis_count_recon_per_cluster[i][j] = 0
                        vis_examples_per_cluster_logged[i][j] = []

                        vis_count_examples_per_cluster[i][j] = 0
                        vis_x_recon_logged[i][j] = []
                        vis_x_hat_recon_logged[i][j] = []

                # preparing visualization end
                for batch_idx, (x, y_true) in enumerate(test_loader):
                    # subselect y_true with chosen attributes in the case of celeba
                    if batch_idx == args.n_test_batches:
                        break
                    if args.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae']:
                        x = x.view(x.size(0), -1).float()
                    elif args.model_type in ['conv_vlae']:
                        x = x.float()
                    x = Variable(x)

                    x_hat, q_z_j_x_list, z_sample_q_z_j_x_list = mfcvae.forward(x, epoch, batch_idx)
                    if args.model_type in ['conv_vlae']:
                        x = x.view(x.size(0), -1)
                        x_hat = x_hat.view(x_hat.size(0), -1).float()
                    loss, _, _, _, _, _, _, _ = mfcvae.compute_loss_5terms(x, x_hat, q_z_j_x_list, z_sample_q_z_j_x_list, epoch)
                    # reshape back to image shapes
                    x = x.view(-1, in_channels, height, width).permute(0, 2, 3, 1)
                    x_hat = x_hat.view(-1, in_channels, height, width).permute(0, 2, 3, 1)

                    eval_loss += loss.data * len(x)

                    prob_p_c_j_z_j_list = mfcvae.compute_q_c_j_x(z_sample_q_z_j_x_list)

                    y_true = y_true.cpu().numpy()
                    y_true_list.append(y_true)

                    for j in range(mfcvae.J_n_mixtures):
                        prob_p_c_j_z_j_list[j] = prob_p_c_j_z_j_list[j].data.cpu().numpy()

                        y_pred_j = np.argmax(prob_p_c_j_z_j_list[j], axis=1)
                        y_pred_j_list[j].append(y_pred_j)

                    if args.do_vis_test and epoch % args.vis_every_n_epochs == 0:
                        # in the 'Gaussian' likelihood model case, x_hat can be outside of [0, 1] -> clamping
                        x_hat = torch.clamp(x_hat, min=1e-10, max=1 - (1e-10))
                        y_pred_list_vis = y_pred_j_list
                        # TODO change i->j; remove usage of j anywhere below
                        for i in range(mfcvae.J_n_mixtures):
                            # only take the last y_pred appended (this is considered to be logged)
                            y_pred = y_pred_list_vis[i][-1]
                            for j in range(mfcvae.n_clusters_j_list[i]):
                                indices_orig = np.where(y_pred == j)[0]
                                count_indices = indices_orig.shape[0]

                                # recon
                                if not vis_recon_logging_list_done[i]:
                                    indices = np.copy(indices_orig).tolist()
                                    n_taken_recon_and_input = min(count_indices, args.n_recons_per_cluster -
                                                                  vis_count_recon_per_cluster[i][j])
                                    if n_taken_recon_and_input > 0:
                                        indices = indices[:n_taken_recon_and_input]
                                        vis_x_recon_logged[i][j] = vis_x_recon_logged[i][j] + [(x[k].cpu().detach().permute(
                                            2, 0, 1), 'input_facet_' + str(i) + '_pred_' + str(j)) for k in indices]
                                        vis_x_hat_recon_logged[i][j] = vis_x_hat_recon_logged[i][j] + [(x_hat[k].cpu().detach().permute(
                                            2, 0, 1), 'recon_facet_' + str(i) + '_pred_' + str(j)) for k in indices]
                                        vis_count_recon_per_cluster[i][j] += n_taken_recon_and_input

                            # log recon
                            if not vis_recon_logging_list_done[i] and ((np.sum(
                                    [vis_count_recon_per_cluster[i][j] for j in
                                     range(mfcvae.n_clusters_j_list[i])]) == mfcvae.n_clusters_j_list[i] * args.n_recons_per_cluster) or (batch_idx == len(test_loader) - 1)):
                                vis_recon_logging_list_done[i] = True

                y_true_all = np.concatenate(y_true_list)
                y_pred_j_cat_list = []
                eval_acc_j_l_list, conf_mat_j_l_list = [[] for j in range(mfcvae.J_n_mixtures)], [[] for j in range(mfcvae.J_n_mixtures)]
                y_pred_j_count_list, y_pred_j_count_dict_list = [], []
                facet_to_label_index = {}
                eval_acc_j_max_list = []
                eval_acc_l_j_list = [[] for l in range(n_labels)]
                eval_acc_l_max_list = []
                for j in range(mfcvae.J_n_mixtures):
                    y_pred_j_cat = np.concatenate(y_pred_j_list[j])
                    y_pred_j_cat_list.append(y_pred_j_cat)
                    y_pred_j_unique, y_pred_j_count = np.unique(y_pred_j_cat, return_counts=True)
                    y_pred_j_count_list.append(y_pred_j_count)
                    y_pred_j_count_dict = dict(zip(y_pred_j_unique, y_pred_j_count))
                    y_pred_j_count_dict_list.append(y_pred_j_count_dict)
                    for l in range(n_labels):
                        if n_labels > 1:
                            y_true = y_true_all[:, l].astype(int)
                        else:
                            y_true = y_true_all.astype(int)
                        eval_acc_j_l, conf_mat_j_l, _ = cluster_acc_and_conf_mat(y_true, y_pred_j_cat)
                        eval_acc_j_l_list[j].append(eval_acc_j_l)
                        conf_mat_j_l_list[j].append(conf_mat_j_l)
                        eval_acc_l_j_list[l].append(eval_acc_j_l)
                    j_facet_max_label_index = np.argmax(eval_acc_j_l_list[j]).item()
                    facet_to_label_index[j] = j_facet_max_label_index
                    eval_acc_j_max = max(eval_acc_j_l_list[j])
                    eval_acc_j_max_list.append(eval_acc_j_max)
                eval_facet_acc_max = np.max(eval_acc_j_max_list).item()
                eval_facet_acc_sum = np.sum(eval_acc_j_max_list).item()
                eval_facet_acc_mean = np.mean(eval_acc_j_max_list).item()
                for l in range(n_labels):
                    eval_acc_l_max = max(eval_acc_l_j_list[l])
                    eval_acc_l_max_list.append(eval_acc_l_max)
                eval_label_acc_max = np.max(eval_acc_l_max_list).item()
                eval_label_acc_sum = np.sum(eval_acc_l_max_list).item()
                eval_label_acc_mean = np.mean(eval_acc_l_max_list).item()

                eval_weighted_acc_j_max_list = []
                if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                    eval_weighted_acc_j_l_list = [[] for j in range(mfcvae.J_n_mixtures)]
                    for j in range(mfcvae.J_n_mixtures):
                        for l in range(n_labels):
                            eval_weighted_acc_j_l = cluster_acc_weighted(conf_mat_j_l_list[j][l])
                            eval_weighted_acc_j_l_list[j].append(eval_weighted_acc_j_l)
                        eval_weighted_acc_j_max = max(eval_weighted_acc_j_l_list[j])
                        eval_weighted_acc_j_max_list.append(eval_weighted_acc_j_max)
                    eval_weighted_acc_max = np.max(eval_weighted_acc_j_max_list).item()
                    eval_weighted_acc_sum = np.sum(eval_weighted_acc_j_max_list).item()
                    eval_weighted_acc_mean = np.mean(eval_weighted_acc_j_max_list).item()

                # plot facet to label dictionary
                if n_labels > 1:
                    # create facet to label dictionary
                    facet_to_label = {}
                    for j in range(mfcvae.J_n_mixtures):
                        facet_to_label[j] = chosen_attr_list[facet_to_label_index[j]]
                    # plot it
                    fig = plot_dict(facet_to_label)
                    wandb.log({"test_vis/Facet to Label mapping": wandb.Image(plt)}, step=epoch)
                    plt.close(fig=fig)  # close the figure
                    # log confusion matrix on wandb
                    if args.do_vis_test and args.do_vis_conf_mat and epoch % args.vis_every_n_epochs == 0:
                        for j in range(mfcvae.J_n_mixtures):
                            fig = plot_confusion_matrix(conf_mat=conf_mat_j_l_list[j][facet_to_label_index[j]],
                                                        n_true_classes=n_true_classes[facet_to_label_index[j]])
                            wandb.log({"test_vis/confusion matrix facet %d" % (j): wandb.Image(plt)}, step=epoch)
                            plt.close(fig=fig)  # close the figure
                else:
                    # log confusion matrix on wandb
                    if args.do_vis_test and args.do_vis_conf_mat and epoch % args.vis_every_n_epochs == 0:
                        for j in range(mfcvae.J_n_mixtures):
                            fig = plot_confusion_matrix(conf_mat=conf_mat_j_l_list[j][facet_to_label_index[j]],
                                                        n_true_classes=n_true_classes)
                            wandb.log({"test_vis/confusion matrix facet %d" % (j): wandb.Image(plt)}, step=epoch)
                            plt.close(fig=fig)  # close the figure

                # visualise the number of samples per cluster on wandb
                if args.do_vis_test and args.do_vis_n_samples_per_cluster and epoch % args.vis_every_n_epochs == 0:
                    for j in range(mfcvae.J_n_mixtures):
                        fig = plot_n_inputs_per_cluster(y_pred_count=y_pred_j_count_list[j],
                                                        n_clusters=mfcvae.n_clusters_j_list[j])
                        wandb.log({
                                      "test_vis/rug plot and empirical cdf plot of number of inputs per predicted cluster for facet %d" % (
                                          j): wandb.Image(plt)}, step=epoch)
                        plt.close(fig=fig)  # close the figure


                # visualise cluster examples and sample generations
                if args.do_vis_test and epoch % args.vis_every_n_epochs == 0 and (args.do_vis_examples_per_cluster or args.do_vis_sample_generations_per_cluster):
                    print("Creating cluster examples and sample generation plots...")
                    fig_list, row_indices, column_indices, num_nonempty_clusters = plot_top10_cluster_examples(
                        test_data, mfcvae, args, epoch)
                    if args.do_vis_examples_per_cluster:
                        for j in range(mfcvae.J_n_mixtures):
                            fig = fig_list[j]
                            wandb.log({"test_vis/cluster examples facet %d (rows sorted by average confidence)" % (
                                j): wandb.Image(fig)}, step=epoch)
                            plt.close(fig=fig)  # close the figure
                    if args.do_vis_sample_generations_per_cluster:
                        fig_list = plot_sample_generation(row_indices, num_nonempty_clusters, mfcvae, args,
                                                          args.temp_sample_generation)
                        for j in range(mfcvae.J_n_mixtures):
                            fig = fig_list[j]
                            wandb.log({"sample_generation/cluster sample generations facet %d (rows sorted by average confidence)" % (
                                              j): wandb.Image(fig)}, step=epoch)
                            plt.close(fig=fig)  # close the figure

                # visualise reconstruction examples
                if args.do_vis_test and epoch % args.vis_every_n_epochs == 0 and args.do_vis_recon:
                    vis_x_recon_logged_sorted, vis_x_hat_recon_logged_sorted, vis_count_recon_per_cluster_sorted = {}, {}, {}
                    for i in range(mfcvae.J_n_mixtures):
                        vis_x_recon_logged_sorted[i], vis_x_hat_recon_logged_sorted[i], vis_count_recon_per_cluster_sorted[i] = {}, {}, {}
                        y_pred_j_count_descend_index = np.argsort(-y_pred_j_count_list[i])
                        for j0 in range(len(y_pred_j_count_descend_index)):
                            j = list(y_pred_j_count_dict_list[i].keys())[y_pred_j_count_descend_index[j0]]
                            vis_x_recon_logged_sorted[i][j0] = vis_x_recon_logged[i][j]
                            vis_x_hat_recon_logged_sorted[i][j0] = vis_x_hat_recon_logged[i][j]
                            vis_count_recon_per_cluster_sorted[i][j0] = vis_count_recon_per_cluster[i][j]
                        fig = plot_inputs_and_recons_torch_grid(inputs_dict=vis_x_recon_logged_sorted[i],
                                                                    recon_dict=vis_x_hat_recon_logged_sorted[i],
                                                                    count_dict=vis_count_recon_per_cluster_sorted[i],
                                                                    # n_clusters=mfcvae.n_clusters_j_list[i],
                                                                    n_clusters=len(y_pred_j_count_descend_index),
                                                                    n_pairs_per_cluster=args.n_recons_per_cluster)
                        wandb.log({"test_vis/input and reconstruction facet %d (sorted, cluster size from large to small)" % (i): wandb.Image(plt)},
                                      step=epoch)
                        plt.close(fig=fig)  # close the figure

                eval_loss = eval_loss / len(test_loader.dataset)

                print_string = "Eval Loss: %.4f, " % (eval_loss)
                for j in range(mfcvae.J_n_mixtures):
                    print_string += "Eval Acc max facet %d: %.4f, " % (j, eval_acc_j_max_list[j])
                for l in range(n_labels):
                    print_string += "Eval Acc max label %d: %.4f, " % (l, eval_acc_l_max_list[l])
                if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                    print_string += "|, "
                    for j in range(mfcvae.J_n_mixtures):
                        print_string += "Eval Weighted Acc max facet %d: %.4f, " % (j, eval_weighted_acc_j_max_list[j])
                print(print_string)

                log_dict = {'test_metric/loss': eval_loss}
                for j in range(mfcvae.J_n_mixtures):
                    log_dict['test_metric/facet_%d_acc_max' % j] = eval_acc_j_max_list[j]
                    if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                        log_dict['test_metric/facet_%d_weighted_acc_max_' % j] = eval_weighted_acc_j_max_list[j]
                    log_dict['test_metric/clusters_present_%d' % j] = len(y_pred_j_count_dict_list[j])
                log_dict['test_metric/facet_acc_max'] = eval_facet_acc_max
                log_dict['test_metric/facet_acc_sum'] = eval_facet_acc_sum
                log_dict['test_metric/facet_acc_mean'] = eval_facet_acc_mean
                if n_labels > 1:
                    for l in range(n_labels):
                        log_dict['test_metric/label_%d_acc_max' % l] = eval_acc_l_max_list[l]
                    log_dict['test_metric/label_acc_max'] = eval_label_acc_max
                    log_dict['test_metric/label_acc_sum'] = eval_label_acc_sum
                    log_dict['test_metric/label_acc_mean'] = eval_label_acc_mean
                if args.dataset in ['fast_svhn', 'fast_3dshapes']:
                    log_dict['test_metric/facet_weighted_acc_max'] = eval_weighted_acc_max
                    log_dict['test_metric/facet_weighted_acc_sum'] = eval_weighted_acc_sum
                    log_dict['test_metric/facet_weighted_acc_mean'] = eval_weighted_acc_mean

                wandb.log(log_dict, step=epoch)
    # save model
    if args.save_model:
        save_dict_path = os.path.join(wandb.run.dir, "save_dict.pt")
        save_dict = {'state_dict': mfcvae.state_dict(),
                     'args': args}  # args dictionary is already part of saving the model
        torch.save(save_dict, save_dict_path)

    # save args config dictionary
    save_args_path = os.path.join(wandb.run.dir, "args.pickle")  # args_dict.py
    with open(save_args_path, 'wb') as file:
        pickle.dump(args, file, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    train()
