
import torch
import torch.distributions as D
import torch.nn as nn
from torch.distributions.utils import logits_to_probs, probs_to_logits
import math
import numpy as np
from sklearn.mixture import GaussianMixture
import abc

from utils import softplus_inverse_numpy  #  build_fc_network, build_cnn_network, softplus_inverse,
from conv_vlae import CONVvlaeEncoderCelebA, CONVvlaeDecoderCelebA
from models_fc import FCsharedEncoder, FCSharedDecoder, FCseparateEncoders, FCvlaeEncoder, FCvlaeDecoder
from typing import List


class MFCVAE(nn.Module):
    def __init__(self, in_dim: int, J_n_mixtures: int, z_j_dim_list: List[int], n_clusters_j_list: List[int],
                 encode_layer_dims: List[int], decode_layer_dims: List[int],
                 device: str, n_train_batches_per_epoch: int, n_epochs_per_progressive_step: List[int],
                 model_type: str = 'fc_vlae', likelihood_model: str = 'Bernoulli', activation_x_hat_z: str = "sigmoid",
                 sigma_multiplier_p_x_z: float = 0.6, cov_type_p_z_c: str = 'diag',
                 init_type_p_z_c: str = 'gmm', init_off_diag_cov_p_z_c: bool = False,
                 fix_pi_p_c: bool = False, facet_to_fix_pi_p_c: str = "all",
                 n_batches_fade_in: int = 15000, gamma_kl_z_pretrain: float = 0.0, gamma_kl_c_pretrain: float = 0.0,
                 do_progressive_training: bool = True, fixed_var_init: float = 0.01,
                 activation: str = 'relu', do_fc_batch_norm: bool = False):
        """
        Initialize MFCVAE model.

        Args:
           in_dim: Integer indicating dimension of input variable x.
           J_n_mixtures: The number of mixtures/clusterings.
           z_j_dim_list: List of Integers where the j-th integer indicates dimension of latent space variable z_j.
           n_clusters_j_list: List of Integers where the j-th integer indicates the number of clusters in clustering j (possible realisations of c_0j).
           encode_layer_dims: List of integers indicating the number of output units of each encoding layer (except z).
           decode_layer_dims: List of integers indicating the number of output units of each decoding layer (except x_hat).
           device: torch.device object.
           model_type: Type of model to be initialised (mostly defining the encoder and decoder architecture).
           n_train_batches_per_epoch: Number of training batches per epoch.
           n_epochs_per_progressive_step: List of nubmer of epochs per progressive step.
           likelihood_model: The likelihood function for p(x | z). 'Bernoulli' or 'Gaussian'.
           activation_x_hat_z: Indicates whether to apply an activation function to the "output" (x_hat_z) of the generative model ("sigmoid") or not (None).
           sigma_multiplier_p_x_z: Fixed sigma parameter of variance of p(x|z).
           cov_type_p_z_c: Whether each p_z_c has a diagonal covariance structure ('diag') or a full covariance structure ('full').
           init_type_p_z_c: How to initialize p(z|c).
           init_off_diag_cov_p_z_c: How to initialize the off-diagonal elements of the covariance matrices of p(z_j | c_j).
           fix_pi_p_c: Whether to fix parameters pi_j in p(c_j) for all j.
           facet_to_fix_pi_p_c: Which facet's pi parameters of p(c_j) to fix.
           n_batches_fade_in: Number of batches during which to perform the fade-in at the beginning of every progessive step.
           gamma_kl_z_pretrain: Value of gamma during pretraining of z.
           gamma_kl_c_pretrain: Value of gamma during pretraining of c.
           do_progressive_training: Whether to do progressive training or not.
           fixed_var_init: Whether to initialise p(z) with fixed variances.
           activation: activation function to use in all fc and conv layers.
           do_fc_batch_norm: whether to have a batch norm layer in the build_fc_network(...) network function or not.
        """
        super(MFCVAE, self).__init__()
        # dimensions
        self.in_dim = in_dim
        self.J_n_mixtures = J_n_mixtures
        self.z_j_dim_list = z_j_dim_list
        self.n_clusters_j_list = n_clusters_j_list
        self.encode_layer_dims = encode_layer_dims
        self.decode_layer_dims = decode_layer_dims
        self.device = device
        self.cov_type_p_z_c = cov_type_p_z_c
        self.init_type_p_z_c = init_type_p_z_c
        self.init_off_diag_cov_p_z_c = init_off_diag_cov_p_z_c
        self.likelihood_model = likelihood_model
        self.activation_x_hat_z = activation_x_hat_z
        self.sigma_multiplier_p_x_z = sigma_multiplier_p_x_z
        self.model_type = model_type
        self.fix_pi_p_c = fix_pi_p_c
        self.facet_to_fix_pi_p_c = facet_to_fix_pi_p_c
        self.n_train_batches_per_epoch = n_train_batches_per_epoch
        self.n_epochs_per_progressive_step = n_epochs_per_progressive_step
        self.n_batches_fade_in = n_batches_fade_in
        self.gamma_kl_z_pretrain = gamma_kl_z_pretrain
        self.gamma_kl_c_pretrain = gamma_kl_c_pretrain
        self.do_progressive_training = do_progressive_training
        self.fixed_var_init = fixed_var_init
        self.activation = activation
        self.do_fc_batch_norm = do_fc_batch_norm

        # parameters
        self._pi_p_c_j_list = torch.nn.ParameterList()
        self.mu_p_z_j_c_j_list = torch.nn.ParameterList()
        if cov_type_p_z_c == 'diag':
            self._sigma_square_p_z_j_c_j_list = torch.nn.ParameterList()
        elif cov_type_p_z_c == 'full':
            self._l_mat_p_z_j_c_j_list = torch.nn.ParameterList()
        for j in range(J_n_mixtures):
            if self.fix_pi_p_c:
                if self.facet_to_fix_pi_p_c == "all":
                    self._pi_p_c_j_list.append(nn.Parameter(torch.ones(n_clusters_j_list[j]) / n_clusters_j_list[j], requires_grad=False))
                elif self.facet_to_fix_pi_p_c == "facet_0" and j == 0:
                    self._pi_p_c_j_list.append(nn.Parameter(torch.ones(n_clusters_j_list[j]) / n_clusters_j_list[j], requires_grad=False))
                elif self.facet_to_fix_pi_p_c == "facet_1" and j == 1:
                    self._pi_p_c_j_list.append(nn.Parameter(torch.ones(n_clusters_j_list[j]) / n_clusters_j_list[j], requires_grad=False))
            else:
                self._pi_p_c_j_list.append(nn.Parameter(torch.ones(n_clusters_j_list[j]) / n_clusters_j_list[j]))
            self.mu_p_z_j_c_j_list.append(nn.Parameter(torch.zeros(z_j_dim_list[j], n_clusters_j_list[j])))
            if cov_type_p_z_c == 'diag':
                # (z_dim, n_clusters)
                self._sigma_square_p_z_j_c_j_list.append(nn.Parameter(torch.ones(z_j_dim_list[j], n_clusters_j_list[j])))
            elif cov_type_p_z_c == 'full':
                ones_j = torch.ones(n_clusters_j_list[j], z_j_dim_list[j])
                eye_mats_j = torch.diag_embed(ones_j)
                eye_mats_j = eye_mats_j.permute(1, 2, 0)  # (z_dim, z_dim, n_clusters)
                self._l_mat_p_z_j_c_j_list.append(nn.Parameter(eye_mats_j))  # covariance matrix of each p(z | c) (full covariance structure), here initialized with identity matrix

        # recognition model   'fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae'
        if self.model_type == 'fc_shared':
            self.encoder = FCsharedEncoder(layer_dims=[in_dim] + encode_layer_dims, J_n_mixtures=self.J_n_mixtures, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)  # most layers of the recognition model
            encoder_output_dims = [encode_layer_dims[-1] for j in range(self.J_n_mixtures)]
        elif self.model_type == 'fc_per_facet_enc_shared_dec':
            layer_dims = [[in_dim] + dims_list for dims_list in encode_layer_dims]
            self.encoder = FCseparateEncoders(layer_dims=layer_dims, J_n_mixtures=self.J_n_mixtures, do_fc_batch_norm=self.do_fc_batch_norm)
            encoder_output_dims = [encode_layer_dims[j][-1] for j in range(self.J_n_mixtures)]
        elif self.model_type == 'fc_vlae':
            # TODO Fix inconsistent use of layer_dims compared to previous two encoders and decoders with in_dim
            self.encoder = FCvlaeEncoder(layer_dims=self.encode_layer_dims, in_dim=self.in_dim, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)
            encoder_output_dims = self.encoder.encoder_output_dims  # [self.encode_layer_dims[j][-1] for j in range(self.J_n_mixtures)]
        elif self.model_type == 'conv_vlae':
            # proVLAE CelebA implementation
            self.encoder = CONVvlaeEncoderCelebA(J_n_mixtures=J_n_mixtures, in_dim=in_dim, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)
            encoder_output_dims = self.encoder.encoder_output_dims
        # only one layer for both z_j
        self.fc_mu_q_z_x_list, self.fc_log_sigma_square_q_z_x_list = nn.ModuleList(), nn.ModuleList()
        for j in range(self.J_n_mixtures):
            self.fc_mu_q_z_x_list.append(nn.Linear(encoder_output_dims[j], z_j_dim_list[j]))  # layer that outputs the mean parameter of q(z | x); no activation function, since on continuous scale
            self.fc_log_sigma_square_q_z_x_list.append(nn.Linear(encoder_output_dims[j], z_j_dim_list[j]))  # layer that outputs the logarithm of the variance parameter of q(z | x) (diagonal covariance structure); no activation function, since on continuous scale

        # generative model
        # self.decoder = build_fc_network([z_0_dim + z_1_dim] + decode_layer_dims)  # most layers of the generative model
        if self.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec']:
            self.decoder = FCSharedDecoder(layer_dims=[sum(z_j_dim_list)] + decode_layer_dims, J_n_mixtures=self.J_n_mixtures, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)
        elif self.model_type == 'fc_vlae':
            # TODO Fix inconsistent use of layer_dims compared to previous two encoders and decoders with in_dim
            self.decoder = FCvlaeDecoder(layer_dims=self.decode_layer_dims, z_j_dim_list=self.z_j_dim_list, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)
        elif self.model_type == 'conv_vlae':
            # proVLAE CelebA implementation
            self.decoder = CONVvlaeDecoderCelebA(J_n_mixtures=J_n_mixtures, in_dim=in_dim, z_j_dim_list=self.z_j_dim_list, activation=self.activation, do_fc_batch_norm=self.do_fc_batch_norm)
        if self.model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec']:
            self.layer_x_hat_z = nn.Linear(decode_layer_dims[-1], in_dim)  # fully-connected layer from decoder layers to x_hat
        elif self.model_type == 'fc_vlae':
            self.layer_x_hat_z = nn.Linear(decode_layer_dims[0][-1], in_dim)  # last output of backbone in facet 0
        elif self.model_type == 'conv_vlae':
            self.layer_x_hat_z = nn.Identity()  # last layer in decoder is already in_dim shape

        if self.activation_x_hat_z == "sigmoid":
            self.act_x_hat_z = nn.Sigmoid()
        else:
            self.act_x_hat_z = None

        # progressive training configs
        # TODO other instance variables to be initialised here with None, also in other classes
        self.encoder.do_progressive_training = do_progressive_training
        self.decoder.do_progressive_training = do_progressive_training


    @property
    def pi_p_c_j_list(self):
        return [torch.softmax(self._pi_p_c_j_list[j], dim=0) for j in range(self.J_n_mixtures)]


    @property
    def sigma_square_p_z_j_c_j_list(self):
        if self.init_type_p_z_c == "gmm":
            return [torch.nn.Softplus(beta=10)(self._sigma_square_p_z_j_c_j_list[j]) for j in range(self.J_n_mixtures)]  # + 1e-8
        elif self.init_type_p_z_c == "glorot":
            return [torch.exp(self._sigma_square_p_z_j_c_j_list[j]) for j in range(self.J_n_mixtures)]


    @property
    def l_mat_p_z_j_c_j_list(self):
        l_mat_p_z_j_c_j_list = []
        if self.init_type_p_z_c == "gmm":
            for j in range(self.J_n_mixtures):
                # only perform softplus on diagonal entries
                l_mat_p_z_j_c_j = self._l_mat_p_z_j_c_j_list[j].clone()
                # get diagonal values back on original diagonal (see initialize_p_z_c_params_with_gmm() implementation in there)
                d = torch.nn.Softplus(beta=10)(torch.diagonal(l_mat_p_z_j_c_j))  # (n_clusters, z_dim)
                d = torch.diag_embed(d).permute(1, 2, 0)
                mask = torch.eye(d.shape[0]).unsqueeze(2).repeat(1, 1, l_mat_p_z_j_c_j.shape[2]).to(device=self.device)
                l_mat_p_z_j_c_j = mask * d + (1 - mask) * l_mat_p_z_j_c_j
                l_mat_p_z_j_c_j_list.append(l_mat_p_z_j_c_j)
        elif self.init_type_p_z_c == "glorot":
            for j in range(self.J_n_mixtures):
                # only perform softplus on diagonal entries
                l_mat_p_z_j_c_j = self._l_mat_p_z_j_c_j_list[j].clone()
                # get diagonal values back on original diagonal (see initialize_p_z_c_params_with_gmm() implementation in there)
                d = torch.exp(torch.diagonal(l_mat_p_z_j_c_j))  # (n_clusters, z_dim)
                d = torch.diag_embed(d).permute(1, 2, 0)
                mask = torch.eye(d.shape[0]).unsqueeze(2).repeat(1, 1, l_mat_p_z_j_c_j.shape[2]).to(device=self.device)
                l_mat_p_z_j_c_j = mask * d + (1 - mask) * l_mat_p_z_j_c_j
                l_mat_p_z_j_c_j_list.append(l_mat_p_z_j_c_j)

        return l_mat_p_z_j_c_j_list



    def forward(self, x: torch.tensor, epoch: int, batch_idx: int):
        """
        Pass x through the MFCVAE network (x -> x_hat).
        Passes x through the encoder, samples from all q(z_j | x) and passes all z_sample through the decoder to obtain reconstructed x.

        Args:
            x: The input of the network, a tensor of dimension (self.in_dim).
            epoch: Current epoch.
            batch_idx: Current batch.

        Returns:
            x_hat: The "autoencoded" output of the network, a tensor of dimension (self.in_dim).
            q_z_j_x_list: List of J Gaussian distribution objects.
            z_sample_q_z_j_x_list: List of J z samples.
        """
        if self.do_progressive_training and self.training:
            self.alpha_enc_fade_in_list, self.alpha_dec_fade_in_list, self.gamma_kl_z_list, self.gamma_kl_c_list = self.compute_progressive_training_coefficients(epoch, batch_idx)
            # also assign alpha lists to encoder and decoder for 'fc_vlae'
            self.encoder.alpha_enc_fade_in_list = self.alpha_enc_fade_in_list
            self.decoder.alpha_dec_fade_in_list = self.alpha_dec_fade_in_list
        elif self.do_progressive_training and not self.training:
            self.alpha_enc_fade_in_list, self.alpha_dec_fade_in_list, self.gamma_kl_z_list, self.gamma_kl_c_list = self.compute_progressive_training_coefficients(epoch, 0)  # do not fade-in yet during evaluation time
            # also assign alpha lists to encoder and decoder for 'fc_vlae'
            self.encoder.alpha_enc_fade_in_list = self.alpha_enc_fade_in_list
            self.decoder.alpha_dec_fade_in_list = self.alpha_dec_fade_in_list

        mu_q_z_j_x_list, log_sigma_square_q_z_j_x_list = self.encode(x)

        # case 2 in table of https://bochang.me/blog/posts/pytorch-distributions/ , shall yield e.g. 128 batch_shape, 10 event_shape.
        q_z_j_x_list = [D.Independent(D.Normal(loc=mu_q_z_j_x_list[j], scale=torch.sqrt(torch.exp(log_sigma_square_q_z_j_x_list[j]))), 1) for j in range(self.J_n_mixtures)]  # do not permute in this case (contrary to the compute_loss_new(...) function)
        if self.training:
            z_sample_q_z_j_x_list = [q_z_j_x_list[j].rsample() for j in range(self.J_n_mixtures)]
        else:
            z_sample_q_z_j_x_list = [mu_q_z_j_x_list[j] for j in range(self.J_n_mixtures)]

        x_hat = self.decode(z_sample_q_z_j_x_list)

        return x_hat, q_z_j_x_list, z_sample_q_z_j_x_list


    def cosine_annealing(self, t: float, t_max: float, min_val: float, max_val: float):
        """
        https://arxiv.org/pdf/1608.03983v5.pdf, equation 5, shifted by pi (to increase rather than decrease)
        """
        return min_val + .5 * (1 - min_val) * (1 + math.cos((t / t_max) * math.pi + math.pi))


    def compute_progressive_training_coefficients(self, epoch: int, batch_idx: int, fade_in_type: float = 'linear'):
        """
        Progressive learning implementation (https://arxiv.org/pdf/2002.10549.pdf). Called in every forward pass

        Args:
            epoch: Current epoch.
            batch_idx: Current training batch within epoch.

        Returns:
            alpha_fade_in_list: alpha coefficients for each facet.
            gamma_kl_list: gamma coefficients for each facet.
        """
        gamma_kl_z_list, gamma_kl_c_list = [], []
        alpha_enc_fade_in_list, alpha_dec_fade_in_list = [], []

        n_epochs_per_progressive_step_cum = np.cumsum(self.n_epochs_per_progressive_step).tolist()
        n_epochs_per_progressive_step_cum_start = [0] + n_epochs_per_progressive_step_cum
        s_progressive_step = next(index for index, e in enumerate(n_epochs_per_progressive_step_cum) if epoch < e)  # first time fulfilling condition looping through list
        n_batches_since_progressive_step_start = batch_idx + self.n_train_batches_per_epoch * (epoch - n_epochs_per_progressive_step_cum_start[s_progressive_step])
        if s_progressive_step == 0:
            alpha_fade_in = 1.  # fully fade in from beginning
            gamma_kl_z_fade_in = 1.
            gamma_kl_c_fade_in = 1.
        else:
            if fade_in_type == 'linear':
                alpha_fade_in = min(n_batches_since_progressive_step_start / self.n_batches_fade_in, 1.0)  # alpha of current progressive step
                gamma_kl_z_fade_in = min((n_batches_since_progressive_step_start / self.n_batches_fade_in) * (1 - self.gamma_kl_z_pretrain) + self.gamma_kl_z_pretrain, 1.0)
                gamma_kl_c_fade_in = min((n_batches_since_progressive_step_start / self.n_batches_fade_in) * (1 - self.gamma_kl_c_pretrain) + self.gamma_kl_c_pretrain, 1.0)
            elif fade_in_type == 'cosine_annealing':
                if n_batches_since_progressive_step_start < self.n_batches_fade_in:
                    alpha_fade_in = self.cosine_annealing(t = n_batches_since_progressive_step_start, t_max = self.n_batches_fade_in, min_val = 0, max_val = 1)
                    gamma_kl_z_fade_in = self.cosine_annealing(t = n_batches_since_progressive_step_start, t_max = self.n_batches_fade_in, min_val = self.gamma_kl_z_pretrain, max_val = 1)
                    gamma_kl_c_fade_in = self.cosine_annealing(t = n_batches_since_progressive_step_start, t_max = self.n_batches_fade_in, min_val = self.gamma_kl_c_pretrain, max_val = 1)
                else:
                    alpha_fade_in = 1.
                    gamma_kl_z_fade_in = 1.
                    gamma_kl_c_fade_in = 1.

        # build alpha_fade_in_list
        for _ in range(s_progressive_step):
            alpha_enc_fade_in_list.append(1.)
            alpha_dec_fade_in_list.append(1.)

        # if exactly as in progressive learning/training paper: append alpha_fade_in
        # if ammended so that more smooth: append 1. (always fully in)
        alpha_enc_fade_in_list.append(1.)

        alpha_dec_fade_in_list.append(alpha_fade_in)

        for _ in range(s_progressive_step + 1, self.J_n_mixtures):
            alpha_enc_fade_in_list.append(1.)
            alpha_dec_fade_in_list.append(0.)
        # reverse list, since last facet is faded in first
        alpha_enc_fade_in_list = list(reversed(alpha_enc_fade_in_list))
        alpha_dec_fade_in_list = list(reversed(alpha_dec_fade_in_list))

        # build gamma_kl_list
        for _ in range(s_progressive_step):
            # active layers (including the current one which is potentially faded-in, have "full" gamma
            gamma_kl_z_list.append(1.0)  # if exactly as in progressive learning/training paper: append beta_z
            gamma_kl_c_list.append(1.0)  # if exactly as in progressive learning/training paper: append beta_c

        # if exactly as in progressive learning/training paper: append beta_z/beta_c
        # if amended so that more smooth: append faded-in gamma (exactly same value as alpha_dec)
        gamma_kl_z_list.append(gamma_kl_z_fade_in)  # gamma of current progressive step
        gamma_kl_c_list.append(gamma_kl_c_fade_in)  # gamma of current progressive step

        for _ in range(s_progressive_step + 1, self.J_n_mixtures):
            # pretraining layers
            gamma_kl_z_list.append(self.gamma_kl_z_pretrain)
            gamma_kl_c_list.append(self.gamma_kl_c_pretrain)
        # reverse lists, since last facet is faded in first
        gamma_kl_z_list = list(reversed(gamma_kl_z_list))
        gamma_kl_c_list = list(reversed(gamma_kl_c_list))

        return alpha_enc_fade_in_list, alpha_dec_fade_in_list, gamma_kl_z_list, gamma_kl_c_list


    def encode(self, x: torch.tensor):
        """
        Estimate parameters of q(z_j | x), and sample from these distribution.

        Args:
            x: Input tensor of dimension (self.in_dim).

        Returns:
            mu and log(variance) of q(z_j | x) (as list), each tensor of dimension (self.z_dim).
        """
        h_list = self.encoder(x)
        if self.do_progressive_training and self.model_type == 'fc_per_facet_enc_shared_dec':
            h_list = [h * self.alpha_enc_fade_in_list[idx] for idx, h in enumerate(h_list)]

        mu_q_z_j_x_list, log_sigma_square_q_z_j_x_list = [], []
        for j in range(self.J_n_mixtures):
            mu_q_z_x = self.fc_mu_q_z_x_list[j](h_list[j])
            log_sigma_square_q_z_x = self.fc_log_sigma_square_q_z_x_list[j](h_list[j])
            mu_q_z_j_x_list.append(mu_q_z_x)
            log_sigma_square_q_z_j_x_list.append(log_sigma_square_q_z_x)

        return mu_q_z_j_x_list, log_sigma_square_q_z_j_x_list


    def decode(self, z_sample_q_z_j_x_list: torch.tensor):
        """
        Estimate the parameters of p(x | z).

        Args:
            z_sample_q_z_0_x: z sample, tensor of dimension (self.z_dim_0).
            z_sample_q_z_1_x: z sample, tensor of dimension (self.z_dim_1).

        Returns:
            x: Mode of p(x | z), a tensor of dimension (self.in_dim).
        """
        if self.do_progressive_training and self.model_type == 'fc_per_facet_enc_shared_dec':
            z_sample_q_z_j_x_list = [z_sample_q_z_j_x_list[j] * self.alpha_dec_fade_in_list[j] for j in range(self.J_n_mixtures)]

        h = self.decoder(z_sample_q_z_j_x_list)
        x_hat = self.layer_x_hat_z(h)
        if self.act_x_hat_z is not None:
            x_hat = self.act_x_hat_z(x_hat)

        return x_hat


    # TODO rename to ... prob -> it computes probs, not the distribution
    def compute_q_c_j_x(self, z_sample_q_z_j_x_list: List[torch.tensor]):
        """
        Compute all q(c_j | x) (referred to as gamma_c in the VaDE paper).
        q(c_j | x) is approximated with p(c_j | z) (See equation (16)).

        Args:
            z_sample_q_z_j_x_list: List of samples drawn from q(z_j | x).

        Returns:
            The probabilities of q(c_0 | x) == the probabilities of p(c_0 | z_0)
        """
        if self.cov_type_p_z_c == 'diag':
            sigma_square_p_z_j_c_j_list = self.sigma_square_p_z_j_c_j_list   # calls property -> only do once
            p_z_j_c_j_list = [D.Independent(D.Normal(loc=self.mu_p_z_j_c_j_list[j].permute(1, 0), scale=torch.sqrt(sigma_square_p_z_j_c_j_list[j].permute(1, 0))), 1) for j in range(self.J_n_mixtures)]
        elif self.cov_type_p_z_c == 'full':
            l_mat_p_z_j_c_j_list = self.l_mat_p_z_j_c_j_list
            p_z_j_c_j_list = [D.MultivariateNormal(loc=self.mu_p_z_j_c_j_list[j].permute(1, 0), scale_tril=l_mat_p_z_j_c_j_list[j].permute(2, 0, 1)) for j in range(self.J_n_mixtures)]


        # like _pad call in https://pytorch.org/docs/stable/_modules/torch/distributions/mixture_same_family.html#MixtureSameFamily.log_prob
        z_sample_q_z_j_x_pad_list = [torch.unsqueeze(z_sample_q_z_j_x_list[j], -2) for j in range(self.J_n_mixtures)]  # see _pad call: self._event_ndims in p_z_0 MixtureSameFamily model below is 1

        log_prob_p_z_j_c_j_list = [p_z_j_c_j_list[j].log_prob(z_sample_q_z_j_x_pad_list[j]) for j in range(self.J_n_mixtures)]
        log_prob_p_c_j_list = [torch.log_softmax(self._pi_p_c_j_list[j], dim=-1) for j in range(self.J_n_mixtures)]

        prob_p_c_j_z_j_list = [torch.softmax(log_prob_p_z_j_c_j_list[j] + log_prob_p_c_j_list[j], dim=1) for j in range(self.J_n_mixtures)]

        # note: q(c_j | x) = p(c_j | z_j)
        return prob_p_c_j_z_j_list


    def compute_loss_5terms(self, x: torch.tensor, x_hat: torch.tensor,
                            q_z_j_x_list: List[torch.distributions.Independent],
                            z_sample_q_z_j_x_list: List[torch.tensor], epoch: int):
        """
        Computes the ELBO of the log likelihood, with a negative sign (-> loss).

        Assumes L=1 (1 MC sample drawn).

        For L>1, all arguments of compute_loss_new would have to have one more dimension which is the l dimension
        (-> we have to sample z multiple times, need multiple x batches etc.).

        Args:
            x: The input of the network, a tensor of dimension (self.in_dim).
            x_hat: The "autoencoded" output of the network, a tensor of dimension (self.in_dim).
            q_z_j_x_list: List of normal distribution objects of all q(z_j | x).
            z_sample_q_z_j_x_list: List of samples drawn from all q(z_j | x).
        Returns:
            The average loss for this batch.
        """
        # term 1: compute log p(x|z), the MC estimate of E_{q(z,c|x)}[log p(x|z)] where z~q(z|x)
        if self.likelihood_model == 'Bernoulli':
            p_x_z = D.Independent(D.Bernoulli(probs=torch.clamp(x_hat, min=1e-10, max=1-(1e-10))), 1)
        elif self.likelihood_model == 'Gaussian':
            p_x_z = D.Independent(D.Normal(loc=x_hat, scale=torch.ones_like(x_hat) * self.sigma_multiplier_p_x_z), 1)
        log_prob_p_x_z = p_x_z.log_prob(x)  # e.g. torch.Size([8])

        # define variables and compute log q(c_1|x) (= log p(c_1|z_1)) and log q(c_2|x) (= log p(c_2|z_2)) for use in the remaining terms
        # p_c_0 = D.Categorical(probs=self.pi_p_c_0)
        # p_c_1 = D.Categorical(probs=self.pi_p_c_1)
        if self.cov_type_p_z_c == 'diag':
            sigma_square_p_z_j_c_j_list = self.sigma_square_p_z_j_c_j_list
            p_z_j_c_j_list = [D.Independent(D.Normal(loc=self.mu_p_z_j_c_j_list[j].permute(1, 0), scale=torch.sqrt(sigma_square_p_z_j_c_j_list[j].permute(1, 0))), 1) for j in range(self.J_n_mixtures)]
        elif self.cov_type_p_z_c == 'full':
            l_mat_p_z_j_c_j_list = self.l_mat_p_z_j_c_j_list
            p_z_j_c_j_list = [D.MultivariateNormal(loc=self.mu_p_z_j_c_j_list[j].permute(1, 0), scale_tril=l_mat_p_z_j_c_j_list[j].permute(2, 0, 1)) for j in range(self.J_n_mixtures)]

        # like _pad call in https://pytorch.org/docs/stable/_modules/torch/distributions/mixture_same_family.html#MixtureSameFamily.log_prob
        z_sample_q_z_j_x_pad_list = [torch.unsqueeze(z_sample_q_z_j_x_list[j], -2) for j in range(self.J_n_mixtures)]  # see _pad call: self._event_ndims in p_z_0 MixtureSameFamily model below is 1

        log_prob_p_z_j_c_j_list = [p_z_j_c_j_list[j].log_prob(z_sample_q_z_j_x_pad_list[j]) for j in range(self.J_n_mixtures)]
        log_prob_p_c_j_list = [torch.log_softmax(self._pi_p_c_j_list[j], dim=-1) for j in range(self.J_n_mixtures)]

        prob_q_c_j_x_list = [torch.softmax(log_prob_p_z_j_c_j_list[j] + log_prob_p_c_j_list[j], dim=1) for j in range(self.J_n_mixtures)]

        # term 2: compute the MC estimate of E_{q(z,c|x)}[log p(z|c)] where z~q(z|x)
        log_prob_E_p_z_c_list = [torch.sum(prob_q_c_j_x_list[j] * log_prob_p_z_j_c_j_list[j], dim=1) for j in range(self.J_n_mixtures)]  # [B]

        # term 3: compute E_{q(z,c|x)}[log p(c)]
        pi_p_c_j_list = self.pi_p_c_j_list
        log_p_c_j_list = [torch.log(pi_p_c_j_list[j]).unsqueeze(0) for j in range(self.J_n_mixtures)]
        log_prob_E_p_c_list = [torch.sum(prob_q_c_j_x_list[j] * log_p_c_j_list[j], dim=1) for j in range(self.J_n_mixtures)]

        # term 4: compute the MC estimate of E_{q(z,c|x)}[log q(z|x)] where z~q(z|x)
        log_prob_E_q_z_j_x_list = [q_z_j_x_list[j].log_prob(z_sample_q_z_j_x_list[j]) for j in range(self.J_n_mixtures)]   # torch.Size([B])

        # term 5: compute E_{q(z,c|x)}[log q(c|x)]
        log_prob_E_q_c_j_x_list = [torch.sum(prob_q_c_j_x_list[j] * torch.log(torch.clamp(prob_q_c_j_x_list[j], min=1e-6)), dim=1) for j in range(self.J_n_mixtures)]

        # compute ELBO
        if self.do_progressive_training:
            beta_z = torch.tensor(self.gamma_kl_z_list).to(self.device)
            beta_c = torch.tensor(self.gamma_kl_c_list).to(self.device)
        else:
            beta_z = 1.0  #  if as in ProVLAE implementation: self.beta_z
            beta_c = 1.0  #  if as in ProVLAE implementation: self.beta_c

        mean_log_prob_p_x_z = torch.mean(log_prob_p_x_z)
        mean_log_prob_E_p_z_c = torch.mean(torch.sum(torch.stack(log_prob_E_p_z_c_list, 1), 1))
        mean_log_prob_E_p_c = torch.mean(torch.sum(torch.stack(log_prob_E_p_c_list, 1), 1))
        mean_log_prob_E_q_z_x = torch.mean(torch.sum(torch.stack(log_prob_E_q_z_j_x_list, 1), 1))
        mean_log_prob_E_q_c_x = torch.mean(torch.sum(torch.stack(log_prob_E_q_c_j_x_list, 1), 1))
        kl_z_each_j = torch.mean(torch.stack(log_prob_E_q_z_j_x_list, 1) - torch.stack(log_prob_E_p_z_c_list, 1), dim=0)
        kl_c_each_j = torch.mean(torch.stack(log_prob_E_q_c_j_x_list, 1) - torch.stack(log_prob_E_p_c_list, 1), dim=0)
        kl_z = torch.sum(kl_z_each_j)  # without clamping -> used for logging
        kl_c = torch.sum(kl_c_each_j)  # without clamping -> used for logging

        ELBO = mean_log_prob_p_x_z - torch.sum(beta_z * kl_z_each_j) - torch.sum(beta_c * kl_c_each_j)

        loss = -ELBO

        return loss, mean_log_prob_p_x_z, mean_log_prob_E_p_z_c, mean_log_prob_E_p_c, mean_log_prob_E_q_z_x, mean_log_prob_E_q_c_x, kl_z, kl_c


    def initialize_p_z_c_params_with_gmm(self, train_loader, model_type: str, epoch: int, batch_idx: int):
        """
        Initialize parameters of p(z | c) with mean and variances of Gaussian Mixture model (with diagonal
        covariance matrix) trained on values sampled from q(z | x).

        Args:
            train_loader: data loader to loop over training data
        """
        self.eval()
        data_j_list = [[] for j in range(self.J_n_mixtures)]  # stores all z_sample of all inputs from training epoch
        # loop over all examples in one epoch of training data
        for batch_idx, (x, _) in enumerate(train_loader):
            if 'cuda' in self.device.type:  # always move to GPU (even if already on there)
                x = x.to(self.device)
            if model_type in ['fc_shared', 'fc_per_facet_enc_shared_dec', 'fc_vlae']:
                x = x.view(x.size(0), -1).float()
            elif model_type in ['resnet', 'convnet']:
                x = x.float()
            x = torch.autograd.Variable(x)
            # OLD VERSION:
            # x_hat, mu_q_z_x, log_sigma_square_q_z_x, z_sample_q_z_x = self.forward(x)
            _x_hat, _q_z_j_x_list, z_sample_q_z_j_x_list = self.forward(x, epoch, batch_idx)
            for j in range(self.J_n_mixtures):
                data_j_list[j].append(z_sample_q_z_j_x_list[j].data.cpu().numpy())
        for j in range(self.J_n_mixtures):
            data_j_list[j] = np.concatenate(data_j_list[j])

        if self.cov_type_p_z_c == 'diag':
            gmm_j_list = []
            for j in range(self.J_n_mixtures):
                gmm_j = GaussianMixture(n_components=self.n_clusters_j_list[j], covariance_type='diag')  # diagonal covariance matrix (also in the case of having a full coviance matrix for p_z_c)
                gmm_j.fit(data_j_list[j])
                gmm_j_list.append(gmm_j)
            # initialize parameters of p(z_0 | c_0) amd p(z_1 | c_1) with GMM
            for j in range(self.J_n_mixtures):
                self.mu_p_z_j_c_j_list[j].data.copy_(torch.from_numpy(gmm_j_list[j].means_.T.astype(np.float32)))
                # initialise variances to be fixed values instead
                nn.init.constant_(self._sigma_square_p_z_j_c_j_list[j], softplus_inverse_numpy(self.fixed_var_init, beta=10))
        elif self.cov_type_p_z_c == 'full':
            variant = 1
            if variant == 1:
                # Variant 1: just use diagonal elements, rest initialized with 0
                # fit Gaussian mixture model on z_0 and z_1 samples
                gmm_j_list = []
                for j in range(self.J_n_mixtures):
                    gmm_j = GaussianMixture(n_components=self.n_clusters_j_list[j], covariance_type='diag')  # diagonal covariance matrix (also in the case of having a full coviance matrix for p_z_c)
                    gmm_j.fit(data_j_list[j])
                    # initialize parameters of p(z_0 | c_0) and p(z_1 | c_1) with GMM
                    self.mu_p_z_j_c_j_list[j].data.copy_(torch.from_numpy(gmm_j.means_.T.astype(np.float32)))

                    # initialise diagonal entries in the covariance matrix to be fixed values instead
                    _l_mat_p_z_j_c_j = nn.init.constant_(self._l_mat_p_z_j_c_j_list[j], softplus_inverse_numpy(np.sqrt(self.fixed_var_init), beta=10))
                    mask = torch.eye(_l_mat_p_z_j_c_j.shape[0]).unsqueeze(2).repeat(1, 1, _l_mat_p_z_j_c_j.shape[2]).to(self.device)
                    self._l_mat_p_z_j_c_j_list[j].data.copy_(mask * _l_mat_p_z_j_c_j)
            elif variant == 2:
                pass


    def initialize_p_z_c_params_with_glorot(self):
        """
        Randomly initialize parameters of p(z | c).
        """
        if self.cov_type_p_z_c == 'diag':
            for j in range(self.J_n_mixtures):
                nn.init.xavier_normal_(self.mu_p_z_j_c_j_list[j])
                # initialise variances to be fixed values instead
                nn.init.constant_(self._sigma_square_p_z_j_c_j_list[j], np.log(self.fixed_var_init))
        elif self.cov_type_p_z_c == 'full':
            for j in range(self.J_n_mixtures):
                nn.init.xavier_normal_(self.mu_p_z_j_c_j_list[j])
                if self.init_off_diag_cov_p_z_c == True:
                    nn.init.xavier_normal_(self._l_mat_p_z_j_c_j_list[j])
                else:
                    # initialise diagonal entries in the covariance matrix to be fixed values instead
                    _l_mat_p_z_j_c_j = nn.init.constant_(self._l_mat_p_z_j_c_j_list[j], softplus_inverse_numpy(np.sqrt(self.fixed_var_init), beta=10))
                    # define a mask with J identity matrices concatenated along axis 2
                    mask = torch.eye(_l_mat_p_z_j_c_j.shape[0]).unsqueeze(2).repeat(1, 1, _l_mat_p_z_j_c_j.shape[2]).to(self.device)
                    # only initialise the diagonals, set off-diagonal entries to be zeros
                    self._l_mat_p_z_j_c_j_list[j].data.copy_(mask * _l_mat_p_z_j_c_j)


    def initialize_fc_layers(self, weights_init_type: str = 'xavier_uniform'):
        """
        Initializes all weights of Linear() layers of a model object.

        Args:
            model: A torch.nn.Module custom model object.
            weights_init_type: Type of initialization used.
        """
        classname = self.__class__.__name__
        if classname.find("Linear") != -1:
            if weights_init_type == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(self.weight.data)


