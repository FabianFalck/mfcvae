import json
import yaml
import argparse

file_name = 'mnist'

in_file_path = 'path/to/configs/dir' + file_name + '.json'
out_file_path = 'path/to/configs/dir' + file_name + '.yaml'

with open(in_file_path) as json_file:
    config = json.load(json_file)
# print(configs.keys())
# remove certain keys
removed_keys = ['_wandb', 'beta_c', 'beta_z', 'lambda_c', 'lambda_z', 'n_stacks', 'pretrain', 'free_bits', 'out_channels_list', 'channels_multiplier', 'non_dim_change_layer',
                'update_lr_every_epoch', 'pretrain_with_standard_gaussian', 'freeze_priors_during_pretraining',
                'config_args_path']  # THIS ONE MUST BE REMOVED!!!
for key in removed_keys:
    config.pop(key, None)
# change to correct dioctionary value
for key, value in config.items():
    config[key] = value['value']

print(config)

# convert to yaml file and save
with open(out_file_path, 'w') as yaml_file:
    yaml.dump(config, yaml_file)






# dict_keys(['seed', 'user', '_wandb', 'beta_c', 'beta_z', 'device', 'min_lr', 'dataset', 'init_lr', 'lambda_c', 'lambda_z', 'lr_decay', 'n_epochs', 'n_stacks', 'pretrain', 'do_vis_pi', 'free_bits',
#            'batch_size', 'fix_pi_p_c', 'model_type', 'save_model', 'wandb_mode', 'do_vis_test', 'J_n_mixtures', 'do_vis_recon', 'do_vis_train', 'z_j_dim_list', 'dec_rung_dims', 'enc_rung_dims',
#            'cov_type_p_z_c', 'dataset_on_gpu', 'fixed_var_init', 'n_test_batches', 'do_vis_conf_mat', 'eval_batch_size', 'init_type_p_z_c', 'config_args_path', 'dec_backbone_dims',
#            'dec_rung_n_hidden', 'decode_layer_dims', 'enc_backbone_dims', 'enc_rung_n_hidden', 'encode_layer_dims', 'n_batches_fade_in', 'n_clusters_j_list', 'out_channels_list',
#            'vis_every_n_epochs', 'channels_multiplier', 'gamma_kl_c_pretrain', 'gamma_kl_z_pretrain', 'do_vis_cluster_means', 'n_recons_per_cluster', 'non_dim_change_layer',
#            'dec_backbone_n_hidden', 'enc_backbone_n_hidden', 'update_lr_every_epoch', 'n_examples_per_cluster', 'sigma_multiplier_p_x_z', 'do_progressive_training',
#            'do_test_during_training', 'init_off_diag_cov_p_z_c', 'do_vis_examples_per_cluster', 'do_vis_latent_code_traversal', 'do_vis_n_samples_per_cluster',
#            'n_epochs_per_progressive_step', 'pretrain_with_standard_gaussian', 'freeze_priors_during_pretraining', 'n_sample_generations_per_cluster', 'n_sample_generation_plots_per_facet',
#            'do_vis_sample_generations_per_cluster'])
