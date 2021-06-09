#!/bin/bash

cd ..
python3 eval_top10_cluster_examples.py --model_path "pretrained_models/mnist.pt" --results_dir "results/mnist" --device "cuda:0"
python3 eval_sample_generation.py --model_path "pretrained_models/mnist.pt" --results_dir "results/mnist" --device "cuda:0" --temperature 1.0
python3 eval_compositionality.py --model_path "pretrained_models/mnist.pt" --results_dir "results/mnist" --device "cuda:0" --swapped_facet 0 --first_n_clusters_swapped_1 10 --first_n_clusters_swapped_2 10
cd shell_scripts

