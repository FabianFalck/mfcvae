#!/bin/bash

cd ..
python3 eval_top10_cluster_examples.py --model_path "pretrained_models/3dshapes_1.pt" --results_dir "results/3dshapes_1" --device "cuda:0"
python3 eval_sample_generation.py --model_path "pretrained_models/3dshapes_1.pt" --results_dir "results/3dshapes_1" --device "cuda:0" --temperature 0.3
python3 eval_compositionality.py --model_path "pretrained_models/3dshapes_1.pt" --results_dir "results/3dshapes_1" --device "cuda:0" --swapped_facet 1 --first_n_clusters_swapped_1 10 --first_n_clusters_swapped_2 10
cd shell_scripts
