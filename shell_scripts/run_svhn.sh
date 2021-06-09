#!/bin/bash

cd ..
python3 train.py --config_args_path "configs/svhn.yml"
cd shell_scripts