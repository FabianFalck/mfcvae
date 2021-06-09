
# !!!!
# Call this script as, for example:
# python3 attach_sweep.py --sweep_id chvie4r1 --config_args_path "configs/svhn.yml"


import wandb
import train
import argparse
from utils import load_args_from_yaml

parser = argparse.ArgumentParser(description='Attaching to a sweep.')
parser.add_argument('--sweep_id', type=str, default="7qifsy12", metavar='N', help="ID of sweep to attach to.")
args, unknown = parser.parse_known_args()

wandb_args = load_args_from_yaml('configs/wandb.yml')
wandb.agent(wandb_args.team_name + '/' + wandb_args.project_name + '/' + args.sweep_id, function=train.train)

