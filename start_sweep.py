# !!!
# Note: One can use command line arguments when running this script in the usual way as for train.py!
# They will be parsed within the call to train(...).

import wandb
import train
from utils import load_args_from_yaml

seed_values = list(range(10))

sweep_config = {
  "name": "Seed sweep",
  "method": "grid",
  "parameters": {
      "seed": {
        "values": seed_values
      },
    },
}

wandb_args = load_args_from_yaml('configs/wandb.yml')
sweep_id = wandb.sweep(sweep_config, project=wandb_args.project_name, entity=wandb_args.team_name)

wandb.agent(sweep_id, function=train.train)
