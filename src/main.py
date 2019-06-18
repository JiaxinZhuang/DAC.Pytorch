"""Main"""

import random
import os
import torch
import numpy as np

from agent import Agent
from config import Config
from action import time_this


@time_this
def init_environment(config: dict):
    """set environment
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']

    # set random seed
    seed = config["seed"]
    if seed != -1:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print(">> Not to seed random seed")

def main():
    """Main function
    """
    # Import config
    config = Config().get_config()
    # Init environ, such as random seed for reproducity
    init_environment(config)
    # Declare an agent to train and eval
    agent = Agent(config)
    # Train and eval
    agent.run()

if __name__ == "__main__":
    main()
