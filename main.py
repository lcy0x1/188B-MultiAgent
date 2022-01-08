
from datetime import datetime

import numpy as np
import tensorflow as tf

from mava.components.tf import networks
from mava.systems.tf import mappo
from mava.utils import lp_utils

network_factory = lp_utils.partial_kwargs(mappo.make_default_networks)

# Directory to store checkpoints and log data.
base_dir = "~./mava"

# File name
mava_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    pass

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
