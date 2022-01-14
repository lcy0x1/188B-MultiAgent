import functools
import json
from datetime import datetime
import pkg_resources

import launchpad as lp
import sonnet as snt
from mava.systems.tf import mappo
from mava.utils import lp_utils
from mava.utils.loggers import logger_utils
from mava.wrappers import MonitorParallelEnvironmentLoop

from pz_vehicle import vehicle_env

config_data = json.load(open(pkg_resources.resource_filename(__name__, "./config.json")))
config = vehicle_env.Config(config_data)
network_factory = lp_utils.partial_kwargs(mappo.make_default_networks)
env_name = "pz_vehicle"
action_space = "continuous"
environment_factory = functools.partial(lambda: vehicle_env.env(config),
                                        env_name=env_name,
                                        action_space=action_space,
                                        )
base_dir = "~./mava"
mava_id = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
log_every = 15
logger_factory = functools.partial(
    logger_utils.make_logger,
    directory=base_dir,
    to_terminal=True,
    to_tensorboard=True,
    time_stamp=mava_id,
    time_delta=log_every,
)
checkpoint_dir = f"{base_dir}/{mava_id}"

system = mappo.MAPPO(
    environment_factory=environment_factory,
    network_factory=network_factory,
    logger_factory=logger_factory,
    num_executors=1,
    policy_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
    critic_optimizer=snt.optimizers.Adam(learning_rate=1e-4),
    checkpoint_subpath=checkpoint_dir,
    max_gradient_norm=40.0,
    checkpoint=False,
    batch_size=1024,

    # Record agents in environment.
    eval_loop_fn=MonitorParallelEnvironmentLoop,
    eval_loop_fn_kwargs={"path": checkpoint_dir, "record_every": 10, "fps": 5},
).build()
local_resources = lp_utils.to_device(program_nodes=system.groups.keys(),nodes_on_gpu=["trainer"])

lp.launch(
    system,
    lp.LaunchType.LOCAL_MULTI_PROCESSING,
    terminal="output_to_files",
    local_resources=local_resources,
)
if __name__ == '__main__':
    pass
