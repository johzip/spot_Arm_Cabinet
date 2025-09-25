import gymnasium as gym
from .spot_cabinet_env import SpotCabinetEnvCfg, SpotCabinetEnv

##
# Register Gym environments.
##

gym.register(
    id="MoDe-Spot-Curtain-v0",
    entry_point="task.Cabinet.spot_cabinet_env:SpotCabinetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SpotCabinetEnvCfg,
    },
)
