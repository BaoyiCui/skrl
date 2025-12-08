import copy
from typing import List, Optional, Union

import atexit
import sys
import tqdm

import torch

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

# fmt: off
# [start-config-dict-torch]
ALTERNATING_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,  # number of timesteps to train for
    "headless": False,  # whether to use headless mode (no rendering)
    "disable_progressbar": False,  # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,  # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
    "stochastic_evaluation": False,  # whether to use actions rather than (deterministic) mean actions during evaluation
}


# [end-config-dict-torch]
# fmt: on

class AlternatingTrainer(Trainer):
    def __init__(
            self,
            env: Wrapper,
            agents: Union[Agent, List[Agent]],
            agents_scope: Optional[List[int]] = None,
            cfg:Optional[dict] = None
    ):
        """Sequential trainer

        Train agents alternately.

        :param env: Environment to train on
        :type env: skrl.envs.wrappers.torch.Wrapper
        :param agents: Agents to train
        :type agents: Union[Agent, List[Agent]]
        :param agents_scope: Number of environments for each agent to train on (default: ``None``)
        :type agents_scope: tuple or list of int, optional
        :param cfg: Configuration dictionary (default: ``None``).
                    See SEQUENTIAL_TRAINER_DEFAULT_CONFIG for default values
        :type cfg: dict, optional
        """
        _cfg = copy.deepcopy(ALTERNATING_TRAINER_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        agents_scope = agents_scope if agents_scope is not None else []
        super().__init__(
            env=env,
            agents=agents,
            agents_scope=agents_scope,
            cfg=_cfg
        )

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)

    def train(self) -> None:
        """Train the agents alternately.

        This method executes the following steps in loop:

        -

        """
        # TODO: WRITE THIS DOCSTRING
        # set running mode
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.set_running_mode("train")
        else:
            self.agents.set_running_mode("train")

        # non-simultaneous agents

    def eval(self) -> None:
        # TODO
        pass

    def single_agent_train(self) -> None:
        # TODO
        pass

    def single_agent_eval(self) -> None:
        # TODO
        pass

    def multi_agent_train(self) -> None:
        # TODO
        pass

    def multi_agent_eval(self) -> None:
        # TODO
        pass