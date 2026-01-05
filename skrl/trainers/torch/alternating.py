import copy
from typing import List, Optional, Union

import atexit
import sys

import numpy as np
import tqdm

import torch

from skrl.agents.torch import Agent
from skrl.envs.wrappers.torch import Wrapper
from skrl.trainers.torch import Trainer

# from skrl.utils.model_instantiators.torch import shared_model

# fmt: off
# [start-config-dict-torch]
ALTERNATING_TRAINER_DEFAULT_CONFIG = {
    "timesteps": 100000,  # number of timesteps to train for
    "headless": False,  # whether to use headless mode (no rendering)
    "disable_progressbar": False,  # whether to disable the progressbar. If None, disable on non-TTY
    "close_environment_at_exit": True,  # whether to close the environment on normal program termination
    "environment_info": "episode",  # key used to get and log environment info
    "stochastic_evaluation": False,  # whether to use actions rather than (deterministic) mean actions during evaluation
    "switch_per_steps": 1e4,
}


# [end-config-dict-torch]
# fmt: on

class AlternatingTrainer(Trainer):
    def __init__(
            self,
            env: Wrapper,
            agents: Union[Agent, List[Agent]],
            agents_scope: Optional[List[int]] = None,
            cfg: Optional[dict] = None
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

        # if agents_scope is None:
        #     agents_scope = [env.num_envs] * len(agents)

        super().__init__(
            env=env,
            agents=agents,
            agents_scope=agents_scope,
            cfg=_cfg
        )

        self.switch_per_steps = self.cfg.get("switch_per_steps", 10000)  # 交替训练切换频率

        # init agents
        if self.num_simultaneous_agents > 1:
            for agent in self.agents:
                agent.init(trainer_cfg=self.cfg)
        else:
            self.agents.init(trainer_cfg=self.cfg)
            raise ValueError(
                f"AlternatingTrainer must have num_simultaneous_agents > 1, "
                f"but found only {self.num_simultaneous_agents}. "
                "Please check your configurations"
            )
        pass

    def train(self) -> None:
        """Train the agents alternately

        This method executes the following steps in loop:

        - Pre-interaction (sequentially)
        - Compute actions (sequentially)
        - Interact with the environments
        - Render scene
        - Record transitions (sequentially)
        - Post-interaction (sequentially)
        - Reset environments
        """

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
                range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # get fixed and non-fixed agents' idx
            non_fixed_agent_idx = (timestep // self.switch_per_steps) % len(self.agents)
            for i, (agent, agent_name) in enumerate(zip(self.agents, self.env._unwrapped.possible_agents)):
                if i == non_fixed_agent_idx:
                    agent.set_running_mode("train")
                else:
                    agent.set_running_mode("eval")

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions
                actions = {}
                for i, (agent, agent_name) in enumerate(zip(self.agents, self.env._unwrapped.possible_agents)):
                    outputs = agent.act(states[agent_name], timestep=timestep, timesteps=self.timesteps)
                    # outputs[0]: sampled_actions with exploration
                    # outputs[1]: log_prob
                    # outputs[2]: extra_info, such as "mean_action" in PPO
                    if i == non_fixed_agent_idx:
                        actions[agent_name] = outputs[0]
                    else:
                        # if mean_actions exists, return outputs[-1]["mean_actions"]
                        # else return outputs[0]
                        actions[agent_name] = outputs[-1].get("mean_actions", outputs[0])

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # record the environments' transitions
                # only record the environments' transitions of non-fixed agents
                for i, (agent, agent_name) in enumerate(zip(self.agents, self.env._unwrapped.possible_agents)):
                    if i == non_fixed_agent_idx:
                        agent.record_transition(
                            states=states[agent_name],
                            actions=actions[agent_name],
                            rewards=rewards[agent_name],
                            next_states=next_states[agent_name],
                            terminated=terminated[agent_name],
                            truncated=truncated[agent_name],
                            infos=infos,
                            timestep=timestep,
                            timesteps=self.timesteps,
                        )

                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            for agent in self.agents:
                                agent.track_data(f"Info / {k}", v.item())

            # post-interaction
            # for agent in self.agents:
            #     agent.post_interaction(timestep=timestep, timesteps=self.timesteps)
            for i, agent in enumerate(self.agents):
                if i == non_fixed_agent_idx:
                    agent.post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def eval(self) -> None:
        assert self.env.num_agents > 1, "This method is not allowed for single-agent"

        # reset env
        states, infos = self.env.reset()
        shared_states = self.env.state()

        for timestep in tqdm.tqdm(
                range(self.initial_timestep, self.timesteps), disable=self.disable_progressbar, file=sys.stdout
        ):

            # pre-interaction
            for agent in self.agents:
                agent.pre_interaction(timestep=timestep, timesteps=self.timesteps)

            with torch.no_grad():
                # compute actions:
                actions = {}
                for i, (agent, agent_name) in enumerate(zip(self.agents, self.env._unwrapped.possible_agents)):
                    outputs = agent.act(states[agent_name], timestep=timestep, timesteps=self.timesteps)
                    actions[agent_name] = (
                        outputs[0]
                        if self.stochastic_evaluation
                        else outputs[-1].get('mean_actions', outputs[0])
                    )

                # step the environments
                next_states, rewards, terminated, truncated, infos = self.env.step(actions)
                shared_next_states = self.env.state()
                infos["shared_states"] = shared_states
                infos["shared_next_states"] = shared_next_states

                # render scene
                if not self.headless:
                    self.env.render()

                # write data to TensorBoard
                for agent, agent_name in zip(self.agents, self.env._unwrapped.possible_agents):
                    agent.record_transition(
                        states=states[agent_name],
                        actions=actions[agent_name],
                        rewards=rewards[agent_name],
                        next_states=next_states[agent_name],
                        terminated=terminated[agent_name],
                        truncated=truncated[agent_name],
                        infos=infos,
                        timestep=timestep,
                        timesteps=self.timesteps,
                    )
                # log environment info
                if self.environment_info in infos:
                    for k, v in infos[self.environment_info].items():
                        if isinstance(v, torch.Tensor) and v.numel() == 1:
                            self.agents.track_data(f"Info / {k}", v.item())

            # post-interaction
            for agent in self.agents:
                super(type(agent), agent).post_interaction(timestep=timestep, timesteps=self.timesteps)

            # reset environments
            if not self.env.agents:
                with torch.no_grad():
                    states, infos = self.env.reset()
                    shared_states = self.env.state()
            else:
                states = next_states
                shared_states = shared_next_states

    def single_agent_train(self) -> None:
        raise NotImplementedError()

    def single_agent_eval(self) -> None:
        raise NotImplementedError()

    def multi_agent_train(self) -> None:
        raise NotImplementedError("AlternatingTrainer use train() rather than multi_agent_train()")

    def multi_agent_eval(self) -> None:
        raise NotImplementedError("AlternatingTrainer use eval() rather than multi_agent_eval()")
