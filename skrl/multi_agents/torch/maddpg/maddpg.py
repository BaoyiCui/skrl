from typing import Any, Mapping, Optional, Sequence, Union

import copy
import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.multi_agents.torch import MultiAgent
from skrl.resources.schedulers.torch import KLAdaptiveLR

# fmt: off
# [start-config-dict-torch]
MADDPG_DEFAULT_CONFIG = {
    'gradient_steps': 1,    # gradient steps
    'batch_size': 64,       # training batch size
    
    'discount_factor': 0.99,    # discount factor (gamma)
    'polyak': 0.005,            # soft update hyperparameter (tau)
    
    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})
    "shared_state_preprocessor": None,      # shared state preprocessor class (see skrl.resources.preprocessors)
    "shared_state_preprocessor_kwargs": {}, # shared state preprocessor's kwargs (e.g. {"size": env.shared_observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "exploration": {
        "noise": None,              # exploration noise
        "initial_scale": 1.0,       # initial scale for the noise
        "final_scale": 1e-3,        # final scale for the noise
        "timesteps": None,          # timesteps for the noise decay
    },

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


class MADDPG(MultiAgent):
    def __init__(
        self,
        possible_agents: Sequence[str],
        models: Mapping[str, Model],
        memories: Optional[Mapping[str, Memory]] = None,
        observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
        shared_observation_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
        joint_action_spaces: Optional[Union[Mapping[str, int], Mapping[str, gymnasium.Space]]] = None,
    ) -> None:
        """Multi-Agent Deep Deterministic Policy Gradient (MADDPG)


        :param possible_agents: Name of all possible agents the environment could generate
        :type possible_agents: list of str
        :param models: Models used by the agents.
                    External keys are environment agents' names. Internal keys are the models required by the algorithm
        :type models: nested dictionary of skrl.models.torch.Model
        :param memories: Memories to storage the transitions.
        :type memories: dictionary of skrl.memory.torch.Memory, optional
        :param observation_spaces: Observation/state spaces or shapes (default: ``None``)
        :type observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param action_spaces: Action spaces or shapes (default: ``None``)
        :type action_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        :param device: Device on which a tensor/array is or will be allocated (default: ``None``).
                        If None, the device will be either ``"cuda"`` if available or ``"cpu"``
        :type device: str or torch.device, optional
        :param cfg: Configuration dictionary
        :type cfg: dict
        :param shared_observation_spaces: Shared observation/state space or shape (default: ``None``)
        :type shared_observation_spaces: dictionary of int, sequence of int or gymnasium.Space, optional
        """
        _cfg = copy.deepcopy(MADDPG_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            possible_agents=possible_agents,
            models=models,
            memories=memories,
            observation_spaces=observation_spaces,
            action_spaces=action_spaces,
            device=device,
            cfg=_cfg,
        )

        self.shared_observation_spaces = shared_observation_spaces
        self.joint_action_spaces = joint_action_spaces

        # models
        self.policies = {uid: self.models[uid].get("policy", None) for uid in self.possible_agents}
        self.target_policies = {uid: self.models[uid].get("target_policy", None) for uid in self.possible_agents}
        self.critics = {uid: self.models[uid].get("critic", None) for uid in self.possible_agents}
        self.target_critics = {uid: self.models[uid].get("target_critic", None) for uid in self.possible_agents}

        for uid in self.possible_agents:
            # checkpoint models
            self.checkpoint_modules[uid]["policy"] = self.policies[uid]
            self.checkpoint_modules[uid]["target_policy"] = self.target_policies[uid]
            self.checkpoint_modules[uid]["critic"] = self.critics[uid]
            self.checkpoint_modules[uid]["target_critic"] = self.target_critics[uid]

            # broadcast models' parameters in distributed runs
            if config.torch.is_distributed:
                logger.info(f"Broadcasting models' parameters")
                if self.policies[uid] is not None:
                    self.policies[uid].broadcast_parameters()
                if self.critics[uid] is not None:
                    self.critics[uid].broadcast_parameters()

            if self.target_policies[uid] is not None and self.target_critics[uid] is not None:
                # freeze target networks w.r.t. optimizers (update via .update_parameters())
                self.target_policies[uid].freeze_parameters(True)
                self.target_critics[uid].freeze_parameters(True)

                # update target networks (hard udpate)
                self.target_policies[uid].update_parameters(self.policies[uid], polyak=1)
                self.target_critics[uid].update_parameters(self.critics[uid], polyak=1)

        # configuration
        # TODO: 这里为什么有的要用_as_dict有的不用，后面的写完了回头检查
        self._gradient_steps = self._as_dict(self.cfg["gradient_steps"])
        self._batch_size = self._as_dict(self.cfg["batch_size"])

        self._discount_factor = self._as_dict(self.cfg["discount_factor"])
        self._polyak = self._as_dict(self.cfg["polyak"])

        self._actor_learning_rate = self._as_dict(self.cfg["actor_learning_rate"])
        self._critic_learning_rate = self._as_dict(self.cfg["critic_learning_rate"])
        self._learning_rate_scheduler = self._as_dict(self.cfg["learning_rate_scheduler"])
        self._learning_rate_scheduler_kwargs = self._as_dict(self.cfg["learning_rate_scheduler_kwargs"])

        self._state_preprocessor = self._as_dict(self.cfg["state_preprocessor"])
        self._state_preprocessor_kwargs = self._as_dict(self.cfg["state_preprocessor_kwargs"])
        self._shared_state_preprocessor = self._as_dict(self.cfg["shared_state_preprocessor"])
        self._shared_state_preprocessor_kwargs = self._as_dict(self.cfg["shared_state_preprocessor_kwargs"])

        self._random_timesteps = self._as_dict(self.cfg["random_timesteps"])
        self._learning_starts = self._as_dict(self.cfg["learning_starts"])

        self._grad_norm_clip = self._as_dict(self.cfg["grad_norm_clip"])

        self._exploration_noise = self.cfg["exploration"]["noise"]
        self._exploration_initial_scale = self.cfg["exploration"]["initial_scale"]
        self._exploration_final_scale = self.cfg["exploration"]["final_scale"]
        self._exploration_timesteps = self.cfg["exploration"]["timesteps"]

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self._mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # set up optimizer and learning rate scheduler
        self.policy_optimizers = {}
        self.critic_optimizers = {}
        self.policy_schedulers = {}
        self.critic_schedulers = {}

        for uid in self.possible_agents:
            policy = self.policies[uid]
            critic = self.critics[uid]
            if policy is not None and critic is not None:
                policy_optimizer = torch.optim.Adam(policy.parameters(), lr=self._actor_learning_rate[uid])
                critic_optimizer = torch.optim.Adam(critic.parameters(), lr=self._critic_learning_rate[uid])
            self.policy_optimizers[uid] = policy_optimizer
            self.critic_optimizers[uid] = critic_optimizer
            if self._learning_rate_scheduler[uid] is not None:
                self.policy_schedulers[uid] = self._learning_rate_scheduler[uid](
                    self.policy_optimizers[uid],
                    **self._learning_rate_scheduler_kwargs[uid],
                )
                self.critic_schedulers[uid] = self._learning_rate_scheduler[uid](
                    self.critic_schedulers[uid],
                    **self._learning_rate_scheduler_kwargs[uid],
                )

            self.checkpoint_modules[uid]["policy_optimizer"] = self.policy_optimizers[uid]
            self.checkpoint_modules[uid]["critic_optimizer"] = self.critic_optimizers[uid]

            # set up preprocessors
            if self._state_preprocessor[uid] is not None:
                self._state_preprocessor[uid] = self._state_preprocessor[uid](**self._state_preprocessor_kwargs[uid])
            else:
                self._state_preprocessor[uid] = self._empty_preprocessor

            if self._shared_state_preprocessor[uid] is not None:
                self._shared_state_preprocessor[uid] = self._shared_state_preprocessor[uid](**self._shared_state_preprocessor_kwargs[uid])
                self.checkpoint_modules[uid]["shared_state_preprocessor"] = self._shared_state_preprocessor[uid]
            else:
                self._shared_state_preprocessor[uid] = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        self.clip_actions_mins = {}
        self.clip_actions_maxs = {}

        # create tensors in memory
        if self.memories:
            for uid in self.possible_agents:
                self.memories[uid].create_tensor(
                    name="states",
                    size=self.observation_spaces[uid],
                    dtype=torch.float32,
                )
                self.memories[uid].create_tensor(
                    name="shared_states",
                    size=self.shared_observation_spaces[uid],
                    dtype=torch.float32,
                )
                self.memories[uid].create_tensor(
                    name="next_states",
                    size=self.observation_spaces[uid],
                    dtype=torch.float32,
                )
                self.memories[uid].create_tensor(
                    name="shared_next_states",
                    size=self.shared_observation_spaces[uid],
                    dtype=torch.float32,
                )
                self.memories[uid].create_tensor(name="actions", size=self.action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="joint_actions", size=self.joint_action_spaces[uid], dtype=torch.float32)
                self.memories[uid].create_tensor(name="rewards", size=1, dtype=torch.float32)
                self.memories[uid].create_tensor(name="terminated", size=1, dtype=torch.bool)
                self.memories[uid].create_tensor(name="truncated", size=1, dtype=torch.bool)

                # tensor sampled during training
                self._tensor_names = [
                    "states",
                    "shared_states",
                    "actions",
                    "joint_actions",
                    "rewards",
                    "next_states",
                    "shared_next_states",
                    "terminated",
                    "truncated",
                ]

            if self.action_spaces[uid] is not None:
                self.clip_actions_mins[uid] = torch.tensor(self.action_spaces[uid].low, device=self.device)
                self.clip_actions_maxs[uid] = torch.tensor(self.action_spaces[uid].high, device=self.device)
            else:
                self.clip_actions_mins[uid] = None
                self.clip_actions_maxs[uid] = None

        # create temporary variables needed for storage and computation
        # self._current_shared_next_states = []

        # self._current_shared_states = []

    def act(self, states: Mapping[str, torch.Tensor], timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policies

        :param states: Environment's states
        :type states: dictionary of torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        actions = {}

        # sample random actions
        if timestep < self._random_timesteps:
            actions = {
                uid: self.policies[uid].random_act({"states": self._state_preprocessor(states[uid])}, role="policy") for uid in self.possible_agents
            }

            return actions

        # sample deterministic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            pass

        # add exploration noise
        if self._exploration_noise is not None:
            # sample noises

            # define exploration timesteps

            # apply exploration noise
            if timestep <= self._exploration_timesteps:
                pass

            # modify actions

            # record noises
            pass
        else:
            # record noises
            self.track_data("Exploration / Exploration noise (max)", 0)
            self.track_data("Exploration / Exploration noise (min)", 0)
            self.track_data("Exploration / Exploration noise (mean)", 0)

    def record_transition(
        self,
        states: Mapping[str, torch.Tensor],
        actions: Mapping[str, torch.Tensor],
        rewards: Mapping[str, torch.Tensor],
        next_states: Mapping[str, torch.Tensor],
        terminated: Mapping[str, torch.Tensor],
        truncated: Mapping[str, torch.Tensor],
        infos: Mapping[str, Any],
        timestep: int,
        timesteps: int,
    ):
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states,
            actions,
            rewards,
            next_states,
            terminated,
            truncated,
            infos,
            timestep,
            timesteps,
        )

        if self.memories:
            self._current_shared_states = infos["shared_states"]
            self._current_shared_next_states = infos["shared_next_states"]

            for uid in self.possible_agents:
                # reward shaping
                if self._rewards_shaper is not None:
                    rewards = self._rewards_shaper(rewards, timestep, timesteps)

                # storage transition in memory
                # TODO： 这里应该加入每个时刻的联合动作
                self.memories[uid].add_samples(
                    states=states[uid],
                    shared_states=self._current_shared_states[uid],
                    actions=actions[uid],
                    rewards=rewards[uid],
                    next_states=next_states[uid],
                    shared_next_states=self._current_shared_next_states[uid],
                    terminated=terminated[uid],
                    truncated=truncated[uid],
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timestep: Number of timesteps
        :type timesteps: int
        """
        def get_joint_actions():
            pass

        for uid in self.possible_agents:
            policy = self.policies[uid]
            target_policy = self.target_policies[uid]
            critic = self.critics[uid]
            target_critic = self.target_critics[uid]

            policy_optimizer = self.policy_optimizers[uid]
            critic_optimizer = self.critic_optimizers[uid]

            memory = self.memories[uid]

            # gradient steps:
            for gradient_step in range(self._gradient_steps):
                # sample a mini-batch from memory
                (
                    sampled_states,
                    sampled_shared_states,
                    sampled_actions,
                    sampled_joint_actions,
                    sampled_rewards,
                    sampled_next_states,
                    sampled_shared_next_states,
                    sampled_terminated,
                    sampled_truncated,
                ) = memory.sample(names=self._tensor_names, batch_size=self._batch_size)[0]

                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    sampled_states = self._state_preprocessor[uid](sampled_states, train=True)
                    sampled_next_states = self._state_preprocessor[uid](sampled_next_states, train=True)
                    sampled_shared_states = self._shared_state_preprocessor[uid](sampled_shared_states, train=True)
                    sampled_shared_next_states = self._shared_state_preprocessor[uid](sampled_shared_next_states, train=True)
                    with torch.no_grad():
                        # next_actions, _, _ = target_policy.act({"states": sampled_states}, role="target_policy")
                        # TODO: 所有policy根据当前状态计算next_actions
                        target_q_values, _, _ = target_critic.act(
                            {"states": sampled_next_states, "taken_actions": next_actions}, role="target_critic"
                        )
                        target_values = (  # TD target
                            sampled_rewards + self._discount_factor * (sampled_terminated | sampled_truncated).logical_not() * target_q_values
                        )
                    # compute critic loss
                    critic_values, _, _ = critic.act({"states": sampled_states, "taken_actions": sampled_actions}, role="critic")
                    critic_loss = F.mse_loss(critic_values, target_values)

                # optimization step (critic)
                self.critic_optimizers[uid].zero_grad()
                self.scaler.scale(critic_loss).backward()

                if config.torch.is_distributed:
                    critic.reduce_parameters()

                if self._grad_norm_clip > 0:
                    self.scaler.unscale_(self.critic_optimizers[uid])

            #     with torch.autocast(
            #         device_type=self._device_type, enabled=self._mixed_precision
            #     ):
            #         sampled_states = self._state_preprocessor[uid](
            #             sampled_states, train=True
            #         )
            #         sampled_next_states = self._state_preprocessor[uid](
            #             sampled_next_states, train=True
            #         )

            #         # compute target values
            #         with torch.no_grad():
            #             next_actions, _, _ = self.target_policies[uid].act(
            #                 {"states": sampled_next_states}, role="target_policy"
            #             )

            #             target_q_values, _, _ = self.target_critics[uid].act(
            #                 {
            #                     "states": sampled_next_states,
            #                     "taken_actions": next_actions,
            #                 },
            #                 role="target_critic",
            #             )
            #             target_values = (
            #                 sampled_rewards
            #                 + self._discount_factor
            #                 * (sampled_terminated | sampled_truncated).logical_not()
            #                 * target_q_values
            #             )
            #         # compute critic loss
            #         critic_values, _, _ = critic.act(
            #             {"states": sampled_states, "taken_actions": sampled_actions},
            #             role="critic",
            #         )
            #         critic_loss = F.mse_loss(critic_values, target_values)

            #     # optimization step (critic)
            #     self.critic_optimizers[uid].zero_grad()
            #     self.scaler.scale(critic_loss).backward()

            #     if config.torch.is_distributed:
            #         critic.reduce_parameters()

            #     if self._grad_norm_clip > 0:
            #         self.scaler.unscale_(self.critic_optimizers[uid])
            #         nn.utils.clip_grad_norm_(critic.parameters(), self._grad_norm_clip)

            #     self.scaler.step(self.critic_optimizers[uid])

            #     with torch.autocast(
            #         device_type=self._device_type, enabled=self._mixed_precision
            #     ):
            #         # compute policy (actor) loss
            #         actions, _, _ = policy.act(
            #             {"states": sampled_states[uid]}, role="policy"
            #         )
            #         critic_values, _, _ = critic.act(
            #             {"states": sampled_states[uid], "taken_actions": actions},
            #             role="critic",
            #         )

            #         policy_loss = -critic_values.mean()

            #     # optimization step (policy)
            #     self.policy_optimizers[uid].zero_grad()
            #     self.scaler.scale(policy_loss).backward()

            #     if config.torch.is_distributed:
            #         policy.reduce_parameters()

            #     if self._grad_norm_clip > 0:
            #         self.scaler.unscale_(self.policy_optimizers[uid])
            #         nn.utils.clip_grad_norm_(policy.parameters(), self._grad_norm_clip)

            #     self.scaler.step(self.policy_optimizers[uid])

            #     self.scaler.update()  # called once, after optimizers have been stepped

            #     # update target networks
            #     self.target_policies[uid].update_parameters(policy, polyak=self._polyak)
            #     self.target_critics[uid].update_parameters(critic, polyak=self._polyak)

            #     # update learning rate
            #     if self._learning_rate_scheduler[uid]:
            #         self.policy_schedulers[uid].step()
            #         self.critic_schedulers[uid].step()

            # # record data
            # self.track_data(
            #     f"Loss / Policy loss ({uid})",
            # )
            # self.track_data(f"")
