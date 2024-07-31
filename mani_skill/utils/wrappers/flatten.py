import copy
from typing import Dict

import gymnasium as gym
import gymnasium.spaces.utils
import numpy as np
import torch
from gymnasium.vector.utils import batch_space

from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.utils import common, gym_utils


class FlattenRGBDObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"
    """

    def __init__(self, env, rgb_only=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.rgb_only = rgb_only



        new_obs = self.observation(self.base_env._init_raw_obs)

        
        self.base_env.update_obs_space(new_obs)


    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            images.append(cam_data["rgb"])
            if not self.rgb_only:
                images.append(cam_data["depth"])
        # print(images[1].shape)
        # assert 0
        images = torch.concat(images, axis=-1)
        # flatten the rest of the data which should just be state data

        observation = common.flatten_state_dict(observation, use_torch=True)
        if self.rgb_only:
            return dict(state=observation, rgb=images)
        else:
            return dict(state=observation, rgbd=images)
        
class FlattenRGBDObservationAsyncWrapper(gym.ObservationWrapper):
    """
    Flattens the rgbd mode observations into a dictionary with two keys, "rgbd" and "state"
    """

    def __init__(self, env, rgb_only=False) -> None:
        self.base_env: BaseEnv = env.unwrapped
        super().__init__(env)
        self.rgb_only = rgb_only

        obs, _ = self.base_env.reset(seed=2022, options=dict(reconfigure=True))
        self._init_raw_obs = common.to_cpu_tensor(obs)

        new_obs = self.observation(self._init_raw_obs)

        self.base_env.single_observation_space = gym_utils.convert_observation_to_space(common.to_numpy(new_obs), unbatched=True)
        self.base_env.observation_space = batch_space(self.single_observation_space, n=self.num_envs)

        #self.base_env.update_obs_space(new_obs)


    def observation(self, observation: Dict):
        sensor_data = observation.pop("sensor_data")
        del observation["sensor_param"]
        images = []
        for cam_data in sensor_data.values():
            images.append(cam_data["rgb"])
            if not self.rgb_only:
                images.append(cam_data["depth"])
        # print(images[1].shape)
        # assert 0
        # print("==============================================", images[0].shape, images[0].dtype)
        if not isinstance(images[0], torch.Tensor):
            images = [torch.tensor(img) for img in images]

        images = torch.concat(images, axis=-1).squeeze(1)

        # flatten the rest of the data which should just be state data
        observation['agent']['qpos'] = torch.tensor(observation['agent']['qpos'].squeeze(1))
        observation['agent']['qvel'] = torch.tensor(observation['agent']['qvel'].squeeze(1))
        observation['extra']['tcp_pose'] = torch.tensor(observation['extra']['tcp_pose'].squeeze(1))

        # agent, extra
        observation = common.flatten_state_dict(observation, use_torch=True)
        if self.rgb_only:
            return dict(state=observation, rgb=images)
        else:
            return dict(state=observation, rgbd=images)


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens the observations into a single vector
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self.base_env.update_obs_space(
            common.flatten_state_dict(self.base_env._init_raw_obs)
        )

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def observation(self, observation):
        return common.flatten_state_dict(observation, use_torch=True)


class FlattenActionSpaceWrapper(gym.ActionWrapper):
    """
    Flattens the action space. The original action space must be spaces.Dict
    """

    def __init__(self, env) -> None:
        super().__init__(env)
        self._orig_single_action_space = copy.deepcopy(
            self.base_env.single_action_space
        )
        self.single_action_space = gymnasium.spaces.utils.flatten_space(
            self.base_env.single_action_space
        )
        if self.base_env.num_envs > 1:
            self.action_space = batch_space(
                self.single_action_space, n=self.base_env.num_envs
            )
        else:
            self.action_space = self.single_action_space

    @property
    def base_env(self) -> BaseEnv:
        return self.env.unwrapped

    def action(self, action):
        if (
            self.base_env.num_envs == 1
            and action.shape == self.single_action_space.shape
        ):
            action = common.batch(action)

        # TODO (stao): This code only supports flat dictionary at the moment
        unflattened_action = dict()
        start, end = 0, 0
        for k, space in self._orig_single_action_space.items():
            end += space.shape[0]
            unflattened_action[k] = action[:, start:end]
            start += space.shape[0]
        return unflattened_action
