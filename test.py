import os
import retro
import time
import numpy as np
import cv2
import torch

from gym import Env
from gym.spaces import Box, MultiBinary

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


class StreetFighter(Env):
    def __init__(self):
        super().__init__()
        self.observation_space = Box(low=0, high=255, shape=(84,84,1), dtype=np.uint8)
        self.action_space = MultiBinary(12)
        self.game = retro.make(
            game='StreetFighterIISpecialChampionEdition-Genesis',
            use_restricted_actions=retro.Actions.FILTERED
        )

    def step(self, action):
        obs, reward, done, info = self.game.step(action)
        obs = self.preprocess(obs)
        
        # Frame delta
        frame_delta = obs - self.previous_frame
        self.previous_frame = obs

        # Reward shaping (based on score delta)
        reward = info.get('score', 0) - self.score
        self.score = info.get('score', 0)

        return frame_delta, reward, done, info

    def reset(self):
        obs = self.game.reset()
        obs = self.preprocess(obs)
        self.previous_frame = obs
        self.score = 0
        return obs

    def render(self, mode = 'human'):
        self.game.render()

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (84, 84, 1))

    def close(self):
        self.game.close()

LOG_DIR = './logs/'
env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

model = PPO.load('./opt/trial_5_best_model.zip')
# mean_reward, _ = evaluate_policy(model, env,render = True,  n_eval_episodes=1)


obs = env.reset()
for game in range(1):
    done = False
    while not done:
        env.render()
        
        action, _ = model.predict(obs)  # ✅ get action
        obs, reward, done, info = env.step(action)
        
        #time.sleep(0.0001)
        print(reward)
        
    obs = env.reset()  # reset after game ends


# model.policy.to("cpu")

# # CHW format
# dummy = torch.zeros((1, 4, 84, 84)).float()

# torch.onnx.export(
#     model.policy,
#     dummy,
#     "streetfighter_ppo.onnx",
#     opset_version=13,  # <-- FIXED
#     input_names=['input'],
#     output_names=['output'],
#     dynamic_axes={
#         'input': {0: 'batch'},
#         'output': {0: 'batch'}
#     }
# )

# print("ONNX model exported successfully!")
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("tb_custom")

# After each episode / evaluation
writer.add_histogram("episode_rewards", rewards_array, global_step)


