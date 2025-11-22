# Imports
import os
import retro
import time
import numpy as np
import cv2

from gym import Env
from gym.spaces import Box, MultiBinary

import optuna
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy


# Logging directories
LOG_DIR = './logs/'
OPT_DIR = './opt/'
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OPT_DIR, exist_ok=True)

# Custom Street Fighter environment
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

    def render(self):
        self.game.render()

    def preprocess(self, observation):
        gray = cv2.cvtColor(observation, cv2.COLOR_BGR2GRAY)
        resize = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_CUBIC)
        return np.reshape(resize, (84, 84, 1))

    def close(self):
        self.game.close()


# Hyperparameter search space
def optimize_ppo(trial):
    return {
        'n_steps': trial.suggest_int('n_steps', 2048, 8192),
        'gamma': trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99)
    }


# Training + evaluation function
def optimize_agent(trial):
    env = None
    try:
        params = optimize_ppo(trial)

        # Setup environment
        env = StreetFighter()
        env = Monitor(env, LOG_DIR)
        env = DummyVecEnv([lambda: env])
        env = VecFrameStack(env, 4, channels_order='last')

        # PPO model
        model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR, **params)
        model.learn(total_timesteps=30000)  # training

        # Evaluate (without forcing render)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=1)

        # Save model
        save_path = os.path.join(OPT_DIR, f'trial_{trial.number}_best_model')
        model.save(save_path)
        print(f"Trial {trial.number} done. Mean reward: {mean_reward}")

        return mean_reward
    except Exception as e:
        print(f"Exception: {e}")
        return -1000
    finally:
        if env is not None:
            env.close()


# Main optimization + testing
if __name__ == "__main__":
    # Run optimization (just 1 trial for testing)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=10, n_jobs=1)

    # After optimization, load and test the saved model
    model_path = os.path.join(OPT_DIR, "trial_0_best_model")
    if os.path.exists(model_path + ".zip"):  # SB3 saves with .zip automatically
        model = PPO.load(model_path)
        print("Model loaded successfully!")

        # Create environment for testing
        # test_env = StreetFighter()
        # test_env = DummyVecEnv([lambda: test_env])
        # test_env = VecFrameStack(test_env, 4, channels_order='last')

        # obs = test_env.reset()
        # for _ in range(5000):  # Play for 5000 steps
        #     action, _ = model.predict(obs, deterministic=True)
        #     obs, reward, done, info = test_env.step(action)
        #     test_env.render()  # ✅ Explicit render only during testing
        #     if done:
        #         obs = test_env.reset()

        # test_env.close()
    else:
        print("Model file not found!")

    print("Files in opt directory:", os.listdir(OPT_DIR))

model = PPO.load(os.path.join(OPT_DIR, 'trial_0_best_model.zip'))

#Import base callback
from stable_baselines3.common.callbacks import BaseCallback

class TrainandLoggingCallback(BaseCallback):
#check_freq for every 10000 steps we save the model
#save_path saves the path
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainandLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok = True)
    
    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, 'best_model_{}'.format(self.n_calls))
            self.model.save(model_path)

        return True

CHECKPOINT_DIR = './train/'
callback = TrainandLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

env = StreetFighter()
env = Monitor(env, LOG_DIR)
env = DummyVecEnv([lambda: env])
env = VecFrameStack(env, 4, channels_order='last')

params = study.best_params
params['n_steps'] = 7488
params  

model = PPO('CnnPolicy', env, verbose=0, tensorboard_log=LOG_DIR, **params)
model.load(os.path.join(OPT_DIR, 'trial_0_best_model.zip'))
model.learn(total_timesteps=100000)


 