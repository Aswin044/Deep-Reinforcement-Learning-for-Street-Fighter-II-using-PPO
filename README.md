🔥 Overview

This project implements a Deep Reinforcement Learning (DRL) agent using Proximal Policy Optimization (PPO) to play Street Fighter II: Special Champion Edition.
It includes a custom Gym Retro environment, optimized preprocessing pipeline, reward shaping, hyperparameter tuning using Optuna, and a fully modular interpretability framework to understand the agent’s behavior.

This work is based on the research paper included in this repository.

🎮 Key Features

Custom Gym Retro Environment for Street Fighter II

Frame Preprocessing Pipeline

Grayscale conversion

Downsampling to 84×84

Frame stacking (4-frame state)

Reward Shaping based on score differences

Stable PPO Training with GAE and clipped objective

Hyperparameter Optimization using Optuna

Glass-Box Interpretability Tools:

Action usage histograms

Reward distribution

Reward progression plots

Policy / value loss curves

Emergent RL Behaviors: spacing, zoning, jump-attacks, punish patterns

🧠 Technical Summary
Observation Pipeline

Converts raw game frames to grayscale

Normalizes and resizes to 84×84

Stacks 4 frames to form temporal context

Action Space

Multi-binary vector representing 12 possible controller actions

Supports simultaneous button presses (jump + punch, etc.)

Reward Shaping
reward(t) = score(t) − score(t−1)


Fallback rewards for hits, damage, and round wins.

PPO Training

Clipped surrogate objective

GAE advantage estimation

Adam optimizer

Gradient clipping

Entropy regularization
