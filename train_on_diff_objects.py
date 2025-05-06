import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from main import Gripper, Ball, Floor
import main
from training import Environment

vertices = [[], [(-25, 0), (25, 0), (25, 50), (-25, 50)] ,[(-25, 0), (25, 0), (25, 50)], [(-25, 0), (25, 0), (0, 50)]]

for vertex in vertices:

    # Training env (headless)
    train_env = DummyVecEnv([lambda: Monitor(Environment(render_mode=None))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Evaluation env (human‐render)
    eval_env = DummyVecEnv([lambda: Monitor(Environment(render_mode="human"))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # Copy the running mean/var from the training env:
    eval_env.obs_rms = train_env.obs_rms

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=50000, render=True, verbose=0,
                                 deterministic=False)

    # Define the policy network architecture
    policy_kwargs = dict(net_arch=[256, 256])

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.02,
    )

    model.learn(total_timesteps=1000000, callback=eval_callback)
    model.save("ppo_pymunk_gripper")
    train_env.save("vecnormalize_stats.pkl")
    print("Training complete and model saved.")

