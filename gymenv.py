import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from main import Gripper, Ball, Floor
import main

class Environment(gym.Env):

    metadata = {"render_modes": ["human"],
                "render_fps": 60}

    def __init__(self, render_mode=None):
        # super().__init__(render_mode=render_mode)

        # print("Initializing environment...")

        self.render_mode = render_mode

        pygame.init()
        self.surface = pygame.Surface((800, 800))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.space.gravity = (0, -1000)  # gravity
        self.FPS = 200

        # Quick HACK for display. Should refactor main
        main.display = self.surface

        # Define objects in the environment
        self.gripper = Gripper(self.space)
        self.ball = Ball(self.space, radius=30)
        self.floor = Floor(self.space, radius=20)

        # Define action space
        self.action_space = spaces.Discrete(7)

        # Define (continuous) observation space
        high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 600
        self.max_steps = 400
        self.current_step = 0

    def reset(self, seed=None, options=None):

        # print("Resetting environment...")

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper(self.space)
        self.ball = Ball(self.space, radius=30)
        self.floor = Floor(self.space, radius=10)
        self.current_step = 0

        # initial observation and info
        obs = self.get_observation()
        info = {}
        return obs, info


    def step(self, action):

        # print("Action taken this step:", action)

        pygame.event.pump()
        dx = dy = 0
        rotation = 0.0

        if action == 0:     # Move left
            dx = -1
        elif action == 1:   # Move right
            dx = 1
        elif action == 2:   # Move up
            dy = 1
        elif action == 3:   # Move down
            dy = -1
        elif action == 4:   # Open gripper
            rotation = 0.1 if self.gripper.left_finger.body.angle > -0.5 else 0.0
        elif action == 5:   # Close gripper
            rotation = -0.1 if self.gripper.left_finger.body.angle < 0.8 else 0.0
        elif action == 6:   # Do nothing
            None

        # Apply the action to the gripper
        self.gripper.base.body.position += (dx, dy)
        self.gripper.left_finger.body.angle -= rotation
        self.gripper.right_finger.body.angle += rotation

        # Step the simulation
        self.space.step(1/self.FPS)
        self.current_step += 1

        # Get the observation
        observation = self.get_observation()

        # Get the reward
        reward, done = self.get_reward(observation)

        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def get_observation(self):

        # print("Getting observation...")

        # Base position and velocity
        bx, by = self.gripper.base.body.position
        # bvx, bvy = self.gripper.base.body.velocity

        # Gripper angles and angular velocities
        left = self.gripper.left_finger.body
        right = self.gripper.right_finger.body
        la, lav = left.angle, left.angular_velocity
        ra, rav = right.angle, right.angular_velocity

        # Ball relative position
        ball = self.ball.body
        rel_pos = ball.position - self.gripper.base.body.position

        obs = np.array([bx, by, la, lav, ra, rav, rel_pos.x, rel_pos.y], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        r1 = - np.linalg.norm(obs[6:8])

        # Reward based on height of ball
        r2 = self.ball.body.position[1]

        # Reward based on distance of left finger tip to the ball
        r3 = - np.linalg.norm(self.gripper.left_finger.body.local_to_world(self.gripper.left_finger.shape.b) - self.ball.body.position)

        # Reward based on distance of right finger tip to the ball
        r4 = - np.linalg.norm(self.gripper.right_finger.body.local_to_world(self.gripper.right_finger.shape.b) - self.ball.body.position)

        reward =  r3 + r4 + r2

        # print(f"Reward: {reward:.2f} (r1: {r1:.2f}, r3: {r3:.2f}, r4: {r4:.2f})")

        done = False

        # Reward based on success
        if self.ball.body.position.y > self.pickup_height:
            reward += 1000000
            done = True

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        # End if left finger tip is below the floor
        if self.gripper.left_finger.body.local_to_world(self.gripper.left_finger.shape.b)[1] < self.floor.shape.a[1] - 10:
            # reward -= 1000000
            done = True

        return reward, done

    def render(self, mode="human"):

        # print("Rendering...")

        # draw physics into your off-screen self.surface
        self.surface.fill((255, 255, 255))
        self.ball.draw()
        self.floor.draw()
        self.gripper.draw()

        if mode == "human":
            # first time only, open the OS window
            if not hasattr(self, "_window"):
                pygame.display.init()
                self._window = pygame.display.set_mode((800, 800))
            # pump to not fill up memory
            pygame.event.pump()
            # blit & flip
            self._window.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return self.surface

    def close(self):
        pygame.quit()

if __name__ == "__main__":

    # Training env (headless)
    train_env = DummyVecEnv([lambda: Monitor(Environment(render_mode=None))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Evaluation env (human‐render)
    eval_env = DummyVecEnv([lambda: Monitor(Environment(render_mode="human"))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Copy the running mean/var from the training env:
    eval_env.obs_rms = train_env.obs_rms

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=10000, render=True, verbose=0, deterministic=True)

    # Define the policy network architecture
    policy_kwargs = dict(net_arch=[128, 128])

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.03,
    )

    model.learn(total_timesteps=500000, callback=eval_callback)
    model.save("ppo_pymunk_gripper")
    train_env.save("vecnormalize_stats.pkl")
    print("Training complete and model saved.")

















