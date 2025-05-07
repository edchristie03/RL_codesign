import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from main import Gripper, Ball, Floor, Poly
import main

class Environment(gym.Env):

    def __init__(self, vertex, render_mode=None):

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
        self.floor = Floor(self.space, 20)
        self.vertex = vertex

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        # Define action space
        self.action_space = spaces.Discrete(7)

        # Define (continuous) observation space
        high = np.array([np.inf] * 8, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
        self.max_steps = 500
        self.current_step = 0

    def reset(self, seed=None, options=None):

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper(self.space)
        self.floor = Floor(self.space, radius=20)

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

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
            dx = -100
        elif action == 1:   # Move right
            dx = 100
        elif action == 2:   # Move up
            dy = 100
        elif action == 3:   # Move down
            dy = -100
        elif action == 4:   # Open gripper
            rotation = 0.1 if self.gripper.left_finger.body.angle > -0.5 else 0.0
        elif action == 5:   # Close gripper
            rotation = -0.1 if self.gripper.left_finger.body.angle < 1 else 0.0
        elif action == 6:   # Do nothing
            None

        # Apply the action to the gripper
        self.gripper.arm.body.velocity = (dx, dy)
        self.gripper.left_finger.body.angle -= rotation
        self.gripper.right_finger.body.angle += rotation

        # Step the simulation
        self.space.step(1/self.FPS)

        self.current_step += 1

        # Get the observation
        observation = self.get_observation()

        # Get the reward
        reward, done = self.get_reward(observation)

        self.gripper.arm.body.velocity = (0, 0)

        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def get_observation(self):

        # print("Getting observation...")

        # Base position
        bx, by = self.gripper.base.body.position

        # Gripper angles and angular velocities
        left = self.gripper.left_finger.body
        right = self.gripper.right_finger.body
        la, lav = left.angle, left.angular_velocity
        ra, rav = right.angle, right.angular_velocity

        # Object relative position
        object = self.object.body
        rel_pos = object.position - self.gripper.base.body.position

        obs = np.array([bx, by, la, lav, ra, rav, rel_pos.x, rel_pos.y], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        # Reward based on height of object if gripper is moving up as well
        r2 = self.object.body.position[1] - 100 if self.gripper.arm.body.velocity[1] > 0 else 0

        # Reward based on distance of left finger tip to the object bottom
        r3 = - np.linalg.norm(self.gripper.left_finger.body.local_to_world(self.gripper.left_finger.shape.b) - (self.object.body.position - (0, 30)))

        # Reward based on distance of right finger tip to the object bottom
        r4 = - np.linalg.norm(self.gripper.right_finger.body.local_to_world(self.gripper.right_finger.shape.b) - (self.object.body.position - (0, 30)))

        reward = r2  + r3 + r4

        done = False

        # Reward based on success
        if self.object.body.position.y > self.pickup_height and self.gripper.base.body.position.y > self.pickup_height:
            reward += 50
            print("Success!")
            done = True

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        # End if left finger tip is below the floor
        if self.gripper.left_finger.body.local_to_world(self.gripper.left_finger.shape.b)[1] < self.floor.shape.a[1] - 10:
            done = True

        # End if object is below the floor
        if self.object.body.position.y < self.floor.shape.a[1] - 10:
            done = True

        return reward, done

    def render(self, mode="human"):

        # draw physics into your off-screen self.surface
        self.surface.fill((255, 255, 255))
        self.object.draw()
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

    # This determines the shape of the object to be picked up. If empty, a ball is created with radius 30
    vertex = [(-30, -30), (30, -30), (30, 30), (-30, 30)]

    # Training env (headless)
    train_env = DummyVecEnv([lambda: Monitor(Environment(vertex, render_mode=None))])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Evaluation env (human‐render)
    eval_env = DummyVecEnv([lambda: Monitor(Environment(vertex, render_mode="human"))])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)

    # Copy the running mean/var from the training env:
    eval_env.obs_rms = train_env.obs_rms

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=100000, render=True, verbose=0, deterministic=False)

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
    model.save("models/ppo_pymunk_gripper")
    train_env.save("normalise_stats/vecnormalize_stats.pkl")
    print("Training complete and model saved.")


# tensorboard --logdir ./ppo_gripper_tensorboard/

