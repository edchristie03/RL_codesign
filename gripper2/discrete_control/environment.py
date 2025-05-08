import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from grippers import Gripper2
from objects import Ball, Floor, Poly
import objects, grippers

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
        objects.display = self.surface
        grippers.display = self.surface

        # Define objects in the environment
        self.gripper = Gripper2(self.space)
        self.floor = Floor(self.space, 20)
        self.vertex = vertex

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        # Define action space
        self.action_space = spaces.Discrete(9)

        # Define 8D (continuous) observation space
        high = np.array([np.inf] * 23, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper2(self.space)
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

        pygame.event.pump()
        dx = dy = 0

        if action == 0:     # Move left
            dx = -100
        elif action == 1:   # Move right
            dx = 100
        elif action == 2:   # Move up
            dy = 100
        elif action == 3:   # Move down
            dy = -100
        elif action == 4:   # Open gripper finger 1
            rotation1 = 0.1
            self.gripper.left_finger1.body.angle -= rotation1 if self.gripper.left_finger1.body.angle > -0.5 else 0.0
            self.gripper.right_finger1.body.angle += rotation1 if self.gripper.right_finger1.body.angle < 0.5 else 0.0
        elif action == 5:   # Close gripper finger 1
            rotation1 = -0.1
            self.gripper.left_finger1.body.angle -= rotation1 if self.gripper.left_finger1.body.angle < 0.7 else 0.0
            self.gripper.right_finger1.body.angle += rotation1 if self.gripper.right_finger1.body.angle > -0.7 else 0.0
        elif action == 6:   # Open gripper finger 2
            rotation2 = 0.1
            self.gripper.left_finger2.body.angle -= rotation2 if self.gripper.left_finger2.body.angle > -0.5 else 0.0
            self.gripper.right_finger2.body.angle += rotation2 if self.gripper.right_finger2.body.angle < 0.5 else 0.0
        elif action == 7:  # Close gripper finger 2
            rotation2 = -0.1
            self.gripper.left_finger2.body.angle -= rotation2 if self.gripper.left_finger2.body.angle < 1.5 else 0.0
            self.gripper.right_finger2.body.angle += rotation2 if self.gripper.right_finger2.body.angle > -1.5 else 0.0
        elif action == 8:   # Do nothing
            None

        # Apply the action to the gripper
        self.gripper.arm.body.velocity = (dx, dy)

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

        base = self.gripper.base.body
        obj = self.object.body

        # Object relative to base
        rel_obj_pos = obj.position - base.position
        rel_obj_vel = obj.velocity - base.velocity

        # Finger Angles and angular velocities
        fingers = [self.gripper.left_finger1.body, self.gripper.left_finger2.body, self.gripper.right_finger1.body, self.gripper.right_finger2.body]
        finger_feats = []

        for j in fingers:
            finger_feats.extend([np.cos(j.angle), np.sin(j.angle), j.angular_velocity])

        # Fingertip positions
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)
        l_tip_rel = l_tip - obj.position
        r_tip_rel = r_tip - obj.position
        gap = np.linalg.norm(l_tip - r_tip) / 200

        # Touch BOOLs
        l_touch = 1.0 if self.gripper.left_finger2.shape.shapes_collide(self.object.shape).points else 0.0
        r_touch = 1.0 if self.gripper.right_finger2.shape.shapes_collide(self.object.shape).points else 0.0

        obs = np.array([ rel_obj_pos.x, rel_obj_pos.y,
                                rel_obj_vel.x, rel_obj_vel.y,
                                *finger_feats,                  # 12 values
                                l_tip_rel.x, l_tip_rel.y,
                                r_tip_rel.x, r_tip_rel.y,
                                gap,
                                l_touch, r_touch
                                ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        # Reward based on height of object if gripper is moving up as well
        r2 = self.object.body.position[1] - 100 if self.gripper.arm.body.velocity[1] > 0 else 0

        # Reward based on distance of left finger tip to the object bottom
        r3 = - np.linalg.norm(self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b) - (self.object.body.position - (0, 30)))

        # Reward based on distance of right finger tip to the object bottom
        r4 = - np.linalg.norm(self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b) - (self.object.body.position - (0, 30)))

        reward = r2 + r3 + r4

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
        if self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)[1] < self.floor.shape.a[1] - 10:
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

    N_ENVS = 8  # Number of parallel environments

    # This determines the shape of the object to be picked up. If empty, a ball is created with radius 30
    vertex = []#[(-25, -25), (25, -25), (25, 25), (-25, 25)]

    # Define the policy network architecture
    policy_kwargs = dict(net_arch=[256, 256])

    def make_env(vertex, rank, render=False):
        """
        Factory that creates a *fresh* environment in its own process.
        `rank` is only used if you want per‑worker seeding or logging.
        """

        def _init():
            env = Environment(vertex, render_mode="human" if render else None)
            env = Monitor(env)  # keeps episode stats
            return env

        return _init


    # Training envs (headless, parallel)
    train_env = SubprocVecEnv([make_env(vertex, i) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True)

    # Evaluation env (still a single window so you can watch)
    eval_env = DummyVecEnv([make_env(vertex, 0, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True, training=False)
    eval_env.obs_rms = train_env.obs_rms  # share running stats

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=5000, render=True, verbose=0, deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=512,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.02
    )

    model.learn(total_timesteps=1000000, callback=eval_callback)
    model.save("models/ppo_pymunk_gripper")
    train_env.save("normalise_stats/vecnormalize_stats.pkl")
    print("Training complete and model saved.")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/

