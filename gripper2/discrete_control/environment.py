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
from objects import Ball, Floor, Poly, Walls
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
        self.walls = Walls(self.space)
        self.vertex = vertex

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        # Define action space
        self.action_space = spaces.Discrete(13)

        # Define 8D (continuous) observation space
        high = np.array([np.inf] * 30, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 300
        self.max_steps = 1500
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
        self.walls = Walls(self.space)

        # self.gripper.left_finger1.body.angle = np.random.uniform(-1, 1)
        # self.gripper.right_finger1.body.angle = np.random.uniform(-1, 1)
        # self.gripper.left_finger2.body.angle = np.random.uniform(-1, 1)
        # self.gripper.right_finger2.body.angle = np.random.uniform(-1, 1)

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
        elif action == 4:   # Open gripper left finger 1
            rotation1 = 0.01
            self.gripper.left_finger1.body.angle -= rotation1 if self.gripper.left_finger1.body.angle > -0.5 else 0.0
        elif action == 5:   # Open gripper right finger 1
            rotation1 = 0.01
            self.gripper.right_finger1.body.angle += rotation1 if self.gripper.right_finger1.body.angle < 0.5 else 0.0
        elif action == 6:   # Close gripper left finger 1
            rotation1 = -0.01
            self.gripper.left_finger1.body.angle -= rotation1 if self.gripper.left_finger1.body.angle < 0.7 else 0.0
        elif action == 7:   # Close gripper right finger 1
            rotation1 = -0.01
            self.gripper.right_finger1.body.angle += rotation1 if self.gripper.right_finger1.body.angle > -0.7 else 0.0
        elif action == 8:   # Open gripper left finger 2
            rotation2 = 0.01
            self.gripper.left_finger2.body.angle -= rotation2 if self.gripper.left_finger2.body.angle > -0.5 else 0.0
        elif action == 9:   # Open gripper right finger 2
            rotation2 = 0.01
            self.gripper.right_finger2.body.angle += rotation2 if self.gripper.right_finger2.body.angle < 0.5 else 0.0
        elif action == 10:  # Close gripper left finger 2
            rotation2 = -0.01
            self.gripper.left_finger2.body.angle -= rotation2 if self.gripper.left_finger2.body.angle < 1.5 else 0.0
        elif action == 11:  # Close gripper right finger 2
            rotation2 = -0.01
            self.gripper.right_finger2.body.angle += rotation2 if self.gripper.right_finger2.body.angle > -1.5 else 0.0
        elif action == 12:   # Do nothing
            None

        # Apply the action to the gripper
        self.gripper.arm.body.velocity = (dx, dy)

        # Step the simulation
        self.space.step(1/self.FPS)
        self.current_step += 1

        # Get the observation
        observation = self.get_observation()

        # Get the reward
        reward, done, success = self.get_reward(observation)

        self.gripper.arm.body.velocity = (0, 0)

        truncated = False
        info = {'success': success}

        return observation, reward, done, truncated, info

    def get_observation(self):

        def tip_to_surface_distance(shape, tip) -> float:
            """
            Signed distance from tip to the *outside* of the shape.
            • positive => tip is outside
            • 0        => tip is exactly on the surface
            • negative => tip is poking inside (rare but possible after penetration)
            Works for both Circles and Polys through Shape.point_query().
            """
            return shape.point_query(tip).distance

        base = self.gripper.base.body
        obj = self.object.body

        # Object relative to base
        rel_obj_pos = obj.position - base.position
        rel_obj_vel = obj.velocity - base.velocity

        # Object orientation
        obj_orient = (np.cos(obj.angle), np.sin(obj.angle))

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

        # Distance from fingertip to object surface
        d_l = tip_to_surface_distance(self.object.shape, l_tip)
        d_r = tip_to_surface_distance(self.object.shape, r_tip)
        d_c = tip_to_surface_distance(self.object.shape, (l_tip + r_tip) * 0.5)

        # Touch BOOLs
        l_touch = 1.0 if self.gripper.left_finger2.shape.shapes_collide(self.object.shape).points else 0.0
        r_touch = 1.0 if self.gripper.right_finger2.shape.shapes_collide(self.object.shape).points else 0.0

        obs = np.array([ obj.position.x, obj.position.y,
                                rel_obj_pos.x, rel_obj_pos.y,
                                rel_obj_vel.x, rel_obj_vel.y,
                                *finger_feats,                  # 12 values
                                l_tip_rel.x, l_tip_rel.y,
                                r_tip_rel.x, r_tip_rel.y,
                                gap,
                                l_touch, r_touch,
                                obj_orient[0], obj_orient[1],
                                d_l, d_r, d_c,
                                ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        r1, r2, r3 = 0,0,0

        # Reward based on height of object if gripper is moving up as well
        r1 = max(self.object.body.position[1] - 100, 0) if (obs[-6] or obs[-7]) else 0 # self.gripper.arm.body.velocity[1] > 0

        # Reward if left fingertip distance within threshold
        l_tip_dist = np.linalg.norm(self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b) - self.object.body.position)
        r_tip_dist = np.linalg.norm(self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b) - self.object.body.position)

        if l_tip_dist < 50:
            r2 = 1

        if r_tip_dist < 50:
            r3 = 1

        reward = r1 + r2 + r3

        # Touch Penalties
        a = 1 if self.gripper.left_finger2.shape.shapes_collide(self.gripper.right_finger2.shape).points else 0.0
        b = 1 if self.gripper.left_finger2.shape.shapes_collide(self.gripper.right_finger1.shape).points else 0.0
        c = 1 if self.gripper.left_finger1.shape.shapes_collide(self.gripper.right_finger2.shape).points else 0.0

        # Penalty for fingers below floor
        d = 1 if self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)[1] < self.floor.shape.a[1] - 10 else 0
        e = 1 if self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)[1] < self.floor.shape.a[1] - 10 else 0

        reward -= (a+b+c+d+e)

        done = False
        success = False

        # Reward based on success
        if self.object.body.position.y > self.pickup_height and self.gripper.base.body.position.y > self.pickup_height:
            reward += 10
            print("Success!")
            success = True
            done = True

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        return reward, done, success

    def render(self, mode="human"):

        # draw physics into your off-screen self.surface
        self.surface.fill((255, 255, 255))
        self.object.draw()
        self.floor.draw()
        self.gripper.draw()
        self.walls.draw()

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
    vertex = [(-30, -30), (30, -30), (0, 30)]

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
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=100000, render=True, verbose=0, deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=512,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.001
    )

    model.learn(total_timesteps=2000000, callback=eval_callback)
    model.save("models/ppo_pymunk_gripper")
    train_env.save("normalise_stats/vecnormalize_stats.pkl")
    print("Training complete and model saved.")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/

