import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pymunk_experiments_push.grippers import Gripper
from pymunk_experiments_push.objects import Ball, Floor, Poly, Walls
from pymunk_experiments_push import objects, grippers

class Environment(gym.Env):

    def __init__(self, vertex, training=True, render_mode=None):

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
        self.gripper = Gripper(self.space)
        self.floor = Floor(self.space, 20)
        self.walls = Walls(self.space)
        self.vertex = vertex

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        # Define action space
        self.action_space = spaces.Discrete(2)

        # Define 8D (continuous) observation space
        high = np.array([np.inf] * 15, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.target = 700
        self.max_steps = 1500
        self.current_step = 0
        self.training = training

    def reset(self, seed=None, options=None):

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper(self.space)
        self.floor = Floor(self.space, radius=20)
        self.walls = Walls(self.space)

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
        dx = 0
        move = 200

        if action == 0:     # Move left
            dx = -move
        elif action == 1:   # Move right
            dx = move

        # Apply the action to the gripper
        self.gripper.arm.body.velocity = (dx, 0)

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

        base = self.gripper.base.body
        obj = self.object.body

        # 1. OBJECT PROPERTIES (8 features)
        # Basic object state
        obj_pos = np.array([obj.position.x, obj.position.y])
        obj_vel = np.array([obj.velocity.x, obj.velocity.y])
        obj_orient = np.array([np.cos(obj.angle), np.sin(obj.angle)])
        obj_ang_vel = obj.angular_velocity

        # Object height above floor
        floor_y = self.floor.shape.a[1] + self.floor.shape.radius
        obj_height = obj.position.y - floor_y

        # 2. GRIPPER STATE (15 features)
        base_pos = np.array([base.position.x, base.position.y])
        finger = self.gripper.left_finger.body

        finger_angle = finger.angle
        finger_ang_vel = finger.angular_velocity

        # Gripper aperture and fingertip positions
        l_tip = self.gripper.left_finger.body.local_to_world(self.gripper.left_finger.shape.b)
        l_tip_rel = l_tip - obj.position

        # 3. CONTACT & GRASP QUALITY METRICS (12 features)
        # Basic contact detection
        l_collision = self.gripper.left_finger.shape.shapes_collide(self.object.shape)
        l_contact = 1.0 if l_collision.points else 0.0

        obs = np.concatenate([
            # object props
            obj_pos,  # 2
            obj_vel,  # 2
            obj_orient,  # 2
            [obj_ang_vel],  # 1
            [obj_height],  # 1

            # gripper state
            base_pos,  # 2
            [finger_angle],  # 1
            [finger_ang_vel],  # 1
            l_tip_rel,  # 2

            # contact
            [l_contact],  # 1
        ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        reward = -1

        start_distance = self.target - 400

        additional_reward = (self.object.body.position.x - 400) / start_distance

        reward += additional_reward

        done = False
        success = False

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            # Reward based on success

        if self.object.body.position.x > self.target:
            success = True
            done = True
            print("Success!")

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
    vertex =  [] #[(-30, -30), (30, -30), (0, 30)]

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [126, 126], "log_std_init": 2}

    def make_env(vertex, rank, render=False):
        """
        Factory that creates a *fresh* environment in its own process.
        `rank` is only used if you want per‑worker seeding or logging.
        """

        def _init():
            env = Environment(vertex, training=True, render_mode="human" if render else None)
            env = Monitor(env)  # keeps episode stats
            return env

        return _init

    class SaveBestWithStats(EvalCallback):
        def __init__(self, *args, vecnormalize, **kwargs):
            super().__init__(*args, **kwargs)
            self.vecnormalize = vecnormalize  # the training wrapper

        def _on_step(self) -> bool:
            # run the usual evaluation logic
            old_reward = self.best_mean_reward
            continue_training = super()._on_step()

            if self.best_mean_reward > old_reward:  # new best just saved
                self.vecnormalize.save(f"normalise_stats/vecnormalize_stats_best.pkl")

            return continue_training

    # Training envs (headless, parallel)
    train_env = SubprocVecEnv([make_env(vertex, i) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env_fast = DummyVecEnv([make_env(vertex, 0, render=False)])
    eval_env_fast = VecNormalize(eval_env_fast, norm_obs=True, norm_reward=False, training=False)
    eval_env_fast.obs_rms = train_env.obs_rms

    best_ckpt = SaveBestWithStats(
        eval_env_fast,
        vecnormalize=train_env,
        best_model_save_path=f"models/ppo_pymunk_gripper_best",
        n_eval_episodes=5,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Evaluation env (still a single window so you can watch)
    eval_env = DummyVecEnv([make_env(vertex, 0, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms  # share running stats

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=10000, render=True, verbose=0,
                                 deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=512,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=5e-3,
    )

    model.learn(total_timesteps=1000000, callback=[eval_callback, best_ckpt])

    print("Training complete and model saved")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/
#  python -m pymunk_experiments.gripper2.discrete_control.RL.environment

