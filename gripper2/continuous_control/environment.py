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

        # Define action space as 3D continuous
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, -1, -1, -1], dtype=np.float32),
                                       high=np.array([1, 1, 1, 1, 1, 1], dtype=np.float32))

        # Define (continuous) observation space
        high = np.array([np.inf] * 29, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
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

        # Scale the 6-D action from [0,1] into your desired ranges
        dx, dy, dl1, dl2, dr1, dr2 = action * np.array(
            [200.0, 200.0, 0.1, 0.1, 0.1, 0.1], dtype=np.float32)

        # 1) Translate the gripper arm
        self.gripper.arm.body.velocity = (dx, dy)

        # Left Finger 1: angle ∈ [–0.5, +0.2]
        lf1 = self.gripper.left_finger1.body
        lf1.angle = np.clip(lf1.angle + dl1, -0.5, 0.2)

        # Right Finger 1: angle ∈ [–0.2, +0.5]
        rf1 = self.gripper.right_finger1.body
        rf1.angle = np.clip(rf1.angle + dr1, -0.2, 0.5)

        # Left Finger 2: must stay outside finger 1,
        # so ∠ ∈ [lf1.angle, +1.5]
        lf2 = self.gripper.left_finger2.body
        min_l2 = lf1.angle
        max_l2 = 1.5
        lf2.angle = np.clip(lf2.angle + dl2, min_l2, max_l2)

        # Right Finger 2: must stay outside finger 1,
        # so ∠ ∈ [–1.5, rf1.angle]
        rf2 = self.gripper.right_finger2.body
        min_r2 = -1.5
        max_r2 = rf1.angle
        rf2.angle = np.clip(rf2.angle + dr2, min_r2, max_r2)

        # 3) Step physics
        self.space.step(1.0 / self.FPS)
        self.current_step += 1

        # 4) Get obs, reward, etc.
        obs = self.get_observation()
        reward, done, success = self.get_reward(obs)

        # 5) (Optional) zero‐out residual velocity if you want no inertia
        self.gripper.arm.body.velocity = (0.0, 0.0)

        info = {'success': success}
        truncated = False
        return obs, reward, done, truncated, info

    def get_observation(self):

        base = self.gripper.base.body
        obj = self.object.body

        # Object relative to base
        rel_obj_pos = obj.position - base.position
        rel_obj_vel = obj.velocity - base.velocity

        # Object orientation
        obj_orient = (np.cos(obj.angle), np.sin(obj.angle))

        # Finger Angles and angular velocities
        fingers = [self.gripper.left_finger1.body, self.gripper.left_finger2.body, self.gripper.right_finger1.body,
                   self.gripper.right_finger2.body]
        finger_feats = []
        for j in fingers:
            finger_feats.extend([np.cos(j.angle), np.sin(j.angle), j.angular_velocity])

        # Fingertip positions
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)
        l_tip_rel = l_tip - obj.position
        r_tip_rel = r_tip - obj.position
        gap = np.linalg.norm(l_tip - r_tip) / 200
        floor_y = self.floor.shape.a[1] + self.floor.shape.radius  # floor y-coordinate
        l_tip_floor = l_tip.y - floor_y
        r_tip_floor = r_tip.y - floor_y

        # Touch BOOLs
        l_touch = 1.0 if self.gripper.left_finger2.shape.shapes_collide(self.object.shape).points else 0.0
        r_touch = 1.0 if self.gripper.right_finger2.shape.shapes_collide(self.object.shape).points else 0.0

        obs = np.array([l_tip_floor, r_tip_floor,
                        obj.position.x, obj.position.y,
                        rel_obj_pos.x, rel_obj_pos.y,
                        rel_obj_vel.x, rel_obj_vel.y,
                        *finger_feats,  # 12 values
                        l_tip_rel.x, l_tip_rel.y,
                        r_tip_rel.x, r_tip_rel.y,
                        gap,
                        l_touch, r_touch,
                        obj_orient[0], obj_orient[1],
                        ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        base_pos = self.gripper.base.body.position
        base_dist = np.linalg.norm(base_pos - self.object.body.position)

        # Reward for distance to surface
        r1 = 10 if base_dist < 140 else 10 - 10 * np.tanh((base_dist - 130) / 100)

        # Get the fingertip world position:
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)

        # Reward is left tip is touching below COM and right tip is touching above COM (or vice versa)
        r3 = 10 if ((l_tip[1] < self.object.body.position[1] and obs[-4]) and (
                    r_tip[1] > self.object.body.position[1] and obs[-3])) \
                   or ((l_tip[1] > self.object.body.position[1] and obs[-4]) and (
                    r_tip[1] < self.object.body.position[1] and obs[-3])) else 0

        # Reward if both tips touching below COM. Either r3 or r4, can't have both
        r4 = 10 if (l_tip[1] < self.object.body.position[1] and obs[-4]) and (
                    r_tip[1] < self.object.body.position[1] and obs[-3]) else 0


        height_off_floor = self.object.body.position[1] - 100
        norm_height = max(height_off_floor, 0) / (self.pickup_height - 100)

        condition1 = self.object.body.position[1] < self.gripper.base.body.position[1]
        condition2 = self.gripper.left_finger1.body.position[0] < self.object.body.position[0] < self.gripper.right_finger1.body.position[0]
        condition3 = self.object.shape.shapes_collide(self.floor.shape).points

        r6 = 100 * np.tanh(norm_height) if (condition1 and condition2 and not condition3) else 0  # and condition3) else 0

        reward = (r1 + r3 + r4 + r6)

        # Touch Penalties
        b = 10 if self.gripper.left_finger2.shape.shapes_collide(self.gripper.right_finger1.shape).points else 0.0
        c = 10 if self.gripper.left_finger1.shape.shapes_collide(self.gripper.right_finger2.shape).points else 0.0

        # Penalty for fingers touching floor
        d = 10 if self.gripper.left_finger2.shape.shapes_collide(self.floor.shape).points else 0.0
        e = 10 if self.gripper.right_finger2.shape.shapes_collide(self.floor.shape).points else 0.0

        reward -= (b + c + d + e)

        done = False
        success = False

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            # Reward based on success
            if self.object.body.position.y > self.pickup_height - 100 and (condition1 and condition2 and not condition3):
                success = True
                print("Success!")

        return reward, done, success

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
    vertex = [(-30, -30), (30, -30), (0, 30)]

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

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
        best_model_save_path=f"models",
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
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
        use_sde=True,  # Use Stochastic Depth for better exploration
    )

    model.learn(total_timesteps=5000000, callback=[eval_callback, best_ckpt])
    print("Training complete and model saved")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/

















