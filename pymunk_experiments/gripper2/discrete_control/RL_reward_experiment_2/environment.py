import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pymunk_experiments.grippers import Gripper2
from pymunk_experiments.objects import Ball, Floor, Poly, Walls
from pymunk_experiments import objects, grippers

from pymunk_experiments.sensing import Sensors

class Environment(gym.Env):

    def __init__(self, vertex, training=True, render_mode=None, design_vector=(200, 120, 120, 120, 120)):

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
        self.gripper = Gripper2(self.space, design_vector=design_vector)
        self.floor = Floor(self.space, 20)
        self.walls = Walls(self.space)
        self.vertex = vertex
        self.design_vector = design_vector

        self.touch = False


        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        self.sensors = Sensors(self.gripper, self.object)

        # Define action space
        self.action_space = spaces.Discrete(13)

        # Define 8D (continuous) observation space
        high = np.array([np.inf] * 41, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
        self.max_steps = 2500
        self.current_step = 0
        self.training = training

    def reset(self, seed=None, options=None):

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper2(self.space, design_vector=self.design_vector)
        self.floor = Floor(self.space, radius=20)
        self.walls = Walls(self.space)

        self.touch = False

        # if self.training:
        #     self.gripper.left_finger1.body.angle = np.random.uniform(-1, 0)
        #     self.gripper.right_finger1.body.angle = np.random.uniform(0, 1)
        #     self.gripper.left_finger2.body.angle = self.gripper.left_finger1.body.angle + np.random.uniform(0, 1.7)
        #     self.gripper.right_finger2.body.angle = self.gripper.right_finger1.body.angle - np.random.uniform(0, 1.7)
        #     self.gripper.arm.body.position = (np.random.uniform(200, 600), 300)

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        self.sensors = Sensors(self.gripper, self.object)

        self.current_step = 0

        # initial observation and info
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):

        pygame.event.pump()
        dx = dy = 0
        move = 400
        rotation1 = 0.01
        rotation2 = 0.01

        if action == 0:     # Move left
            dx = -move
        elif action == 1:   # Move right
            dx = move
        elif action == 2:   # Move up
            dy = move
        elif action == 3:   # Move down
            dy = -move
        elif action == 4:   # Open gripper left finger 1
            self.gripper.left_finger1.body.angle -= rotation1 if self.gripper.left_finger1.body.angle > -0.5 else 0.0
        elif action == 5:   # Open gripper right finger 1
            self.gripper.right_finger1.body.angle += rotation1 if self.gripper.right_finger1.body.angle < 0.5 else 0.0
        elif action == 6:   # Close gripper left finger 1
            self.gripper.left_finger1.body.angle += rotation1 if self.gripper.left_finger1.body.angle < 0.2 else 0.0
        elif action == 7:   # Close gripper right finger 1
            self.gripper.right_finger1.body.angle -= rotation1 if self.gripper.right_finger1.body.angle > -0.2 else 0.0
        elif action == 8:   # Open gripper left finger 2
            self.gripper.left_finger2.body.angle -= rotation2 if self.gripper.left_finger2.body.angle - self.gripper.left_finger1.body.angle > 0 else 0.0
        elif action == 9:   # Open gripper right finger 2
            self.gripper.right_finger2.body.angle += rotation2 if self.gripper.right_finger2.body.angle - self.gripper.right_finger1.body.angle < 0 else 0.0
        elif action == 10:  # Close gripper left finger 2
            self.gripper.left_finger2.body.angle += rotation2 if self.gripper.left_finger2.body.angle < 1.5 else 0.0
        elif action == 11:  # Close gripper right finger 2
            self.gripper.right_finger2.body.angle -= rotation2 if self.gripper.right_finger2.body.angle > -1.5 else 0.0
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

        base = self.gripper.base.body
        obj = self.object.body

        # Fingertip positions
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)

        # Object relative to base
        rel_obj_pos = obj.position - base.position
        rel_obj_vel = obj.velocity - base.velocity

        # Object relative to fingertips
        l_tip_rel = l_tip - obj.position
        r_tip_rel = r_tip - obj.position

        # Object orientation
        obj_orient = (np.cos(obj.angle), np.sin(obj.angle))

        # Finger Angles and angular velocities
        fingers = [self.gripper.left_finger1.body, self.gripper.left_finger2.body, self.gripper.right_finger1.body, self.gripper.right_finger2.body]
        finger_feats = []
        for j in fingers:
            finger_feats.extend([np.cos(j.angle), np.sin(j.angle), j.angular_velocity])

        # Gap between fingertips
        gap = np.linalg.norm(l_tip - r_tip) / 200

        # Distance of fingertips to the floor
        floor_y = self.floor.shape.a[1] + self.floor.shape.radius  # floor y-coordinate
        l_tip_floor = l_tip.y - floor_y
        r_tip_floor = r_tip.y - floor_y

        # Three-segment touch forces for each finger
        touch_forces = self.sensors.get_three_segment_forces()  # Returns 12 values (4 fingers × 3 segments)

        obs = np.array([ base.position.x, base.position.y,  # 2 values,  # 2 values
                                l_tip_floor, r_tip_floor,
                                obj.position.x, obj.position.y,
                                rel_obj_pos.x, rel_obj_pos.y,
                                rel_obj_vel.x, rel_obj_vel.y,
                                *finger_feats,                  # 12 values
                                l_tip_rel.x, l_tip_rel.y,
                                r_tip_rel.x, r_tip_rel.y,
                                gap,
                                obj_orient[0], obj_orient[1],
                                *touch_forces
                                ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        # # Distance of base to object
        # base_dist = np.linalg.norm(self.gripper.base.body.position - self.object.body.position)
        # r0 = 10 if base_dist < 150 else 10 - 10 * np.tanh((base_dist - 120) / 150)

        l_collision = self.gripper.left_finger2.shape.shapes_collide(self.object.shape)
        r_collision = self.gripper.right_finger2.shape.shapes_collide(self.object.shape)
        l_contact = 1.0 if l_collision.points else 0.0
        r_contact = 1.0 if r_collision.points else 0.0

        # Reward for lift
        height_off_floor = self.object.body.position[1] - 100
        norm_height = max(height_off_floor, 0) / (self.pickup_height - 100)

        if norm_height > 0.01:
            if not (l_contact and r_contact):
                self.touch = False
        else:
            if l_contact and r_contact:
                self.touch = True

        r6 = 20 * np.tanh(norm_height) if (self.touch) else 0

        reward =  r6

        # Touch Penalties
        a = 10 if self.gripper.left_finger2.shape.shapes_collide(self.gripper.right_finger1.shape).points else 0.0
        b = 10 if self.gripper.left_finger1.shape.shapes_collide(self.gripper.right_finger2.shape).points else 0.0

        # Penalty for fingers touching floor
        c = 10 if self.gripper.left_finger2.shape.shapes_collide(self.floor.shape).points else 0.0
        d = 10 if self.gripper.right_finger2.shape.shapes_collide(self.floor.shape).points else 0.0

        # Penalty for object too fast
        obj_speed = np.linalg.norm([self.object.body.velocity.x, self.object.body.velocity.y])
        e = 50 if obj_speed > 200 else 0.0
        f = 50 if any(force > 30 for force in obs[-12:]) else 0.0

        # reward -= (a+ b + c + d + e + f)

        done = False
        success = False

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True
            # Reward based on success
            if self.object.body.position.y > self.pickup_height - 100 and (l_contact or r_contact):
                success = True
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

class CustomLoggingCallback(BaseCallback):
    """
    This callback works with DummyVecEnv to directly access environment attributes
    """

    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.num_timesteps % self.log_freq == 0:
            # With DummyVecEnv, we can directly access the environment
            if hasattr(self.training_env, 'envs'):
                # Get the actual environment (unwrap Monitor if present)
                env = self.training_env.envs[0]
                if hasattr(env, 'unwrapped'):
                    actual_env = env.unwrapped
                else:
                    actual_env = env

                # Now we can directly access environment attributes!
                if hasattr(actual_env, 'object') and hasattr(actual_env, 'gripper'):
                    # Get object metrics
                    obj_speed = np.linalg.norm([actual_env.object.body.velocity.x, actual_env.object.body.velocity.y])
                    obj_rotation = abs(actual_env.object.body.angular_velocity)
                    obj_height = actual_env.object.body.position.y

                    # Get gripper position
                    gripper_pos = actual_env.gripper.base.body.position

                    # Calculate gripper-object distance
                    obj_gripper_dist = np.linalg.norm([gripper_pos.x - actual_env.object.body.position.x,
                                                      gripper_pos.y - actual_env.object.body.position.y])

                    # Log all metrics to TensorBoard
                    self.logger.record("physics/object_speed", obj_speed)
                    self.logger.record("physics/object_rotation", obj_rotation)
                    self.logger.record("physics/object_height", obj_height)
                    self.logger.record("physics/gripper_object_distance", obj_gripper_dist)

                    # Log observation statistics
                    if hasattr(self.training_env, 'obs_rms'):
                        obs_mean = self.training_env.obs_rms.mean
                        obs_var = self.training_env.obs_rms.var


                        # Log specific dimensions that are likely to have distance issues
                        # Based on your observation structure:
                        self.logger.record("normalization/obj_pos_mean", np.mean(obs_mean[0:2]))  # obj position
                        self.logger.record("normalization/obj_vel_mean", np.mean(obs_mean[2:4]))  # obj velocity
                        self.logger.record("normalization/distance_mean", np.mean(obs_mean[34:36]))  # distance-related mean

                        # Variances
                        self.logger.record("normalization/obj_pos_var", np.mean(obs_var[0:2]))  # obj position
                        self.logger.record("normalization/obj_vel_var", np.mean(obs_var[2:4]))  # obj velocity
                        self.logger.record("normalization/distance_var", np.mean(obs_var[34:36]))  # distance-related




                    # # Log finger angles
                    # finger_angles = [
                    #     actual_env.gripper.left_finger1.body.angle,
                    #     actual_env.gripper.left_finger2.body.angle,
                    #     actual_env.gripper.right_finger1.body.angle,
                    #     actual_env.gripper.right_finger2.body.angle
                    # ]
                    # for i, angle in enumerate(finger_angles):
                    #     self.logger.record(f"gripper/finger_{i}_angle", angle)





        return True

def make_env(vertex, rank, design_vector, render=False):
    """Factory that creates a fresh environment in its own process."""

    def _init():
        env = Environment(vertex, training=True, render_mode="human" if render else None,
                          design_vector=design_vector)
        env = Monitor(env)  # keeps episode stats
        return env

    return _init

class SaveBestWithStats(EvalCallback):
    def __init__(self, *args, vecnormalize, number=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vecnormalize = vecnormalize  # the training wrapper
        self.number = number  # used to save the stats with a unique name

    def _on_step(self) -> bool:
        # run the usual evaluation logic
        old_reward = self.best_mean_reward
        continue_training = super()._on_step()

        if self.best_mean_reward > old_reward:  # new best just saved
            if self.number is not None:
                self.vecnormalize.save(f"normalise_stats/vecnormalize_stats_best.pkl")
            else:
                self.vecnormalize.save(f"normalise_stats/vecnormalize_stats_best.pkl")

        return continue_training

if __name__ == "__main__":

    N_ENVS = 8  # Number of parallel environments

    # This determines the shape of the object to be picked up. If empty, a ball is created with radius 30
    vertex = [(-30, -30), (30, -30), (0, 30)]
    design_vector = (200, 120, 120, 120, 120)

    # Define the policy network architecture
    policy_kwargs = {'net_arch':[256, 256], "log_std_init": 2}

    train_env = SubprocVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env_fast = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
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
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms  # share running stats

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=100000, render=True, verbose=0, deterministic=False)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=256,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="./ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
    )

    # custom_log = CustomLoggingCallback(log_freq=1000, verbose=1)

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt])
    print("Training complete and model saved")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/

