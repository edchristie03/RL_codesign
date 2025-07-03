import numpy as np
import pygame
import pymunk

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pymunk_experiments.grippers import Gripper2
from pymunk_experiments.objects import Ball, Floor, Poly, Walls
from pymunk_experiments import objects, grippers

class Environment(gym.Env):

    def __init__(self, vertex, training=True, render_mode=None, design_vector=(251.0, 161.0, 50.0, 56.0, 163.0)):

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

        if self.vertex:
            self.object = Poly(self.space, self.vertex)
        else:
            self.object = Ball(self.space, 30)

        # Define action space
        self.action_space = spaces.Discrete(13)

        # Define 8D (continuous) observation space
        high = np.array([np.inf] * 45, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
        self.max_steps = 2500
        self.current_step = 0
        self.training = training

        self.action_history = [0] * 5  # Track last 5 actions
        self.contact_duration = {'left': 0, 'right': 0}  # Track contact duration
        self.prev_contacts = {'left': False, 'right': False}

    def reset(self, seed=None, options=None):

        # comply with Gym API: accept seed and options
        super().reset(seed=seed)

        # Remove objects from the space
        for obj in self.space.bodies + self.space.shapes + self.space.constraints:
            self.space.remove(obj)

        # Recreate the objects
        self.gripper = Gripper2(self.space, self.design_vector)
        self.floor = Floor(self.space, radius=20)
        self.walls = Walls(self.space)

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

        self.current_step = 0

        # initial observation and info
        obs = self.get_observation()
        info = {}
        return obs, info

    def step(self, action):

        pygame.event.pump()
        dx = dy = 0
        move = 200
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

        # 1. OBJECT PROPERTIES (8 features)
        # Basic object state
        obj_pos = np.array([obj.position.x, obj.position.y])
        obj_vel = np.array([obj.velocity.x, obj.velocity.y])
        obj_orient = np.array([np.cos(obj.angle), np.sin(obj.angle)])
        obj_ang_vel = obj.angular_velocity

        # Object physical properties
        obj_mass = obj.mass
        obj_friction = self.object.shape.friction

        # Object height above floor
        floor_y = self.floor.shape.a[1] + self.floor.shape.radius
        obj_height = obj.position.y - floor_y

        # 2. GRIPPER STATE (15 features)
        base_pos = np.array([base.position.x, base.position.y])

        fingers = [self.gripper.left_finger1.body, self.gripper.left_finger2.body,
                   self.gripper.right_finger1.body, self.gripper.right_finger2.body]

        finger_angles = np.array([f.angle for f in fingers])
        finger_ang_vels = np.array([f.angular_velocity for f in fingers])

        # Gripper aperture and fingertip positions
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)
        gap = np.linalg.norm(l_tip - r_tip)

        l_tip_rel = l_tip - obj.position
        r_tip_rel = r_tip - obj.position

        # 3. CONTACT & GRASP QUALITY METRICS (12 features)
        # Basic contact detection
        l_collision = self.gripper.left_finger2.shape.shapes_collide(self.object.shape)
        r_collision = self.gripper.right_finger2.shape.shapes_collide(self.object.shape)
        l_contact = 1.0 if l_collision.points else 0.0
        r_contact = 1.0 if r_collision.points else 0.0

        # Contact point distances from object COM
        l_contact_com_dist = 0.0
        r_contact_com_dist = 0.0
        if l_collision.points:
            contact_point = l_collision.points[0]
            l_contact_com_dist = np.linalg.norm(contact_point.point_a - obj.position)
        if r_collision.points:
            contact_point = r_collision.points[0]
            r_contact_com_dist = np.linalg.norm(contact_point.point_a - obj.position)

        # Distance to object surface for each fingertip
        l_query = self.object.shape.point_query(l_tip)
        r_query = self.object.shape.point_query(r_tip)
        l_surf_dist = max(l_query.distance, 0.0)
        r_surf_dist = max(r_query.distance, 0.0)

        # 4. TEMPORAL INFORMATION (7 features)
        # Update contact duration tracking
        self.contact_duration['left'] = self.contact_duration['left'] + 1 if l_contact else 0
        self.contact_duration['right'] = self.contact_duration['right'] + 1 if r_contact else 0

        # Contact stability (how long contacts have been maintained)
        l_contact_duration = self.contact_duration['left']
        r_contact_duration = self.contact_duration['right']

        # Recent action history (last 5 actions encoded as frequency)
        action_freq = np.bincount(self.action_history, minlength=13)[:5]  # Top 5 most common actions

        # 5. GEOMETRIC RELATIONSHIPS (10 features)
        # More precise finger-to-object positioning
        obj_to_gripper = obj.position - base.position

        # Angles from object to each fingertip
        l_tip_angle = np.arctan2(l_tip_rel.y, l_tip_rel.x)
        r_tip_angle = np.arctan2(r_tip_rel.y, r_tip_rel.x)

        # Distance from each fingertip to object COM
        l_tip_com_dist = np.linalg.norm(l_tip_rel)
        r_tip_com_dist = np.linalg.norm(r_tip_rel)

        # Gripper orientation relative to object
        gripper_obj_angle = np.arctan2(obj_to_gripper.y, obj_to_gripper.x)

        # Gripper alignment with object (how well centered)
        horizontal_alignment = abs(base.position.x - obj.position.x)
        vertical_alignment = abs(base.position.y - obj.position.y)

        # 6. TASK-SPECIFIC FEATURES (3 features)
        above_object = 1.0 if base.position.y > obj.position.y else 0.0

        # Object stability (low velocity indicates stable grasp)
        obj_stability = 1.0 / (1.0 + np.linalg.norm(obj_vel) + abs(obj_ang_vel))

        # Combine all features
        obs = np.concatenate([
            # Object properties
            obj_pos,  # 2
            obj_vel,  # 2
            obj_orient,  # 2
            [obj_ang_vel],  # 1
            [obj_mass],  # 1
            [obj_friction],  # 1
            [obj_height],  # 1

            # Gripper state
            base_pos,  # 2
            finger_angles,  # 4
            finger_ang_vels,  # 4
            [gap],  # 1

            # Contact & grasp quality
            [l_contact, r_contact],  # 2
            [l_contact_com_dist, r_contact_com_dist],  # 2
            [l_surf_dist, r_surf_dist],  # 2
            [l_contact_duration, r_contact_duration],  # 2

            # Temporal info
            action_freq,  # 5

            # Geometric relationships
            obj_to_gripper,  # 2
            [l_tip_angle, r_tip_angle],  # 2
            [l_tip_com_dist, r_tip_com_dist],  # 2
            [gripper_obj_angle],  # 1
            [horizontal_alignment, vertical_alignment],  # 2

            # Task-specific
            [above_object],  # 1
            [obj_stability],  # 1
        ], dtype=np.float32)

        return obs

    def get_reward(self, obs):

        # Get the fingertip world position:
        l_tip = self.gripper.left_finger2.body.local_to_world(self.gripper.left_finger2.shape.b)
        r_tip = self.gripper.right_finger2.body.local_to_world(self.gripper.right_finger2.shape.b)

        # Query distance to surface (positive outside, negative if inside)
        l_query = self.object.shape.point_query(l_tip)
        r_query = self.object.shape.point_query(r_tip)

        # Get the unsigned distance (clamp negative to zero if you only care about outside)
        l_surf_dist = max(l_query.distance, 0.0)
        r_surf_dist = max(r_query.distance, 0.0)

        # Reward for distance to surface
        r1 = 10 if l_surf_dist < 20 else 10 - 10 * np.tanh((l_surf_dist - 20) / 100)
        r2 = 10 if r_surf_dist < 20 else 10 - 10 * np.tanh((r_surf_dist - 20) / 100)

        # Reward is left tip is touching below COM and right tip is touching above COM (or vice versa)
        r3 = 10 if ((l_tip[1] < self.object.body.position[1] and obs[21]) and (r_tip[1] > self.object.body.position[1] and obs[22])) \
                or ((l_tip[1] > self.object.body.position[1] and obs[21]) and (r_tip[1] < self.object.body.position[1] and obs[2])) else 0

        # Reward if both tips touching below COM. Either r3 or r4, can't have both
        r4 = 10 if (l_tip[1] < self.object.body.position[1] and obs[21]) and (r_tip[1] < self.object.body.position[1] and obs[22]) else 0

        # Reward based on height of object if under gripper. Diminishes to a max of 100 at the target pickup height
        condition1 = self.object.body.position[1] < self.gripper.base.body.position[1]
        condition2 = self.gripper.left_finger1.body.position[0] < self.object.body.position[0] < self.gripper.right_finger1.body.position[0]

        height_off_floor = self.object.body.position[1] - 100
        norm_height = max(height_off_floor, 0) / (self.pickup_height - 100)

        r6 = 100 * np.tanh(norm_height) if (condition1 and condition2)  else 0

        reward = (r1 + r2 + r3 + r4 + r6)

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
            if self.object.body.position.y > self.pickup_height - 100 and (condition1 and condition2):
                success = True


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
                self.vecnormalize.save(f"pymunk_experiments/gripper2/discrete_control/RL_experiments/{self.number}/normalise_stats/vecnormalize_stats_best.pkl")
            else:
                self.vecnormalize.save(f"pymunk_experiments/gripper2/discrete_control/RL_experiments/normalise_stats/vecnormalize_stats_best.pkl")

        return continue_training

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

if __name__ == "__main__":

    N_ENVS = 1  # Number of parallel environments

    # This determines the shape of the object to be picked up. If empty, a ball is created with radius 30
    vertex = [(-30, -30), (30, -30), (0, 30)]
    design_vector = (251.0, 161.0, 50.0, 56.0, 163.0)  # Example design vector for the gripper

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

    # Training envs (headless, parallel)
    train_env = DummyVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)])
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Copy the observation normalization stats from the training env to the eval env
    eval_env.obs_rms = train_env.obs_rms

    # Stop training if no model improvement after 10 evaluations
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=200,
        min_evals=1,
        verbose=1
    )

    best_ckpt = SaveBestWithStats(
        eval_env,
        vecnormalize=train_env,
        best_model_save_path=f"pymunk_experiments/gripper2/discrete_control/RL/models/ppo_pymunk_gripper_best",
        n_eval_episodes=5,
        eval_freq=20_000,
        deterministic=True,
        render=False,
        verbose=0,
        callback_after_eval=stop_callback
    )

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=10000000, render=True, verbose=0, deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=256,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log="pymunk_experiments/gripper2/discrete_control/RL/ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
    )

    custom_logger = CustomLoggingCallback()

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt, custom_logger])

    print("Training complete and model saved")

    train_env.close()
    eval_env.close()


# tensorboard --logdir ./ppo_gripper_tensorboard/
#  python -m pymunk_experiments.gripper2.discrete_control.RL.environment

