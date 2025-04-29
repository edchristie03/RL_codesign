import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3.common.callbacks import EvalCallback
import numpy as np
import pygame
import pymunk
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from main import Gripper, Ball, Floor
import main

class Environment(gym.Env):

    metadata = {"render_modes": ["human"],
                "render_fps": 60}

    def __init__(self, render_mode=None):
        # super().__init__(render_mode=render_mode)

        print("Initializing environment...")

        self.render_mode = render_mode

        pygame.init()
        self.surface = pygame.Surface((800, 800))
        self.clock = pygame.time.Clock()
        self.space = pymunk.Space()
        self.space.gravity = (0, -1000)  # gravity
        self.FPS = 60

        # Quick HACK for display. Should refactor main
        main.display = self.surface

        # Define objects in the environment
        self.gripper = Gripper(self.space)
        self.ball = Ball(self.space, radius=30)
        self.floor = Floor(self.space, radius=10)

        # Define action space
        self.action_space = spaces.Discrete(6)

        # Define (continuous) observation space
        high = np.array([np.inf] * 10, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Other simulation parameters
        self.pickup_height = 400
        self.max_steps = 1000
        self.current_step = 0

    def reset(self, seed=None, options=None):

        print("Resetting environment...")

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

        print("Action taken this step:", action)

        pygame.event.pump()
        # Map discrete action to motion or gripper command
        dx = dy = 0
        grip_rate = 0.0
        if action == 0:     # Move left
            dx = -1
        elif action == 1:   # Move right
            dx = 1
        elif action == 2:   # Move up
            dy = 1
        elif action == 3:   # Move down
            dy = -1
        elif action == 4:   # Open gripper
            grip_rate = 2.0
        elif action == 5:   # Close gripper
            grip_rate = -2.0

        # Apply the action to the gripper
        self.gripper.base.body.position += (dx, dy)
        self.gripper.left_finger.motor.rate = grip_rate
        self.gripper.right_finger.motor.rate = -grip_rate

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

        print("Getting observation...")

        # Base position and velocity
        bx, by = self.gripper.base.body.position
        bvx, bvy = self.gripper.base.body.velocity

        # Gripper angles and angular velocities
        left = self.gripper.left_finger.body
        right = self.gripper.right_finger.body
        la, lav = left.angle, left.angular_velocity
        ra, rav = right.angle, right.angular_velocity

        # Ball relative position
        ball = self.ball.body
        rel_pos = ball.position - self.gripper.base.body.position

        obs = np.array([bx, by, bvx, bvy, la, lav, ra, rav, rel_pos.x, rel_pos.y], dtype=np.float32)
        return obs

    def get_reward(self, obs):
        # Reward based on the distance of gripper base to the ball
        r1 = - np.linalg.norm(obs[8:10])

        # Reward based on height of ball
        r2 = self.ball.body.position[1]

        print(f'r1: {r1}, r2: {r2}')

        reward = r1 + r2
        done = False

        # Reward based on success
        if self.ball.body.position.y > self.pickup_height:
            reward += 10.0
            done = True

        # Episode termination if max steps reached
        if self.current_step >= self.max_steps:
            done = True

        print("Reward:", reward)

        return reward, done

    def render(self, mode="human"):

        print("Rendering...")

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
            # pump events again (optional)
            pygame.event.pump()
            # blit & flip
            self._window.blit(self.surface, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.FPS)
        return self.surface

    def close(self):
        pygame.quit()

if __name__ == "__main__":

    # def make_env():
    #     env = Environment()
    #     return Monitor(env)
    #
    # vec_env = DummyVecEnv([make_env])
    # model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log="./ppo_gripper_tensorboard/")
    # model.learn(total_timesteps=100000)
    # model.save("ppo_pymunk_gripper")
    # print("Training complete and model saved.")

    # Training env (headless)
    train_env = DummyVecEnv([lambda: Monitor(Environment(render_mode=None))])

    # Evaluation env (human‐render)
    eval_env = DummyVecEnv([lambda: Environment(render_mode="human")])

    # Make callback to run 1 episode of eval every 1000 episodes
    eval_callback = EvalCallback(
        eval_env,
        n_eval_episodes=1,
        eval_freq=20000,
        render=True,  # actually calls env.render("human")
        verbose=1
    )

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./ppo_gripper_tensorboard/"
    )
    model.learn(
        total_timesteps=100_0000,
        callback=eval_callback
    )
    model.save("ppo_pymunk_gripper")















