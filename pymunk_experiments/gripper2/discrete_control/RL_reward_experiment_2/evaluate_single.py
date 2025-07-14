from stable_baselines3 import PPO
from environment import Environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

vertex = [(-30, -30), (30, -30), (0, 30)]

# 1Load the trained policy
model = PPO.load("models/best_model")

# Create a fresh env in human‐render mode
test_env = DummyVecEnv([lambda: Environment(vertex, training=False, render_mode="human")])
test_env = VecNormalize.load("normalise_stats/vecnormalize_stats_best.pkl", test_env)
test_env.training = False        # freeze stats, use them consistently

# Run N test episodes
N = 10
for ep in range(1, N + 1):
    obs = test_env.reset()
    done = False
    total_reward = 0.0
    print(f"Starting test episode {ep}")
    while not done:
        # deterministic=True for consistent playback
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        total_reward += reward
        test_env.render()  # pops up the Pygame window and draws each frame
    print(f"Episode {ep} finished with total reward {total_reward}")

test_env.close()