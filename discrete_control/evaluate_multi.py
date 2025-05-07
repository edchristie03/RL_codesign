from stable_baselines3 import PPO
from environment import Environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt

vertices = [[], [(-25, -25), (25, -25), (25, 25), (-25, 25)] ,[(-25, -25), (25, -25), (25, 25)], [(-25, -25), (25, -25), (0, 25)]]

avg_returns = []

for idx, vertex in enumerate(vertices):

    # 1Load the trained policy
    model = PPO.load(f"models/ppo_pymunk_gripper_{idx}")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv([lambda: Environment(vertex, render_mode="human")])
    test_env = VecNormalize.load(f"normalise_stats/vecnormalize_stats_{idx}.pkl", test_env)
    test_env.training = False        # freeze stats, use them consistently

    # Run N test episodes
    N = 10
    returns = []
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
        returns.append(total_reward)
    avg_returns.append(np.mean(returns))

    test_env.close()

# Now plot
plt.figure(figsize=(6,4))
plt.plot(range(len(vertices)), avg_returns, marker='o', linewidth=2)
plt.xticks(range(len(vertices)), ['Circle', 'Square', 'RA Triangle', 'Equilateral Triangle'], rotation=45)
plt.xlabel("Vertex Configuration")
plt.ylabel("Average Return")
plt.title("Average Return per Vertex Configuration")
plt.grid(True)
plt.tight_layout()
plt.show()
