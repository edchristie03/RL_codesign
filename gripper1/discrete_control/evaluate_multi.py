from stable_baselines3 import PPO
from environment import Environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt

vertices = [[], [(-30, -30), (30, -30), (30, 30), (-30, 30)] ,[(-30, -30), (30, -30), (30, 30)], [(-30, -30), (30, -30), (0, 30)],
                [(-30, -30), (30, -30), (0, 30), (-30, 30)], [(-10, -30), (0, -30), (0, 30), (-10, 30)], [(-80, -30), (80, -30), (80, 0), (-80, 0)]]

avg_returns = []
successes = []

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
    shape_successes = 0
    for ep in range(1, N + 1):
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        print(f"Starting test episode {ep}")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            test_env.render()
            if info[0]['success']:
                shape_successes += 1
        print(f"Episode {ep} finished with total reward {total_reward}")
        returns.append(total_reward)
    avg_returns.append(np.mean(returns))
    successes.append(shape_successes)

    test_env.close()

labels = ["Circle", "Square", "RA triangle", "Equi Triangle", "Half Trapezoid", "Tall Rectangle", "Wide Rectangle"]


# Now plot average returns
plt.figure(figsize=(6,4))
plt.plot(range(len(vertices)), avg_returns, marker='o', linewidth=2)
plt.xticks(range(len(vertices)), labels, rotation=45)
plt.xlabel("Vertex Configuration")
plt.ylabel("Average Return")
plt.title("Gripper 1: Average Return per Vertex Configuration")
plt.grid(True)
plt.tight_layout()
plt.show()

# ball, square, RA triangle, equilateral triangle, half trapezoid, tall rectangle, wide rectangle

# Plot success rates
plt.figure(figsize=(6,4))
plt.bar(range(len(vertices)), successes)
plt.xticks(range(len(vertices)), labels, rotation=45)
plt.xlabel("Vertex Configuration")
plt.ylabel("Successes")
plt.title("Gripper 1: Successes per Vertex Configuration")
plt.grid(True)
plt.tight_layout()
plt.show()
