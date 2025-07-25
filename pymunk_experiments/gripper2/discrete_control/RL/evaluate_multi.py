from stable_baselines3 import PPO
from environment import Environment
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import numpy as np
import matplotlib.pyplot as plt

shape_vertices = {
    "Circle":          [],
    "Square":          [(-30, -30), (30, -30), (30, 30), (-30, 30)],
    "RA triangle":     [(-30, -30), (30, -30), (30, 30)],
    "Equi Triangle":   [(-30, -30), (30, -30), (0, 30)]}
#     "Half Trapezoid":  [(-30, -30), (30, -30), (0, 30), (-30, 30)],
#     "Tall Rectangle":  [(-10, -30), (0, -30), (0, 30), (-10, 30)],
#     "Wide Rectangle":  [(-80, -30), (80, -30), (80, 0), (-80, 0)],
# }

avg_returns = []
successes = []

for shape_name, vertex in shape_vertices.items():

    # 1Load the trained policy
    model = PPO.load(f"models/{shape_name}/best_model")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv([lambda: Environment(vertex, training=False, render_mode="human")])
    test_env = VecNormalize.load(f"normalise_stats/vecnormalize_stats_best_{shape_name}.pkl", test_env)
    test_env.training = False        # freeze stats, use them consistently

    # Run N test episodes
    N = 3
    returns = []
    shape_successes = 0
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
            test_env.render()
            if info[0]['success']:
                shape_successes += 1
        print(f"Episode {ep} finished with total reward {total_reward}")
        returns.append(total_reward)

    avg_returns.append(np.mean(returns))
    successes.append(shape_successes)
    test_env.close()

labels = ["Circle", "Square", "RA triangle", "Equi Triangle"] #, "Half Trapezoid", "Tall Rectangle", "Wide Rectangle"]


# Now plot average returns
plt.figure(figsize=(6,4))
plt.plot(range(len(shape_vertices)), avg_returns, marker='o', linewidth=2)
plt.xticks(range(len(shape_vertices)), labels, rotation=45)
plt.xlabel("Vertex Configuration")
plt.ylabel("Average Return")
plt.title("Gripper 2: Average Return per Vertex Configuration")
plt.grid(True)
plt.tight_layout()
plt.show()

# ball, square, RA triangle, equilateral triangle, half trapezoid, tall rectangle, wide rectangle

# Plot success rates
plt.figure(figsize=(6,4))
plt.bar(range(len(shape_vertices)), successes)
plt.xticks(range(len(shape_vertices)), labels, rotation=45)
plt.xlabel("Vertex Configuration")
plt.ylabel("Successes")
plt.title("Gripper 2: Successes per Vertex Configuration")
plt.grid(True)
plt.tight_layout()
plt.show()
