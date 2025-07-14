from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from environment import Environment, make_env, SaveBestWithStats
import random
import numpy as np
import os
import matplotlib.pyplot as plt

def create_individual(bounds):
    """Create a random design vector within bounds"""
    design = (
        random.uniform(*bounds['base_width']),
        random.uniform(*bounds['left_finger_1']),
        random.uniform(*bounds['right_finger_1']),
        random.uniform(*bounds['left_finger_2']),
        random.uniform(*bounds['right_finger_2'])
    )
    return tuple(int(x) for x in design)

def generate_population(size, bounds):

    print(f"\nGenerating population:\n")

    population = {}
    for i in range(size):
        design = create_individual(bounds)
        population[i] = [design, 0, 0]  # [design_vector, fitness, success_rate]

    # Print the generated population
    for idx, (design, fitness, success_rate) in population.items():
        print(f"Individual {idx+1} design: {design}, Fitness: {fitness}, Success Rate: {success_rate}")

    print()

    return population

def get_bounds(base_width, left_finger_1, right_finger_1, left_finger_2, right_finger_2, max_difference):
    """Get the bounds for each design parameter"""
    half_delta = max_difference / 2.0
    return {
        'base_width': (base_width - half_delta, base_width + half_delta),
        'left_finger_1': (left_finger_1 - half_delta, left_finger_1 + half_delta),
        'right_finger_1': (right_finger_1 - half_delta, right_finger_1 + half_delta),
        'left_finger_2': (left_finger_2 - half_delta, left_finger_2 + half_delta),
        'right_finger_2': (right_finger_2 - half_delta, right_finger_2 + half_delta),
    }

def get_mean_design(population):
    """Calculate the mean design vector from the population"""
    total = [0, 0, 0, 0, 0]
    for design, _, _ in population.values():
        for i in range(len(design)):
            total[i] += design[i]

    mean_design = tuple(round(x / len(population), 0) for x in total)

    print(f"Mean Design Vector: {mean_design}")

    return mean_design

def train_from_scratch(vertex, design_vector, experiment_id):

    print(f"Training mean design vector")

    N_ENVS = 8  # Number of parallel environments

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

    # Training envs (headless, parallel)
    train_env = SubprocVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    # Copy the observation normalization stats from the training env to the eval env
    eval_env.obs_rms = train_env.obs_rms

    best_ckpt = SaveBestWithStats(
        eval_env,
        vecnormalize=train_env,
        number=f'Experiments/{experiment_id}/mean',
        best_model_save_path=f"Experiments/{experiment_id}/mean/models/ppo_pymunk_gripper_best",
        n_eval_episodes=5,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        verbose=0,

    )

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=500000, render=True, verbose=0,
                                 deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=256,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log=f'Experiments/{experiment_id}/ppo_gripper_tensorboard/',
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
    )

    model.learn(total_timesteps=5000000, callback=[eval_callback, best_ckpt])

    print("Training complete and model saved\n")

    train_env.close()
    eval_env.close()

def evaluate(design_vector, vertex, experiment_id):

    # 1Load the trained policy
    model = PPO.load(f'Experiments/{experiment_id}/mean/models/ppo_pymunk_gripper_best/best_model')

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv([lambda: Environment(vertex, training=False, design_vector=design_vector, render_mode="human")])
    test_env = VecNormalize.load(f'Experiments/{experiment_id}/mean/normalise_stats/vecnormalize_stats_best.pkl', test_env)
    test_env.training = False  # freeze stats, use them consistently

    # Run N test episodes
    N = 10
    returns = []
    successes = 0
    for ep in range(1, N + 1):
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            # deterministic=True for consistent playback
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            test_env.render() if ep == 1 else None
            if info[0]['success']:
                successes += 1
        print(f"Episode {ep} finished with total reward {total_reward}")
        returns.append(total_reward)

    test_env.close()

    mean_return = np.mean(returns)
    success_rate = successes / N

    print(f"\nMean Return: {mean_return}, Success Rate: {success_rate}\n")

    return mean_return, success_rate

def evaluate_population(vertex, population):
    """Evaluate the trained policy on all grippers in the population"""

    print("\nEvaluating population:\n")
    for idx, (design, _, _) in population.items():
        print(f"Evaluating individual {idx+1} design: {design}")
        mean_return, success_rate = evaluate(design, vertex, experiment_id)
        population[idx][1] = mean_return  # Update fitness with mean return
        population[idx][2] = success_rate  # Update success rate
        print(f"Individual {idx+1} design: {design}, Mean Return: {mean_return}, Success Rate: {success_rate}")

    return population

def plot_results(population, mean_return, mean_success_rate, perturbation):
    """Plot the results of the evaluation"""
    designs = [design for design, _, _ in population.values()]
    fitness = [fitness for _, fitness, _ in population.values()]
    success_rates = [success_rate for _, _, success_rate in population.values()]

    # Add mean values as separate bars
    designs.append("Mean Design")
    fitness.append(mean_return)
    success_rates.append(mean_success_rate)

    plt.figure(figsize=(14, 6))

    # Plot mean returns
    plt.subplot(1, 2, 1)
    bars1 = plt.bar(range(len(designs)), fitness, tick_label=[str(d) for d in designs])
    # Color the last bar (mean) differently
    bars1[-1].set_color('orange')
    plt.axhline(mean_return, color='r', linestyle='--', label='Mean Return Reference')
    plt.title(f'Mean Returns of Gripper Designs with Perturbation = {perturbation}')
    plt.xlabel('Design Vector')
    plt.ylabel('Mean Return')
    plt.xticks(rotation=90)
    plt.legend()

    # Plot success rates
    plt.subplot(1, 2, 2)
    bars2 = plt.bar(range(len(designs)), success_rates, tick_label=[str(d) for d in designs])
    # Color the last bar (mean) differently
    bars2[-1].set_color('orange')
    plt.axhline(mean_success_rate, color='r', linestyle='--', label='Mean Success Rate Reference')
    plt.title(f'Success Rates of Gripper Designs with Perturbation = {perturbation}')
    plt.xlabel('Design Vector')
    plt.ylabel('Success Rate')
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    # Save the plot
    os.makedirs(f'Experiments/{experiment_id}/plots', exist_ok=True)
    plt.savefig(f'Experiments/{experiment_id}/plots/results_{perturbation}.png')

    plt.show()

def get_next_experiment_id():
    """
    Get the next experiment ID based on existing directories.
    This assumes directories are named as 'Experiments/{id}'.
    """
    if not os.path.exists("Experiments"):
        os.makedirs("Experiments")
        return 1

    existing_ids = [int(f) for f in os.listdir("Experiments")]
    return max(existing_ids, default=0) + 1

def plot_perturbations(perturbations, average_returns, average_success_rates, experiment_id):

    """Plot average returns and success rates across perturbations"""
    plt.figure(figsize=(12, 6))

    # Plot average returns
    plt.subplot(1, 2, 1)
    plt.plot(perturbations, average_returns, marker='o', label='Average Return')
    plt.title('Average Returns Across Perturbations')
    plt.xlabel('Perturbation')
    plt.ylabel('Average Return')
    plt.xticks(perturbations)
    plt.grid()
    plt.legend()

    # Plot average success rates
    plt.subplot(1, 2, 2)
    plt.plot(perturbations, average_success_rates, marker='o', color='orange', label='Average Success Rate')
    plt.title('Average Success Rates Across Perturbations')
    plt.xlabel('Perturbation')
    plt.ylabel('Average Success Rate')
    plt.xticks(perturbations)
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'Experiments/{experiment_id}/plots/average_perturbations.png')
    plt.show()

def main(experiment_id):

    perturbations = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    average_returns = []
    average_success_rates = []

    vertex = [(-30, -30), (30, -30), (30, 30), (-30, 30)]
    population_size = 10

    base_width = 200
    left_finger_1 = 120
    right_finger_1 = 120
    left_finger_2 = 120
    right_finger_2 = 120

    mean_design = (base_width, left_finger_1, right_finger_1, left_finger_2, right_finger_2)
    train_from_scratch(vertex, mean_design, experiment_id)

    for perturbation in perturbations:
        print(f"\nRunning experiment with perturbation: {perturbation}")

        # Calculate bounds for the design parameters based on the perturbation
        bounds = get_bounds(base_width, left_finger_1, right_finger_1, left_finger_2, right_finger_2, perturbation)
        # Generate initial population
        population = generate_population(population_size, bounds)
        # Evaluate the mean design vector
        print(f"Evaluating mean design vector")
        mean_return, mean_success_rate = evaluate(mean_design, vertex, experiment_id)
        # Evaluate the entire population using the trained policy
        population = evaluate_population(vertex, population)
        # Plot the results
        plot_results(population, mean_return, mean_success_rate, perturbation)

        # Store the average return and success rate for this perturbation
        average_return_of_population = np.mean([individual[1] for individual in population.values()])
        average_returns.append(average_return_of_population)  # Assuming the first individual is the mean design
        average_success_rate_of_population = np.mean([individual[2] for individual in population.values()])
        average_success_rates.append(average_success_rate_of_population)

    # Plot average returns and success rates across perturbations
    plot_perturbations(perturbations, average_returns, average_success_rates, experiment_id)

if __name__ == "__main__":

    experiment_id = get_next_experiment_id()
    main(experiment_id)



