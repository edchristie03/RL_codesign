from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import Environment
import random
import numpy as np
import pickle
import os
from datetime import datetime

shape_name, vertex = "Equi Triangle", [(-30, -30), (30, -30), (0, 30)]

# BEST = (251.0, 161.0, 50.0, 56.0, 163.0)

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 5
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
ELITE_SIZE = 2  # Number of best individuals to keep unchanged
TOURNAMENT_SIZE = 3

# Design vector bounds
BOUNDS = {
    'base_width': (50, 300),
    'left_finger_1': (30, 200),
    'right_finger_1': (30, 200),
    'left_finger_2': (30, 200),
    'right_finger_2': (30, 200)
}

def train(id, design_vector, generation):
    """Train a PPO model for a given design vector"""
    N_ENVS = 8  # Number of parallel environments

    # Define the policy network architecture
    policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

    def make_env(vertex, rank, design_vector, render=False):
        """Factory that creates a fresh environment in its own process."""

        def _init():
            env = Environment(vertex, training=True, render_mode="human" if render else None,
                              design_vector=design_vector)
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
                os.makedirs("normalise_stats", exist_ok=True)
                self.vecnormalize.save(f"normalise_stats/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl")

            return continue_training

    # Training envs (headless, parallel)
    train_env = SubprocVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env_fast = DummyVecEnv([make_env(vertex, 0, design_vector, render=False)])
    eval_env_fast = VecNormalize(eval_env_fast, norm_obs=True, norm_reward=False, training=False)
    eval_env_fast.obs_rms = train_env.obs_rms

    os.makedirs(f"models/{shape_name}_Gen{generation}_ID_{id}", exist_ok=True)
    best_ckpt = SaveBestWithStats(
        eval_env_fast,
        vecnormalize=train_env,
        best_model_save_path=f"models/{shape_name}_Gen{generation}_ID_{id}",
        n_eval_episodes=10,
        eval_freq=20_000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Evaluation env (still a single window so you can watch)
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])  # Set to False for faster training
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms  # share running stats

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=100000, render=True, verbose=0,
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
    )

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt])
    print(f"Training complete and model saved for Gen{generation}_ID{id}")

    train_env.close()
    eval_env.close()

def evaluate_model(id, design_vector, generation):
    """Evaluate a trained model and return fitness score"""
    # Load the trained policy
    model = PPO.load(f"models/{shape_name}_Gen{generation}_ID_{id}/best_model")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv([lambda: Environment(vertex, training=False, render_mode="human", design_vector=design_vector)])
    test_env = VecNormalize.load(f"normalise_stats/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl",
                                 test_env)
    test_env.training = False  # freeze stats, use them consistently

    # Run N test episodes
    N = 5  # Multiple episodes for more robust evaluation
    returns = []
    success_count = 0

    for ep in range(1, N + 1):
        obs = test_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # deterministic=True for consistent playback
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward

            if info[0].get('success', False):
                success_count += 1

        returns.append(total_reward)

    avg_return = np.mean(returns)
    success_rate = success_count / N

    # Fitness function: combine average return and success rate
    fitness = avg_return #+ (success_rate * 100)  # Bonus for successful episodes

    test_env.close()
    return fitness

def create_individual():
    """Create a random design vector within bounds"""
    design = (
        random.uniform(*BOUNDS['base_width']),
        random.uniform(*BOUNDS['left_finger_1']),
        random.uniform(*BOUNDS['right_finger_1']),
        random.uniform(*BOUNDS['left_finger_2']),
        random.uniform(*BOUNDS['right_finger_2'])
    )
    return tuple(round(x, 0) for x in design)

def tournament_selection(population, tournament_size=TOURNAMENT_SIZE):
    """Select an individual using tournament selection"""
    tournament = random.sample(list(population.items()), tournament_size)
    winner = max(tournament, key=lambda x: x[1][1])  # Select based on fitness
    return winner[1][0]  # Return design vector

def crossover(parent1, parent2):
    """Perform uniform crossover between two parents"""
    if random.random() > CROSSOVER_RATE:
        return parent1, parent2

    child1 = []
    child2 = []

    for i in range(len(parent1)):
        if random.random() < 0.5:
            child1.append(parent1[i])
            child2.append(parent2[i])
        else:
            child1.append(parent2[i])
            child2.append(parent1[i])

    return tuple(child1), tuple(child2)

def mutate(individual):
    """Mutate an individual by adding Gaussian noise"""
    if random.random() > MUTATION_RATE:
        return individual

    mutated = []
    bounds_list = list(BOUNDS.values())

    for i, value in enumerate(individual):
        if random.random() < 0.3:  # 30% chance to mutate each gene
            # Add Gaussian noise (10% of the range)
            noise_scale = (bounds_list[i][1] - bounds_list[i][0]) * 0.1
            new_value = value + random.gauss(0, noise_scale)
            # Clamp to bounds
            new_value = max(bounds_list[i][0], min(bounds_list[i][1], new_value))
            mutated.append(round(new_value, 0))
        else:
            mutated.append(value)

    return tuple(mutated)

def save_generation_data(generation, population, best_individual, best_fitness):
    """Save generation data to file"""
    os.makedirs("evolution_data", exist_ok=True)

    data = {
        'generation': generation,
        'population': population,
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'timestamp': datetime.now().isoformat()
    }

    with open(f"evolution_data/generation_{generation}.pkl", 'wb') as f:
        pickle.dump(data, f)

def evolutionary_algorithm():
    """Main evolutionary algorithm loop"""
    print("=== Starting Evolutionary Algorithm ===")

    # Initialize population
    population = {}
    for i in range(POPULATION_SIZE):
        design = create_individual()
        population[i] = [design, 0]  # [design_vector, fitness]

    print(f"Initial population size: {len(population)}")

    # Track best individual across all generations
    global_best_fitness = float('-inf')
    global_best_individual = None
    global_best_generation = 0

    # Evolution loop
    for generation in range(NUM_GENERATIONS):
        print(f"\n=== GENERATION {generation + 1}/{NUM_GENERATIONS} ===")

        # Evaluate population
        for id, (design_vector, _) in population.items():
            print(f"Training Gen{generation + 1}_ID{id} with design: {design_vector}")
            train(id, design_vector, generation + 1)

            print(f"Evaluating Gen{generation + 1}_ID{id}")
            fitness = evaluate_model(id, design_vector, generation + 1)
            population[id][1] = fitness

            print(f"Gen{generation + 1}_ID{id} - Fitness: {fitness:.2f}")

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_individual = design_vector
                global_best_generation = generation + 1

        # Sort population by fitness
        sorted_population = sorted(population.items(), key=lambda x: x[1][1], reverse=True)

        # Print generation statistics
        fitnesses = [individual[1][1] for individual in sorted_population]
        print(f"\nGeneration {generation + 1} Statistics:")
        print(f"Best fitness: {max(fitnesses):.2f}")
        print(f"Average fitness: {np.mean(fitnesses):.2f}")
        print(f"Worst fitness: {min(fitnesses):.2f}")
        print(f"Best design: {sorted_population[0][1][0]}")

        # Save generation data
        save_generation_data(generation + 1, dict(sorted_population),
                             sorted_population[0][1][0], sorted_population[0][1][1])

        # Create next generation (if not the last generation)
        if generation < NUM_GENERATIONS - 1:
            new_population = {}
            next_id = 0

            # Elitism: Keep best individuals
            for i in range(ELITE_SIZE):
                new_population[next_id] = [sorted_population[i][1][0], 0]
                next_id += 1

            # Generate offspring
            while next_id < POPULATION_SIZE:
                # Selection
                parent1 = tournament_selection(population)
                parent2 = tournament_selection(population)

                # Crossover
                child1, child2 = crossover(parent1, parent2)

                # Mutation
                child1 = mutate(child1)
                child2 = mutate(child2)

                # Add to new population
                if next_id < POPULATION_SIZE:
                    new_population[next_id] = [child1, 0]
                    next_id += 1
                if next_id < POPULATION_SIZE:
                    new_population[next_id] = [child2, 0]
                    next_id += 1

            population = new_population
            print(f"Generated {len(population)} individuals for next generation")

    # Final results
    print("\n" + "=" * 50)
    print("EVOLUTIONARY ALGORITHM COMPLETE")
    print("=" * 50)
    print(f"Global best fitness: {global_best_fitness:.2f}")
    print(f"Global best design: {global_best_individual}")
    print(f"Found in generation: {global_best_generation}")

    # Save final results
    final_results = {
        'global_best_fitness': global_best_fitness,
        'global_best_individual': global_best_individual,
        'global_best_generation': global_best_generation,
        'final_population': population,
        'parameters': {
            'population_size': POPULATION_SIZE,
            'num_generations': NUM_GENERATIONS,
            'mutation_rate': MUTATION_RATE,
            'crossover_rate': CROSSOVER_RATE,
            'elite_size': ELITE_SIZE,
            'tournament_size': TOURNAMENT_SIZE
        }
    }

    with open("evolution_data/final_results.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    return global_best_individual, global_best_fitness

def test_best_design(design_vector, generation, id):
    """Test the best design with visual rendering"""
    print(f"\n=== Testing Best Design Visually ===")
    print(f"Design: {design_vector}")

    # Load the trained policy
    model = PPO.load(f"models/{shape_name}_Gen{generation}_ID_{id}/best_model")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv(
        [lambda: Environment(vertex, training=False, render_mode="human", design_vector=design_vector)])
    test_env = VecNormalize.load(f"normalise_stats/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl",
                                 test_env)
    test_env.training = False

    # Run test episodes
    N = 3
    for ep in range(1, N + 1):
        obs = test_env.reset()
        done = False
        total_reward = 0.0
        print(f"Starting visual test episode {ep}")

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            test_env.render()

        print(f"Episode {ep} finished with total reward {total_reward}")

    test_env.close()


if __name__ == "__main__":
    # Run evolutionary algorithm
    best_design, best_fitness = evolutionary_algorithm()

    # Optionally test the best design visually
    # Find which generation and ID had the best design
    print(f"\nBest design found: {best_design}")
    print(f"Best fitness: {best_fitness}")

    # You can uncomment the line below to visually test the best design
    # test_best_design(best_design, best_generation, best_id)