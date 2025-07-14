
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
from natsort import natsorted

shape_name, vertex = "Equi Triangle", [(-30, -30), (30, -30), (0, 30)] # "Square", [(-30, -30), (30, -30), (30, 30), (-30, 30)] #"Wide_rectangle", [(-200, -10), (200, -10), (200, 10), (-200, 10)]  # "Equi Triangle", [(-30, -30), (30, -30), (0, 30)] #

# Evolutionary Algorithm Parameters
POPULATION_SIZE = 10
NUM_GENERATIONS = 10
MUTATION_RATE = 0.2
CROSSOVER_RATE = 0.7
ELITE_SIZE = 2  # Number of best individuals to keep unchanged
TOURNAMENT_SIZE = 3

# Design vector bounds
BOUNDS = {
    'base_width': (50, 600),
    'left_finger_1': (30, 200),
    'right_finger_1': (30, 200),
    'left_finger_2': (30, 200),
    'right_finger_2': (30, 200)
}


def train(id, design_vector, generation, experiment_id):
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
        def __init__(self, *args, vecnormalize, experiment_id, **kwargs):
            super().__init__(*args, **kwargs)
            self.vecnormalize = vecnormalize  # the training wrapper
            self.experiment_id = experiment_id  # store experiment ID for saving

        def _on_step(self) -> bool:
            # run the usual evaluation logic
            old_reward = self.best_mean_reward
            continue_training = super()._on_step()

            if self.best_mean_reward > old_reward:
                stats_dir = f"Experiments/{self.experiment_id}/normalise_stats"
                os.makedirs(stats_dir, exist_ok=True)
                self.vecnormalize.save(f"{stats_dir}/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl")

            return continue_training

    # Training envs (headless, parallel)
    train_env = SubprocVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)], start_method="spawn")
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

    # Env for saving best model
    eval_env_fast = DummyVecEnv([make_env(vertex, 0, design_vector, render=False)])
    eval_env_fast = VecNormalize(eval_env_fast, norm_obs=True, norm_reward=False, training=False)
    eval_env_fast.obs_rms = train_env.obs_rms

    models_dir = f"Experiments/{experiment_id}/models/{shape_name}_Gen{generation}_ID_{id}"
    os.makedirs(models_dir, exist_ok=True)
    best_ckpt = SaveBestWithStats(
        eval_env_fast,
        vecnormalize=train_env,
        experiment_id=experiment_id,
        best_model_save_path=models_dir,
        n_eval_episodes=10,
        eval_freq=20000,
        deterministic=True,
        render=False,
        verbose=0,
    )

    # Evaluation env (still a single window so you can watch)
    eval_env = DummyVecEnv([make_env(vertex, 0, design_vector, render=True)])  # Set to False for faster training
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    eval_env.obs_rms = train_env.obs_rms  # share running stats

    # Make callback to run 1 episode every eval_freq steps
    eval_callback = EvalCallback(eval_env, n_eval_episodes=1, eval_freq=10000000, render=True, verbose=0,
                                 deterministic=True)

    # Instantiate PPO on the train_env, pass the callback to learn()
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=256,  # 256 × 8 = 2048 steps / update
        batch_size=512,  # must divide N_ENVS × N_STEPS
        verbose=0,
        tensorboard_log=f"Experiments/{experiment_id}/ppo_gripper_tensorboard/",
        policy_kwargs=policy_kwargs,
        ent_coef=0.05,
        learning_rate=1e-3,
    )

    model.learn(total_timesteps=3000000, callback=[eval_callback, best_ckpt])
    print(f"Training complete and model saved for Gen{generation}_ID{id}")

    train_env.close()
    eval_env.close()

def evaluate_model(id, design_vector, generation, experiment_id):
    """Evaluate a trained model and return fitness score"""
    # Load the trained policy
    model_path = f"Experiments/{experiment_id}/models/{shape_name}_Gen{generation}_ID_{id}/best_model"
    model = PPO.load(model_path)

    # Create a fresh env in human‐render mode
    stats_path = f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl"
    test_env = DummyVecEnv(
        [lambda: Environment(vertex, training=False, render_mode="human", design_vector=design_vector)])
    test_env = VecNormalize.load(stats_path, test_env)
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
            test_env.render() if ep == 0 else None

            if info[0].get('success', False):
                success_count += 1

        returns.append(total_reward)

    avg_return = np.mean(returns)
    success_rate = success_count / N

    # Fitness function: combine average return and success rate
    fitness = avg_return  # + (success_rate * 100)  # Bonus for successful episodes

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

def save_generation_data(generation, population, best_individual, best_fitness, best_id, experiment_id):
    """Save generation data to file"""
    os.makedirs(f"Experiments/{experiment_id}/evolution_data", exist_ok=True)

    # Convert population format for consistency
    population_list = []
    fitness_values = []
    for id, (design, fitness) in population.items():
        population_list.append(design)
        fitness_values.append(fitness)

    data = {
        'generation': generation,
        'population': population_list,
        'fitness_values': fitness_values,
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'best_id': best_id,
        'timestamp': datetime.now().isoformat()
    }

    with open(f"Experiments/{experiment_id}/evolution_data/generation_{generation}.pkl", 'wb') as f:
        pickle.dump(data, f)

def evolutionary_algorithm(experiment_id):
    """Main evolutionary algorithm loop"""
    print("=== Starting Evolutionary Algorithm ===")
    print(f"Population size: {POPULATION_SIZE}")
    print(f"Number of generations: {NUM_GENERATIONS}")
    print(f"Mutation rate: {MUTATION_RATE}")
    print(f"Crossover rate: {CROSSOVER_RATE}")
    print(f"Elite size: {ELITE_SIZE}")
    print(f"Tournament size: {TOURNAMENT_SIZE}")

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
    global_best_id = 0

    # Evolution loop
    for generation in range(NUM_GENERATIONS):
        print(f"\n=== GENERATION {generation + 1}/{NUM_GENERATIONS} ===")

        # Evaluate population
        for id, (design_vector, _) in population.items():
            print(f"Training Gen{generation + 1}_ID{id} with design: {design_vector}")
            train(id, design_vector, generation + 1, experiment_id)

            print(f"Evaluating Gen{generation + 1}_ID{id}")
            fitness = evaluate_model(id, design_vector, generation + 1, experiment_id)
            population[id][1] = fitness

            print(f"Gen{generation + 1}_ID{id} - Fitness: {fitness:.2f}")

            # Update global best
            if fitness > global_best_fitness:
                global_best_fitness = fitness
                global_best_individual = design_vector
                global_best_generation = generation + 1
                global_best_id = id
                print(f"    *** NEW GLOBAL BEST: {fitness:.2f} ***")

        # Sort population by fitness
        sorted_population = sorted(population.items(), key=lambda x: x[1][1], reverse=True)

        # Print generation statistics
        fitnesses = [individual[1][1] for individual in sorted_population]

        print(f"\nGeneration {generation + 1} Statistics:")
        print(f"Best fitness: {max(fitnesses):.2f}")
        print(f"Average fitness: {np.mean(fitnesses):.2f}")
        print(f"Worst fitness: {min(fitnesses):.2f}")
        print(f"Std dev fitness: {np.std(fitnesses):.2f}")
        print(f"Best design: {sorted_population[0][1][0]}")
        print(f"Global best so far: {global_best_fitness:.2f} (Gen {global_best_generation})")

        # Find best individual in this generation
        gen_best_individual = sorted_population[0][1][0]
        gen_best_fitness = sorted_population[0][1][1]
        gen_best_id = sorted_population[0][0]

        # Save generation data
        save_generation_data(generation + 1, dict(sorted_population),
                             gen_best_individual, gen_best_fitness, gen_best_id, experiment_id)

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
    print("\n" + "=" * 60)
    print("EVOLUTIONARY ALGORITHM COMPLETE")
    print("=" * 60)
    print(f"Completed generations: {NUM_GENERATIONS}")
    print(f"Total evaluations: {POPULATION_SIZE * NUM_GENERATIONS}")
    print(f"Global best fitness: {global_best_fitness:.2f}")
    print(f"Global best design: {global_best_individual}")
    print(f"Found in generation: {global_best_generation}")
    print(f"Model ID: {global_best_id}")

    # Save final results
    final_results = {
        'global_best_fitness': global_best_fitness,
        'global_best_individual': list(global_best_individual),
        'global_best_generation': global_best_generation,
        'global_best_id': global_best_id,
        'completed_generations': NUM_GENERATIONS,
        'planned_generations': NUM_GENERATIONS,
        'total_evaluations': POPULATION_SIZE * NUM_GENERATIONS,
        'evaluations_per_generation': POPULATION_SIZE,
        'final_population': {k: [list(v[0]), v[1]] for k, v in population.items()},
        'parameters': {
            'population_size': POPULATION_SIZE,
            'num_generations': NUM_GENERATIONS,
            'mutation_rate': MUTATION_RATE,
            'crossover_rate': CROSSOVER_RATE,
            'elite_size': ELITE_SIZE,
            'tournament_size': TOURNAMENT_SIZE,
            'bounds': BOUNDS
        }
    }

    with open(f"Experiments/{experiment_id}/evolution_data/final_results_ga.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    return global_best_individual, global_best_fitness, global_best_generation, global_best_id

def test_best_design(design_vector, generation, id, experiment_id):
    """Test the best design with visual rendering"""
    print(f"\n=== Testing Best Design Visually ===")
    print(f"Design: {design_vector}")

    # Load the trained policy
    model = PPO.load(f"Experiments/{experiment_id}/models/{shape_name}_Gen{generation}_ID_{id}/best_model")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv(
        [lambda: Environment(vertex, training=False, render_mode="human", design_vector=design_vector)])
    stats_path = f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Gen{generation}_ID_{id}.pkl"
    test_env = VecNormalize.load(stats_path, test_env)
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

def analyze_ga_results(experiment_id):
    """Analyze and visualize GA optimization results"""
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_ga.pkl", 'rb') as f:
            results = pickle.load(f)

        print("\n=== GA Results Analysis ===")
        print(f"Best fitness achieved: {results['global_best_fitness']:.2f}")
        print(f"Best design: {tuple(results['global_best_individual'])}")
        print(f"Found in generation: {results['global_best_generation']}")
        print(f"Model ID: {results['global_best_id']}")
        print(f"Completed generations: {results['completed_generations']}/{results['planned_generations']}")
        print(f"Total evaluations used: {results['total_evaluations']}")
        print(f"Evaluations per generation: {results['evaluations_per_generation']}")

        # Load generation data for progression analysis
        generation_files = [f for f in os.listdir(f"Experiments/{experiment_id}/evolution_data") if
                            f.startswith("generation_") and f.endswith(".pkl")]
        generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        if generation_files:
            print(f"\nOptimization progression:")
            print("Gen | Best Fitness | Avg Fitness | ID  | Best Design")
            print("-" * 70)

            for gen_file in generation_files:
                with open(f"Experiments/{experiment_id}/evolution_data/{gen_file}", 'rb') as f:
                    gen_data = pickle.load(f)

                gen_num = gen_data['generation']
                best_fit = gen_data['best_fitness']
                best_id = gen_data.get('best_id', 'N/A')
                avg_fitness = np.mean(gen_data['fitness_values'])
                best_design = gen_data['best_individual']

                print(f"{gen_num:3d} | {best_fit:11.2f} | {avg_fitness:11.2f} | {best_id:3} | {best_design}")

        # Print algorithm parameters
        print(f"\nGA Parameters:")
        params = results['parameters']
        print(f"  Population size: {params['population_size']}")
        print(f"  Generations: {params['num_generations']}")
        print(f"  Mutation rate: {params['mutation_rate']}")
        print(f"  Crossover rate: {params['crossover_rate']}")
        print(f"  Elite size: {params['elite_size']}")
        print(f"  Tournament size: {params['tournament_size']}")

    except FileNotFoundError:
        print("No GA results found. Run the optimization first.")

def load_and_test_best_model(experiment_id):
    """
    Load the best model from GA optimization and test it visually.
    This function can be run independently after optimization is complete.
    """
    try:
        # Load the final results
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_ga.pkl", 'rb') as f:
            results = pickle.load(f)

        best_fitness = results['global_best_fitness']
        best_individual = tuple(results['global_best_individual'])
        best_generation = results['global_best_generation']
        best_id = results['global_best_id']

        print("=== Loading Best Model from GA Optimization ===")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Best design: {best_individual}")
        print(f"Found in generation: {best_generation}")
        print(f"Model ID: {best_id}")

        # Check if model files exist
        model_path = f"Experiments/{experiment_id}/models/{shape_name}_Gen{best_generation}_ID_{best_id}/best_model.zip"
        stats_path = f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Gen{best_generation}_ID_{best_id}.pkl"

        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return

        if not os.path.exists(stats_path):
            print(f"ERROR: Normalization stats file not found: {stats_path}")
            return

        print(f"Model path: {model_path}")
        print(f"Stats path: {stats_path}")

        # Test the best design
        test_best_design(best_individual, best_generation, best_id, experiment_id)

        return best_individual, best_fitness, best_generation, best_id

    except FileNotFoundError as e:
        print(f"ERROR: Could not load optimization results: {e}")
        print("Make sure you have run the GA optimization first.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading best model: {e}")
        return None

def get_best_model_info(experiment_id):
    """
    Get information about the best model without running visual tests.
    Useful for checking results without opening rendering windows.
    """
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_ga.pkl", 'rb') as f:
            results = pickle.load(f)

        best_info = {
            'fitness': results['global_best_fitness'],
            'design': tuple(results['global_best_individual']),
            'generation': results['global_best_generation'],
            'id': results['global_best_id'],
            'completed_generations': results['completed_generations'],
            'planned_generations': results['planned_generations'],
            'model_path': f"Experiments/{experiment_id}/models/{shape_name}_Gen{results['global_best_generation']}_ID_{results['global_best_id']}/best_model.zip",
            'stats_path': f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Gen{results['global_best_generation']}_ID_{results['global_best_id']}.pkl"
        }

        return best_info

    except FileNotFoundError:
        print("No GA results found. Run optimization first.")
        return None

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

if __name__ == "__main__":
    # Choose what to run:
    # Option 1: Run full GA optimization
    # Option 2: Load and test best model after optimization
    # Option 3: Just get best model info without running anything
    # Option 4: Analyze and visualize best result from each generation

    OPTION = 2  # Change this to 1, 2, or 3 as needed

    if OPTION == 1:
        # Run full GA optimization
        experiment_id = get_next_experiment_id()
        print(f"Starting new experiment with ID: {experiment_id}")
        best_design, best_fitness, best_generation, best_id = evolutionary_algorithm(experiment_id)
        analyze_ga_results(experiment_id)
    elif OPTION == 2:
        # Option 2: Load and test best model from most recent experiment
        experiment_id = 1 #get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            print(f"Loading experiment ID: {experiment_id}")
            load_and_test_best_model(experiment_id)
    elif OPTION == 3:
        # Option 3: Just get best model info from most recent experiment
        experiment_id = get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            print(f"Analyzing experiment ID: {experiment_id}")
            info = get_best_model_info(experiment_id)
            if info:
                print(f"Best design: {info['design']}")
                print(f"Best fitness: {info['fitness']:.2f}")
                print(f"Completed: {info['completed_generations']}/{info['planned_generations']} generations")
                print(f"Model path: {info['model_path']}")
                print(f"Stats path: {info['stats_path']}")
                print()

            analyze_ga_results(experiment_id)

    elif OPTION == 4:
        # Option 4: Analyze and visualize best result from each generation

        experiment_id = get_next_experiment_id() - 1

        files = os.listdir(f"Experiments/{experiment_id}/evolution_data")
        files_sorted = natsorted(files)

        for f in files_sorted:
            if f.startswith("generation_") and f.endswith(".pkl"):

                with open(f"Experiments/{experiment_id}/evolution_data/{f}", 'rb') as file:
                    data = pickle.load(file)

                best_fitness = data['best_fitness']
                best_individual = tuple(data['best_individual'])
                generation = data['generation']
                best_id = data['best_id']

                print("\n" + "=" * 50)
                print(f"=== Best model from Generation {generation} ===")
                print(f"Best fitness: {best_fitness:.2f}")
                print(f"Best design: {best_individual}")
                print(f"Model ID: {best_id}")

                # Test the best design
                test_best_design(best_individual, generation, best_id, experiment_id)






