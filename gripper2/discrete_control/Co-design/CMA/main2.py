from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from environment import Environment
import numpy as np
import pickle
import os
from datetime import datetime
import cma

shape_name, vertex = "Equi Triangle", [(-30, -30), (30, -30), (0, 30)]

# CMA-ES Parameters
LAMBDA = 12  # Population size (offspring per generation)
SIGMA0 = 30.0  # Initial standard deviation
MAX_GENERATIONS = 5  # Maximum number of complete generations
# MAX_EVALUATIONS will be calculated as LAMBDA * MAX_GENERATIONS

# Design vector bounds
BOUNDS = np.array([
    [50, 300],  # base_width
    [30, 200],  # left_finger_1
    [30, 200],  # right_finger_1
    [30, 200],  # left_finger_2
    [30, 200]  # right_finger_2
])

# Initial mean (center of search space)
INITIAL_MEAN = np.mean(BOUNDS, axis=1)


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
    test_env = DummyVecEnv(
        [lambda: Environment(vertex, training=False, render_mode="human", design_vector=design_vector)])
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
            test_env.render()  if ep == 1 else None

            if info[0].get('success', False):
                success_count += 1

        returns.append(total_reward)

    avg_return = np.mean(returns)
    success_rate = success_count / N

    # Fitness function: combine average return and success rate
    fitness = avg_return  # + (success_rate * 100)  # Bonus for successful episodes

    test_env.close()
    return fitness


def clip_to_bounds(x, bounds):
    """Clip solution to bounds"""
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def objective_function(design_vector, generation, id):
    """
    Objective function for CMA-ES optimization.
    Returns negative fitness since CMA-ES minimizes.
    """
    # Clip to bounds and round to integers
    clipped_design = clip_to_bounds(design_vector, BOUNDS)
    rounded_design = tuple(round(x, 0) for x in clipped_design)

    print(f"Evaluating Gen{generation}_ID{id} with design: {rounded_design}")

    # Train the model
    train(id, rounded_design, generation)

    # Evaluate the model
    fitness = evaluate_model(id, rounded_design, generation)

    print(f"Gen{generation}_ID{id} - Fitness: {fitness:.2f}")

    # Return negative fitness (CMA-ES minimizes)
    return -fitness


def save_generation_data(generation, population, fitness_values, best_individual, best_fitness, best_id, es):
    """Save generation data to file"""
    os.makedirs("evolution_data", exist_ok=True)

    data = {
        'generation': generation,
        'population': population.tolist(),
        'fitness_values': fitness_values.tolist(),
        'best_individual': best_individual.tolist(),
        'best_fitness': best_fitness,
        'best_id': best_id,
        'sigma': es.sigma,
        'mean': es.mean.tolist(),
        'timestamp': datetime.now().isoformat()
    }

    with open(f"evolution_data/generation_{generation}.pkl", 'wb') as f:
        pickle.dump(data, f)


def cmaes_optimization():
    """Main CMA-ES optimization loop"""
    # Calculate total evaluations based on generations
    total_evaluations = LAMBDA * MAX_GENERATIONS

    print("=== Starting CMA-ES Optimization ===")
    print(f"Population size (lambda): {LAMBDA}")
    print(f"Initial sigma: {SIGMA0}")
    print(f"Max generations: {MAX_GENERATIONS}")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Initial mean: {INITIAL_MEAN}")
    print(f"Bounds: {BOUNDS}")

    # Initialize CMA-ES
    es = cma.CMAEvolutionStrategy(INITIAL_MEAN, SIGMA0, {
        'popsize': LAMBDA,
        'bounds': [BOUNDS[:, 0].tolist(), BOUNDS[:, 1].tolist()],
        'maxfevals': total_evaluations,  # Set this high enough to not interfere
        'seed': 42,
        'verbose': 1
    })

    # Track best individual across all generations
    global_best_fitness = float('-inf')
    global_best_individual = None
    global_best_generation = 0
    global_best_id = 0

    generation = 0
    total_evaluation_count = 0

    # Evolution loop - run for exactly MAX_GENERATIONS complete generations
    while generation < MAX_GENERATIONS and not es.stop():
        generation += 1
        print(f"\n=== GENERATION {generation}/{MAX_GENERATIONS} ===")
        print(f"Current sigma: {es.sigma:.3f}")
        print(f"Current mean: {es.mean}")

        # Ask for new candidate solutions
        solutions = es.ask()

        # Evaluate ALL solutions in this generation
        fitness_values = []

        print(f"Evaluating {len(solutions)} individuals in generation {generation}...")

        for i, solution in enumerate(solutions):
            print(f"  Individual {i + 1}/{len(solutions)}")

            # Evaluate this solution
            fitness = objective_function(solution, generation, i)
            fitness_values.append(fitness)

            # Update global best (remember fitness is negative in CMA-ES)
            actual_fitness = -fitness
            if actual_fitness > global_best_fitness:
                global_best_fitness = actual_fitness
                global_best_individual = clip_to_bounds(solution, BOUNDS)
                global_best_generation = generation
                global_best_id = i
                print(f"    *** NEW GLOBAL BEST: {actual_fitness:.2f} ***")

            total_evaluation_count += 1

        # Tell CMA-ES the fitness values for this complete generation
        es.tell(solutions, fitness_values)

        # Print generation statistics
        actual_fitnesses = [-f for f in fitness_values]
        print(f"\nGeneration {generation} Complete:")
        print(f"  Best fitness: {max(actual_fitnesses):.2f}")
        print(f"  Average fitness: {np.mean(actual_fitnesses):.2f}")
        print(f"  Worst fitness: {min(actual_fitnesses):.2f}")
        print(f"  Std dev fitness: {np.std(actual_fitnesses):.2f}")
        print(f"  Total evaluations so far: {total_evaluation_count}")

        # Find best individual in this generation
        best_idx = np.argmax(actual_fitnesses)
        gen_best_individual = clip_to_bounds(solutions[best_idx], BOUNDS)
        gen_best_fitness = actual_fitnesses[best_idx]
        gen_best_id = best_idx

        print(f"  Best design this generation: {tuple(round(x, 0) for x in gen_best_individual)}")
        print(f"  Global best so far: {global_best_fitness:.2f} (Gen {global_best_generation})")

        # Save generation data
        save_generation_data(generation, np.array(solutions),
                             np.array(actual_fitnesses), gen_best_individual, gen_best_fitness, gen_best_id, es)

    # Final results
    print("\n" + "=" * 60)
    print("CMA-ES OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Completed generations: {generation}")
    print(f"Total evaluations: {total_evaluation_count}")
    print(f"Global best fitness: {global_best_fitness:.2f}")
    print(f"Global best design: {tuple(round(x, 0) for x in global_best_individual)}")
    print(f"Found in generation: {global_best_generation}")
    print(f"Model ID: {global_best_id}")

    # Check if we completed all planned generations
    if generation == MAX_GENERATIONS:
        print(f"✓ Successfully completed all {MAX_GENERATIONS} generations")
    else:
        print(f"⚠ Stopped early after {generation} generations due to CMA-ES convergence")

    # Print CMA-ES termination reason if any
    stop_conditions = es.stop()
    if stop_conditions:
        print(f"\nCMA-ES termination conditions:")
        for condition, status in stop_conditions.items():
            if status:
                print(f"  {condition}: {status}")

    # Save final results
    final_results = {
        'global_best_fitness': global_best_fitness,
        'global_best_individual': global_best_individual.tolist(),
        'global_best_generation': global_best_generation,
        'global_best_id': global_best_id,
        'completed_generations': generation,
        'planned_generations': MAX_GENERATIONS,
        'total_evaluations': total_evaluation_count,
        'evaluations_per_generation': LAMBDA,
        'final_sigma': es.sigma,
        'final_mean': es.mean.tolist(),
        'termination_conditions': stop_conditions,
        'parameters': {
            'lambda': LAMBDA,
            'sigma0': SIGMA0,
            'max_generations': MAX_GENERATIONS,
            'bounds': BOUNDS.tolist()
        }
    }

    with open("evolution_data/final_results_cmaes.pkl", 'wb') as f:
        pickle.dump(final_results, f)

    return global_best_individual, global_best_fitness, global_best_generation, global_best_id


def test_best_design(design_vector, generation, id):
    """Test the best design with visual rendering"""
    print(f"\n=== Testing Best Design Visually ===")
    rounded_design = tuple(round(x, 0) for x in design_vector)
    print(f"Design: {rounded_design}")

    # Load the trained policy
    model = PPO.load(f"models/{shape_name}_Gen{generation}_ID_{id}/best_model")

    # Create a fresh env in human‐render mode
    test_env = DummyVecEnv(
        [lambda: Environment(vertex, training=False, render_mode="human", design_vector=rounded_design)])
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


def analyze_cmaes_results():
    """Analyze and visualize CMA-ES optimization results"""
    try:
        with open("evolution_data/final_results_cmaes.pkl", 'rb') as f:
            results = pickle.load(f)

        print("\n=== CMA-ES Results Analysis ===")
        print(f"Best fitness achieved: {results['global_best_fitness']:.2f}")
        print(f"Best design: {tuple(round(x, 0) for x in results['global_best_individual'])}")
        print(f"Found in generation: {results['global_best_generation']}")
        print(f"Model ID: {results['global_best_id']}")
        print(f"Completed generations: {results['completed_generations']}/{results['planned_generations']}")
        print(f"Total evaluations used: {results['total_evaluations']}")
        print(f"Evaluations per generation: {results['evaluations_per_generation']}")
        print(f"Final sigma: {results['final_sigma']:.3f}")
        print(f"Final mean: {[round(x, 1) for x in results['final_mean']]}")

        # Load generation data for progression analysis
        generation_files = [f for f in os.listdir("evolution_data") if
                            f.startswith("generation_") and f.endswith(".pkl")]
        generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        if generation_files:
            print(f"\nOptimization progression:")
            print("Gen | Best Fitness | Avg Fitness | Sigma   | ID  | Mean Design")
            print("-" * 75)

            for gen_file in generation_files:
                with open(f"evolution_data/{gen_file}", 'rb') as f:
                    gen_data = pickle.load(f)

                gen_num = gen_data['generation']
                best_fit = gen_data['best_fitness']
                best_id = gen_data.get('best_id', 'N/A')
                sigma = gen_data['sigma']
                mean_design = [round(x, 0) for x in gen_data['mean']]
                avg_fitness = np.mean(gen_data['fitness_values'])

                print(
                    f"{gen_num:3d} | {best_fit:11.2f} | {avg_fitness:11.2f} | {sigma:7.3f} | {best_id:3} | {mean_design}")

        # Print efficiency stats
        if 'completed_generations' in results and 'planned_generations' in results:
            efficiency = (results['completed_generations'] / results['planned_generations']) * 100
            print(f"\nEfficiency: {efficiency:.1f}% of planned generations completed")

    except FileNotFoundError:
        print("No CMA-ES results found. Run the optimization first.")


def resume_optimization_from_generation(start_generation):
    """
    Resume optimization from a specific generation.
    Useful if optimization was interrupted.
    """
    try:
        # Load the generation data
        with open(f"evolution_data/generation_{start_generation}.pkl", 'rb') as f:
            gen_data = pickle.load(f)

        print(f"Resuming optimization from generation {start_generation}")
        print(f"Previous best fitness: {gen_data['best_fitness']:.2f}")

        # You would need to reconstruct the CMA-ES state here
        # This is more complex and depends on your specific needs
        print("Note: Full resume functionality would require saving/loading CMA-ES state")

    except FileNotFoundError:
        print(f"Generation {start_generation} data not found")


def load_and_test_best_model():
    """
    Load the best model from CMA-ES optimization and test it visually.
    This function can be run independently after optimization is complete.
    """
    try:
        # Load the final results
        with open("evolution_data/final_results_cmaes.pkl", 'rb') as f:
            results = pickle.load(f)

        best_fitness = results['global_best_fitness']
        best_individual = np.array(results['global_best_individual'])
        best_generation = results['global_best_generation']
        best_id = results['global_best_id']

        print("=== Loading Best Model from CMA-ES Optimization ===")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Best design: {tuple(round(x, 0) for x in best_individual)}")
        print(f"Found in generation: {best_generation}")
        print(f"Model ID: {best_id}")

        # Check if model files exist
        model_path = f"models/{shape_name}_Gen{best_generation}_ID_{best_id}/best_model.zip"
        stats_path = f"normalise_stats/vecnormalize_stats_{shape_name}_Gen{best_generation}_ID_{best_id}.pkl"

        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found: {model_path}")
            return

        if not os.path.exists(stats_path):
            print(f"ERROR: Normalization stats file not found: {stats_path}")
            return

        print(f"Model path: {model_path}")
        print(f"Stats path: {stats_path}")

        # Test the best design
        test_best_design(best_individual, best_generation, best_id)

        return best_individual, best_fitness, best_generation, best_id

    except FileNotFoundError as e:
        print(f"ERROR: Could not load optimization results: {e}")
        print("Make sure you have run the CMA-ES optimization first.")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error loading best model: {e}")
        return None


def get_best_model_info():
    """
    Get information about the best model without running visual tests.
    Useful for checking results without opening rendering windows.
    """
    try:
        with open("evolution_data/final_results_cmaes.pkl", 'rb') as f:
            results = pickle.load(f)

        best_info = {
            'fitness': results['global_best_fitness'],
            'design': tuple(round(x, 0) for x in results['global_best_individual']),
            'generation': results['global_best_generation'],
            'id': results['global_best_id'],
            'completed_generations': results['completed_generations'],
            'planned_generations': results['planned_generations'],
            'model_path': f"models/{shape_name}_Gen{results['global_best_generation']}_ID_{results['global_best_id']}/best_model.zip",
            'stats_path': f"normalise_stats/vecnormalize_stats_{shape_name}_Gen{results['global_best_generation']}_ID_{results['global_best_id']}.pkl"
        }

        return best_info

    except FileNotFoundError:
        print("No CMA-ES results found. Run optimization first.")
        return None


if __name__ == "__main__":
    # Choose what to run:

    # # Option 1: Run full CMA-ES optimization (comment out to skip)
    best_design, best_fitness, best_generation, best_id = cmaes_optimization()
    analyze_cmaes_results()

    # Option 2: Load and test best model (uncomment to use after optimization)
    # load_and_test_best_model()

    # Option 3: Just get best model info (uncomment to use)
    # info = get_best_model_info()
    # if info:
    #     print(f"Best design: {info['design']}")
    #     print(f"Best fitness: {info['fitness']:.2f}")
    #     print(f"Completed: {info['completed_generations']}/{info['planned_generations']} generations")
    #     print(f"Model path: {info['model_path']}")
    #     print(f"Stats path: {info['stats_path']}")

