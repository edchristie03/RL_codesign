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
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Product, WhiteKernel, Matern
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import warnings

warnings.filterwarnings('ignore')

shape_name, vertex = "Square", [(-30, -30), (30, -30), (30, 30), (-30, 30)] #"Equi Triangle", [(-30, -30), (30, -30), (0, 30)]

# Surrogate-based Co-Design Parameters
INITIAL_SAMPLES = 1000  # Initial design samples
K_CLUSTERS = 10  # Number of clusters
SAMPLES_PER_CLUSTER = 100  # Initial samples per cluster for GP training
POPULATION_SIZE = 10  # Population size to maintain
OFFSPRING_SIZE = 15  # Candidates generated per generation
TOP_CANDIDATES = 5  # Top candidates to evaluate each generation
NUM_GENERATIONS = 8  # Number of co-design generations
UCB_KAPPA = 2.0  # UCB exploration parameter
MUTATION_RATE = 0.3
CROSSOVER_RATE = 0.7
FINETUNE_THRESHOLD = 0.8  # Uncertainty threshold for fine-tuning

# Design vector bounds
BOUNDS = {
    'base_width': (50, 300),
    'left_finger_1': (30, 200),
    'right_finger_1': (30, 200),
    'left_finger_2': (30, 200),
    'right_finger_2': (30, 200)
}

class SurrogateCoDesign:
    def __init__(self, experiment_id):
        self.experiment_id = experiment_id
        self.clusters = {}  # {cluster_id: {'centroid': design, 'policy': model, 'members': [designs], 'generation': generation}}
        self.gp_data = []  # [(phi_k, d_ki, R_ki), ...]
        self.gp_model = None
        self.scaler = StandardScaler()
        self.population = []  # Current population of designs
        self.evaluated_designs = {}  # {design: {'fitness': float, 'cluster': int}}

        self.global_best_design = None
        self.global_best_fitness = float('-inf')
        self.global_best_cluster = None
        self.global_best_generation = 0

    def create_individual(self):
        """Create a random design vector within bounds"""
        design = (
            random.uniform(*BOUNDS['base_width']),
            random.uniform(*BOUNDS['left_finger_1']),
            random.uniform(*BOUNDS['right_finger_1']),
            random.uniform(*BOUNDS['left_finger_2']),
            random.uniform(*BOUNDS['right_finger_2'])
        )
        return tuple(round(x, 0) for x in design)

    def train_policy(self, design_vector, cluster_id, generation=0):
        """Train a PPO policy for a given design vector"""
        print(f"Training policy for cluster {cluster_id} with design: {design_vector}")

        N_ENVS = 8
        policy_kwargs = {'net_arch': [256, 256], "log_std_init": 2}

        def make_env(vertex, rank, design_vector, render=False):
            def _init():
                env = Environment(vertex, training=True, render_mode="human" if render else None,
                                  design_vector=design_vector)
                env = Monitor(env)
                return env

            return _init

        class SaveBestWithStats(EvalCallback):
            def __init__(self, *args, vecnormalize, experiment_id, cluster_id, **kwargs):
                super().__init__(*args, **kwargs)
                self.vecnormalize = vecnormalize
                self.experiment_id = experiment_id
                self.cluster_id = cluster_id

            def _on_step(self) -> bool:
                old_reward = self.best_mean_reward
                continue_training = super()._on_step()

                if self.best_mean_reward > old_reward:
                    stats_dir = f"Experiments/{self.experiment_id}/normalise_stats"
                    os.makedirs(stats_dir, exist_ok=True)
                    self.vecnormalize.save(f"{stats_dir}/vecnormalize_stats_{shape_name}_Cluster{self.cluster_id}_Gen{generation}.pkl")

                return continue_training

        # Training envs
        train_env = SubprocVecEnv([make_env(vertex, i, design_vector) for i in range(N_ENVS)], start_method="spawn")
        train_env = VecNormalize(train_env, norm_obs=True, norm_reward=False)

        # Evaluation env for saving best model
        eval_env_fast = DummyVecEnv([make_env(vertex, 0, design_vector, render=False)])
        eval_env_fast = VecNormalize(eval_env_fast, norm_obs=True, norm_reward=False, training=False)
        eval_env_fast.obs_rms = train_env.obs_rms

        models_dir = f"Experiments/{self.experiment_id}/models/{shape_name}_Cluster{cluster_id}_Gen{generation}"
        os.makedirs(models_dir, exist_ok=True)

        best_ckpt = SaveBestWithStats(
            eval_env_fast,
            vecnormalize=train_env,
            experiment_id=self.experiment_id,
            cluster_id=cluster_id,
            best_model_save_path=models_dir,
            n_eval_episodes=10,
            eval_freq=10000,
            deterministic=True,
            render=False,
            verbose=0,
        )

        # Instantiate PPO
        model = PPO(
            "MlpPolicy",
            train_env,
            n_steps=256,
            batch_size=512,
            verbose=0,
            tensorboard_log=f"Experiments/{self.experiment_id}/ppo_gripper_tensorboard/",
            policy_kwargs=policy_kwargs,
            ent_coef=0.05,
            learning_rate=1e-3,
        )

        model.learn(total_timesteps=3000000, callback=[best_ckpt])  # Reduced for faster clustering phase

        train_env.close()
        eval_env_fast.close()

        # Load the best model
        best_model = PPO.load(f"{models_dir}/best_model")
        return best_model

    def evaluate_design(self, design_vector, policy_model, cluster_id):
        """Evaluate a design using a transferred policy"""

        # print(f"Evaluating design {design_vector} with cluster {cluster_id} policy")

        # Create test environment
        test_env = DummyVecEnv(
            [lambda: Environment(vertex, training=False, render_mode=None, design_vector=design_vector)])

        # Load normalization stats from the cluster's training
        cluster_generation = self.clusters[cluster_id]['generation']
        stats_path = self.get_stats_path(cluster_id, cluster_generation)

        if os.path.exists(stats_path):
            test_env = VecNormalize.load(stats_path, test_env)
        else:
            test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, training=False)

        test_env.training = False

        # Run evaluation episodes
        N = 10
        returns = []
        success_count = 0

        for ep in range(N):
            obs = test_env.reset()
            done = False
            total_reward = 0.0

            while not done:
                action, _ = policy_model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward

                if info[0].get('success', False):
                    success_count += 1

            returns.append(total_reward)

        avg_return = np.mean(returns)
        success_rate = success_count / N
        fitness = avg_return  # * (0.1 + success_rate) # Could incorporate success rate

        test_env.close()
        return fitness, success_rate

    def crossover(self, parent1, parent2):
        """Perform uniform crossover"""
        child = []
        for i in range(len(parent1)):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return tuple(child)

    def mutate(self, individual):
        """Mutate an individual with Gaussian noise"""
        mutated = []
        bounds_list = list(BOUNDS.values())

        for i, value in enumerate(individual):
            if random.random() < MUTATION_RATE:
                # Add Gaussian noise
                noise_scale = (bounds_list[i][1] - bounds_list[i][0]) * 0.1
                new_value = value + random.gauss(0, noise_scale)
                # Clamp to bounds
                new_value = max(bounds_list[i][0], min(bounds_list[i][1], new_value))
                mutated.append(round(new_value, 0))
            else:
                mutated.append(value)

        return tuple(mutated)

    def find_nearest_cluster(self, design):
        """Find the nearest cluster centroid to a design"""
        distances = []
        for cluster_id, cluster_info in self.clusters.items():
            centroid = cluster_info['centroid']
            dist = np.linalg.norm(np.array(design) - np.array(centroid))
            distances.append((cluster_id, dist))

        distances.sort(key=lambda x: x[1])
        return distances[0][0]

    def get_model_path(self, cluster_id, generation, best_model=True):
        """Get the path to a cluster's model"""
        model_dir = f"Experiments/{self.experiment_id}/models/{shape_name}_Cluster{cluster_id}_Gen{generation}"
        if best_model:
            return f"{model_dir}/best_model"
        return model_dir

    def get_stats_path(self, cluster_id, generation):
        """Get the path to a cluster's normalization stats"""
        return f"Experiments/{self.experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Cluster{cluster_id}_Gen{generation}.pkl"

    def update_global_best(self, generation):
        """Update global best if current generation has better solutions"""
        # Check all evaluated designs for improvements
        for design, info in self.evaluated_designs.items():
            if info['fitness'] > self.global_best_fitness:
                self.global_best_fitness = info['fitness']
                self.global_best_design = design
                self.global_best_cluster = info['cluster']
                self.global_best_generation = generation
                print(f"NEW GLOBAL BEST in generation {generation}: {design} with fitness {info['fitness']:.2f}")

    def _validate_gp_predictions(self, X_scaled, y_scaled):
        """Validate GP predictions to ensure it's learning properly"""
        if len(X_scaled) < 10:
            return

        # Make predictions on training data
        y_pred, y_std = self.gp_model.predict(X_scaled, return_std=True)

        # Calculate metrics
        mse = np.mean((y_pred - y_scaled) ** 2)
        mae = np.mean(np.abs(y_pred - y_scaled))
        r2 = 1 - np.var(y_pred - y_scaled) / np.var(y_scaled)

        print(f"GP Validation on training data:")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Std range: [{np.min(y_std):.4f}, {np.max(y_std):.4f}]")

        # Check if uncertainty is varying
        if np.std(y_std) < 1e-6:
            print("WARNING: GP uncertainty is nearly constant - model may not be learning spatial structure!")

    def predict_with_gp(self, phi_k, d_ki):
        """Make predictions with proper scaling"""
        if self.gp_model is None:
            return 0.0, 1.0

        # Prepare input
        x_sample = np.array([list(phi_k) + list(d_ki)])
        x_scaled = self.scaler.transform(x_sample)

        # Make prediction
        y_pred_scaled, y_std_scaled = self.gp_model.predict(x_scaled, return_std=True)

        # Transform back to original scale
        y_pred = y_pred_scaled[0] * self.y_std + self.y_mean
        y_std = y_std_scaled[0] * self.y_std

        return y_pred, y_std

    def step1_cluster_and_policy_prep(self):
        """Sample initial designs, cluster them, and train policies on centroids"""
        print("=== Step 1: Cluster & Policy Prep ===")

        # Sample initial designs
        initial_designs = [self.create_individual() for _ in range(INITIAL_SAMPLES)]
        print(f"Generated {len(initial_designs)} initial designs")

        # Cluster designs using K-means
        design_array = np.array(initial_designs)
        kmeans = KMeans(n_clusters=K_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(design_array)
        centroids = kmeans.cluster_centers_

        # Round centroids to valid design vectors
        centroids = [tuple(int(x) for x in centroid) for centroid in centroids]

        # Train policies on centroids
        for k in range(K_CLUSTERS):
            cluster_members = [initial_designs[i] for i in range(len(initial_designs)) if cluster_labels[i] == k]
            centroid = centroids[k]

            print(f"Cluster {k}: {len(cluster_members)} members, centroid: {centroid}")

            # Train policy on centroid and evaluate
            policy = self.train_policy(centroid, k, generation=0)

            self.clusters[k] = {
                'centroid': centroid,
                'policy': policy,
                'members': cluster_members,
                'generation': 0  # Initial clusters are generation 0
            }

            fitness, success_rate = self.evaluate_design(centroid, policy, k)
            print(f"  Actual fitness: {fitness:.2f}, Success rate: {success_rate:.2f}")

        print(f"Trained {len(self.clusters)} cluster policies")
        return initial_designs

    def step2_surrogate_dataset(self):
        """Create surrogate dataset by evaluating cluster members"""
        print("=== Step 2: Surrogate Dataset ===")

        for cluster_id, cluster_info in self.clusters.items():
            centroid = cluster_info['centroid']
            policy = cluster_info['policy']
            members = cluster_info['members']

            # Sample a few members from this cluster
            sample_members = random.sample(members, min(SAMPLES_PER_CLUSTER, len(members)))

            for member_design in sample_members:
                fitness, success_rate = self.evaluate_design(member_design, policy, cluster_id)

                # Add to GP dataset: (phi_k, d_ki, R_ki)
                self.gp_data.append((centroid, member_design, fitness))
                self.evaluated_designs[member_design] = {'fitness': fitness, 'cluster': cluster_id}

                print(f"Cluster {cluster_id}: {member_design} -> fitness {fitness:.2f}, success rate {success_rate:.2f}")

        print(f"Created surrogate dataset with {len(self.gp_data)} samples")

    # def step3_fit_joint_gp(self):
    #     """Train Gaussian Process on the surrogate dataset"""
    #     print("=== Step 3: Fit Joint GP ===")
    #
    #     if len(self.gp_data) == 0:
    #         print("No GP data available!")
    #         return
    #
    #     # Prepare training data
    #     X = []  # [phi_k; d_ki] concatenated
    #     y = []  # R_ki
    #
    #     for phi_k, d_ki, R_ki in self.gp_data:
    #         # Concatenate centroid and design
    #         x_sample = list(phi_k) + list(d_ki)
    #         X.append(x_sample)
    #         y.append(R_ki)
    #
    #     X = np.array(X)
    #     y = np.array(y)
    #
    #     # Standardize inputs
    #     X_scaled = self.scaler.fit_transform(X)
    #
    #     # Create kernel: RBF x RBF + noise
    #     kernel = Product(RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2)),
    #                      RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))) + WhiteKernel(noise_level=1.0)
    #
    #     # Train GP
    #     self.gp_model = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, n_restarts_optimizer=10, normalize_y=True)
    #     self.gp_model.fit(X_scaled, y)
    #
    #     print(f"Trained GP on {len(X)} samples")
    #     print(f"GP kernel: {self.gp_model.kernel_}")

    def step3_fit_joint_gp(self):
        """Train Gaussian Process on the surrogate dataset with improved fitting"""
        print("=== Step 3: Fit Joint GP ===")

        if len(self.gp_data) == 0:
            print("No GP data available!")
            return

        # Prepare training data
        X = []  # [phi_k; d_ki] concatenated
        y = []  # R_ki

        for phi_k, d_ki, R_ki in self.gp_data:
            # Concatenate centroid and design
            x_sample = list(phi_k) + list(d_ki)
            X.append(x_sample)
            y.append(R_ki)

        X = np.array(X)
        y = np.array(y)

        # Check for valid data
        if len(X) < 2:
            print("Not enough data for GP training!")
            return

        # Remove any infinite or NaN values
        valid_mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = X[valid_mask]
        y = y[valid_mask]

        print(f"Using {len(X)} valid samples out of {len(self.gp_data)} total")

        # Standardize inputs and outputs
        X_scaled = self.scaler.fit_transform(X)

        # Standardize outputs for better numerical stability
        y_mean = np.mean(y)
        y_std = np.std(y)
        y_scaled = (y - y_mean) / (y_std + 1e-8)  # Add small epsilon to avoid division by zero

        # Store scaling parameters for later use
        self.y_mean = y_mean
        self.y_std = y_std

        # Estimate reasonable noise level from data
        # Use a fraction of the output variance as initial noise estimate
        initial_noise = max(np.var(y_scaled) * 0.01, 1e-6)  # 1% of variance, minimum 1e-6
        noise_level_bounds = (1e-8,1e-2) #(1e-8, 1e0)  # Reasonable bounds for noise level

        # Create kernel with better bounds and initialization
        # Use separate RBF kernels for different parts of the input space
        n_dims = X.shape[1]

        # Try different kernel configurations
        kernels_to_try = [
            # Single RBF kernel
            RBF(length_scale=1.0, length_scale_bounds=(1e-3, 1e3)) +
            WhiteKernel(noise_level=initial_noise, noise_level_bounds=noise_level_bounds),

            # ARD (Automatic Relevance Determination) kernel
            RBF(length_scale=[1.0] * n_dims, length_scale_bounds=(1e-3, 1e3)) +
            WhiteKernel(noise_level=initial_noise, noise_level_bounds=noise_level_bounds),

            # Matern kernel (often more robust than RBF)
            Matern(length_scale=1.0, length_scale_bounds=(1e-3, 1e3), nu=2.5) +
            WhiteKernel(noise_level=initial_noise, noise_level_bounds=noise_level_bounds),
        ]

        best_gp = None
        best_score = -np.inf

        # Try different kernels and pick the best one
        for i, kernel in enumerate(kernels_to_try):
            try:
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    alpha=1e-10,  # Very small alpha since we have explicit noise kernel
                    n_restarts_optimizer=20,  # More restarts for better optimization
                    normalize_y=False,  # We're doing our own normalization
                    random_state=42
                )

                gp.fit(X_scaled, y_scaled)

                # Evaluate kernel using log marginal likelihood
                score = gp.log_marginal_likelihood()

                print(f"Kernel {i + 1}: Log marginal likelihood = {score:.2f}")
                print(f"  Kernel: {gp.kernel_}")

                if score > best_score:
                    best_score = score
                    best_gp = gp

            except Exception as e:
                print(f"Kernel {i + 1} failed: {e}")
                continue

        if best_gp is None:
            print("All kernels failed! Using simple fallback.")
            # Fallback to simple kernel
            kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
            best_gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
            best_gp.fit(X_scaled, y)
            self.y_mean = 0
            self.y_std = 1

        self.gp_model = best_gp

        print(f"Best GP kernel: {self.gp_model.kernel_}")
        print(f"Log marginal likelihood: {best_score:.2f}")

        # Validate the GP by checking predictions on training data
        self._validate_gp_predictions(X_scaled, y_scaled)

    def step4_maintain_population(self, initial_designs):
        """Initialize population from initial designs and centroids"""
        print("=== Step 4: Maintain Population ===")

        # Start with evaluated designs
        candidates = list(self.evaluated_designs.keys())

        # Add centroids if not already evaluated
        for cluster_info in self.clusters.values():
            centroid = cluster_info['centroid']
            if centroid not in self.evaluated_designs:
                candidates.append(centroid)

        # Fill remaining spots with random initial designs
        remaining_designs = [d for d in initial_designs if d not in candidates]
        candidates.extend(remaining_designs)

        # Select top N by fitness (evaluate missing ones with nearest cluster)
        population_candidates = []
        for design in candidates[:POPULATION_SIZE * 2]:  # Evaluate more than needed
            if design in self.evaluated_designs:
                fitness = self.evaluated_designs[design]['fitness']
            else:
                # Use nearest cluster for quick evaluation
                nearest_cluster = self.find_nearest_cluster(design)
                fitness, success_rate = self.evaluate_design(design, self.clusters[nearest_cluster]['policy'], nearest_cluster)
                self.evaluated_designs[design] = {'fitness': fitness, 'cluster': nearest_cluster}

            population_candidates.append((design, fitness))

        # Sort by fitness and take top N
        population_candidates.sort(key=lambda x: x[1], reverse=True)
        self.population = [design for design, _ in population_candidates[:POPULATION_SIZE]]

        print(f"Initialized population with {len(self.population)} designs")
        print(f"Population fitness range: {population_candidates[0][1]:.2f} to {population_candidates[POPULATION_SIZE - 1][1]:.2f}")

    def step5_propose_offspring(self):
        """Generate offspring through mutation and crossover"""
        print("=== Step 5: Propose Offspring ===")

        # Sort population by fitness
        pop_with_fitness = [(design, self.evaluated_designs[design]['fitness'])
                            for design in self.population if design in self.evaluated_designs]
        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Select top performers for breeding
        top_performers = [design for design, _ in pop_with_fitness[:POPULATION_SIZE // 2]]

        offspring = []
        for _ in range(OFFSPRING_SIZE):
            if random.random() < CROSSOVER_RATE and len(top_performers) >= 2:
                # Crossover
                parent1, parent2 = random.sample(top_performers, 2)
                child = self.crossover(parent1, parent2)
            else:
                # Mutation
                parent = random.choice(top_performers)
                child = self.mutate(parent)

            offspring.append(child)

        print(f"Generated {len(offspring)} offspring")
        return offspring

    def step6_surrogate_scoring(self, candidates):
        """Score candidates using GP with UCB"""
        print("=== Step 6: Surrogate Scoring ===")

        if self.gp_model is None:
            print("GP model not trained!")
            return []

        scored_candidates = []

        for candidate in candidates:
            # Find nearest cluster
            nearest_cluster = self.find_nearest_cluster(candidate)
            centroid = self.clusters[nearest_cluster]['centroid']

            # Prepare input for GP: [phi_k; d]
            x_input = list(centroid) + list(candidate)
            x_input = np.array(x_input).reshape(1, -1)
            x_input_scaled = self.scaler.transform(x_input)

            # Get GP prediction
            mu, sigma = self.gp_model.predict(x_input_scaled, return_std=True)

            # UCB score
            ucb_score = mu[0] + UCB_KAPPA * sigma[0]

            scored_candidates.append((candidate, ucb_score, mu[0], sigma[0], nearest_cluster))

        # Sort by UCB score
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        print(f"Scored {len(scored_candidates)} candidates")
        print("15 candidates by UCB score:")

        for i, (candidate, ucb, mu, sigma, cluster) in enumerate(scored_candidates):
            print(f"  {i + 1}. UCB={ucb:.2f} (μ={mu:.2f}, σ={sigma:.2f}), cluster={cluster}, design={candidate}")

        return scored_candidates

    def step7_select_and_evaluate(self, scored_candidates, generation):
        """Select top candidates and evaluate them"""
        print("=== Step 7: Select & Evaluate ===")

        # Take top m candidates
        top_candidates = scored_candidates[:TOP_CANDIDATES]
        evaluated_candidates = []

        for candidate, ucb_score, mu, sigma, cluster_id in top_candidates:
            # print(f"Evaluating candidate: {candidate} (UCB={ucb_score:.2f}, mu={mu:.2f} ,σ={sigma:.2f})")

            # Decide whether to fine-tune based on uncertainty
            fine_tune = sigma > FINETUNE_THRESHOLD

            if fine_tune:
                new_cluster_id = max(self.clusters.keys()) + 1
                print(f"High Uncertainty: Training new policy for candidate {candidate} with cluster ID {new_cluster_id}")
                # retrain from scratch
                new_policy = self.train_policy(candidate, new_cluster_id, generation)
                # store it as its own “cluster”:

                self.clusters[new_cluster_id] = {
                    'centroid': candidate,
                    'policy': new_policy,
                    'members': [candidate],
                    'generation': generation
                }

                policy = new_policy
                cluster_id = new_cluster_id
            else:
                policy = self.clusters[cluster_id]['policy']

            fitness, success_rate = self.evaluate_design(candidate, policy, cluster_id)
            evaluated_candidates.append((candidate, fitness, cluster_id, generation))
            print(f"  Actual fitness: {fitness:.2f}, Success rate: {success_rate:.2f}")

        return evaluated_candidates

    def step8_update_surrogate(self, evaluated_candidates):
        """Update GP with new evaluation data"""
        print("=== Step 8: Update Surrogate ===")

        for candidate, fitness, cluster_id, generation in evaluated_candidates:
            centroid = self.clusters[cluster_id]['centroid']

            # Add to GP dataset
            self.gp_data.append((centroid, candidate, fitness))
            self.evaluated_designs[candidate] = {'fitness': fitness, 'cluster': cluster_id, 'cluster_generation': generation}

        # Refit GP
        self.step3_fit_joint_gp()

        print(f"Updated GP with {len(evaluated_candidates)} new samples")
        print(f"Total GP dataset size: {len(self.gp_data)}")

    def step9_evolve_population(self, evaluated_candidates):
        """Merge candidates into population and keep top N"""
        print("=== Step 9: Evolve Population ===")

        # Add new candidates to population
        for candidate, fitness, cluster_id, generation in evaluated_candidates:
            if candidate not in self.population:
                self.population.append(candidate)

        # Sort all population members by fitness
        pop_with_fitness = []
        for design in self.population:
            if design in self.evaluated_designs:
                fitness = self.evaluated_designs[design]['fitness']
                pop_with_fitness.append((design, fitness))

        pop_with_fitness.sort(key=lambda x: x[1], reverse=True)

        # Keep top N
        self.population = [design for design, _ in pop_with_fitness[:POPULATION_SIZE]]

        print(f"Updated population (size {len(self.population)})")
        print(f"Fitness range: {pop_with_fitness[0][1]:.2f} to {pop_with_fitness[POPULATION_SIZE - 1][1]:.2f}")

    def save_generation_data(self, generation):
        """Save generation data"""
        os.makedirs(f"Experiments/{self.experiment_id}/evolution_data", exist_ok=True)

        # Get population fitness data
        pop_fitness = {}
        for design in self.population:
            if design in self.evaluated_designs:
                pop_fitness[design] = self.evaluated_designs[design]['fitness']

        # Find best
        best_design = max(pop_fitness.keys(), key=lambda x: pop_fitness[x])
        best_fitness = pop_fitness[best_design]

        data = {
            'generation': generation,
            'population': self.population,
            'population_fitness': pop_fitness,
            'best_individual': best_design,
            'best_fitness': best_fitness,
            'global_best_design': self.global_best_design,  # Add this
            'global_best_fitness': self.global_best_fitness,  # Add this
            'global_best_generation': self.global_best_generation,  # Add this
            'gp_data_size': len(self.gp_data),
            'evaluated_designs_count': len(self.evaluated_designs),
            'timestamp': datetime.now().isoformat()
        }

        with open(f"Experiments/{self.experiment_id}/evolution_data/generation_{generation}.pkl", 'wb') as f:
            pickle.dump(data, f)

    def run_codesign_loop(self):
        """Run the complete surrogate-based co-design algorithm"""
        print("=== SURROGATE-BASED CO-DESIGN ALGORITHM ===")

        # Steps 1-4: Initialization
        initial_designs = self.step1_cluster_and_policy_prep()
        self.step2_surrogate_dataset()
        self.step3_fit_joint_gp()
        self.step4_maintain_population(initial_designs)

        # Save initial state
        self.save_generation_data(0)

        self.update_global_best(0)

        # Main co-design loop
        for generation in range(1, NUM_GENERATIONS + 1):
            print(f"\n{'=' * 60}")
            print(f"GENERATION {generation}/{NUM_GENERATIONS}")
            print('=' * 60)

            # Step 5: Generate offspring
            offspring = self.step5_propose_offspring()

            # Step 6: Score with surrogate
            scored_candidates = self.step6_surrogate_scoring(offspring)

            # Step 7: Evaluate top candidates
            evaluated_candidates = self.step7_select_and_evaluate(scored_candidates, generation)

            # Step 8: Update surrogate
            self.step8_update_surrogate(evaluated_candidates)

            self.update_global_best(generation)

            # Step 9: Evolve population
            self.step9_evolve_population(evaluated_candidates)

            # Save generation data
            self.save_generation_data(generation)

            # Print generation statistics
            pop_fitness = [self.evaluated_designs[d]['fitness'] for d in self.population
                           if d in self.evaluated_designs]
            if pop_fitness:
                print(f"Generation {generation} Statistics:")
                print(f"  Best fitness: {max(pop_fitness):.2f}")
                print(f"  Average fitness: {np.mean(pop_fitness):.2f}")
                print(f"  Worst fitness: {min(pop_fitness):.2f}")
                print(f"  GP dataset size: {len(self.gp_data)}")

        # Step 10: Return best design
        best_design = self.global_best_design
        best_fitness = self.global_best_fitness
        best_cluster = self.global_best_cluster

        print(f"\n{'=' * 60}")
        print("CO-DESIGN OPTIMIZATION COMPLETE")
        print('=' * 60)
        print(f"Best design: {best_design}")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Best cluster: {best_cluster}")
        print(f"Total evaluations: {len(self.evaluated_designs)}")
        print(f"GP dataset size: {len(self.gp_data)}")

        # Save final results
        final_results = {
            'best_design': best_design,
            'best_fitness': best_fitness,
            'best_cluster': best_cluster,
            'population': self.population,
            'evaluated_designs': self.evaluated_designs,
            'gp_data_size': len(self.gp_data),
            'total_evaluations': len(self.evaluated_designs),
            'clusters': {k: {'centroid': v['centroid'], 'generation': v['generation']} for k, v in self.clusters.items()},
            'parameters': {
                'initial_samples': INITIAL_SAMPLES,
                'k_clusters': K_CLUSTERS,
                'population_size': POPULATION_SIZE,
                'offspring_size': OFFSPRING_SIZE,
                'top_candidates': TOP_CANDIDATES,
                'num_generations': NUM_GENERATIONS,
                'ucb_kappa': UCB_KAPPA,
                'mutation_rate': MUTATION_RATE,
                'crossover_rate': CROSSOVER_RATE,
                'finetune_threshold': FINETUNE_THRESHOLD
            }
        }

        with open(f"Experiments/{self.experiment_id}/evolution_data/final_results_surrogate.pkl", 'wb') as f:
            pickle.dump(final_results, f)

        return best_design, best_fitness, best_cluster


def test_best_design_surrogate(experiment_id):
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_surrogate.pkl", 'rb') as f:
            results = pickle.load(f)

        best_design = results['best_design']
        best_cluster = results['best_cluster']

        # Get the generation for this cluster
        cluster_generation = results['clusters'][best_cluster]['generation']  # Note: might be string key

        print(f"Testing best design: {best_design} from cluster {best_cluster} (generation {cluster_generation})")

        # Use the actual generation in paths
        model_path = f"Experiments/{experiment_id}/models/{shape_name}_Cluster{best_cluster}_Gen{cluster_generation}/best_model"
        stats_path = f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Cluster{best_cluster}_Gen{cluster_generation}.pkl"

        model = PPO.load(model_path)

        # Create test environment
        test_env = DummyVecEnv(
            [lambda: Environment(vertex, training=False, render_mode="human", design_vector=best_design)])

        if os.path.exists(stats_path):
            test_env = VecNormalize.load(stats_path, test_env)
        else:
            test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, training=False)

        test_env.training = False

        # Run test episodes
        for ep in range(10):
            obs = test_env.reset()
            done = False
            total_reward = 0.0
            print(f"Starting test episode {ep + 1}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
                test_env.render()

            print(f"Episode {ep + 1} finished with total reward {total_reward}")

        test_env.close()

    except FileNotFoundError:
        print("No surrogate results found. Run optimization first.")

def analyze_surrogate_results(experiment_id):
    """Analyze surrogate optimization results"""
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_surrogate.pkl", 'rb') as f:
            results = pickle.load(f)

        print("\n=== Surrogate Co-Design Results ===")
        print(f"Best design: {results['best_design']}")
        print(f"Best fitness: {results['best_fitness']:.2f}")
        print(f"Best cluster: {results['best_cluster']}")
        print(f"Total evaluations: {results['total_evaluations']}")
        print(f"GP dataset size: {results['gp_data_size']}")

        # Print parameters
        params = results['parameters']
        print(f"\nAlgorithm Parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")

        # Print cluster info
        print(f"\nCluster Information:")
        for cluster_id, info in results['clusters'].items():
            print(f"  Cluster {cluster_id}: {info['centroid']}")

        # Load and print generation progression
        generation_files = [f for f in os.listdir(f"Experiments/{experiment_id}/evolution_data")
                            if f.startswith("generation_") and f.endswith(".pkl")]
        generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

        if generation_files:
            print(f"\nOptimization progression:")
            print("Gen | Best Fitness | Avg Fitness | GP Size | Evaluations")
            print("-" * 55)

            for gen_file in generation_files:
                with open(f"Experiments/{experiment_id}/evolution_data/{gen_file}", 'rb') as f:
                    gen_data = pickle.load(f)

                gen_num = gen_data['generation']
                best_fit = gen_data['best_fitness']
                pop_fitness = list(gen_data['population_fitness'].values())
                avg_fitness = np.mean(pop_fitness) if pop_fitness else 0
                gp_size = gen_data.get('gp_data_size', 0)
                eval_count = gen_data.get('evaluated_designs_count', 0)

                print(f"{gen_num:3d} | {best_fit:11.2f} | {avg_fitness:11.2f} | {gp_size:7d} | {eval_count:11d}")

    except FileNotFoundError:
        print("No surrogate results found. Run optimization first.")

def get_best_surrogate_model_info(experiment_id):
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_surrogate.pkl", 'rb') as f:
            results = pickle.load(f)

        best_cluster = results['best_cluster']
        cluster_generation = results['clusters'][str(best_cluster)]['generation']

        best_info = {
            'design': results['best_design'],
            'fitness': results['best_fitness'],
            'cluster': results['best_cluster'],
            'cluster_generation': cluster_generation,  # Add this
            'total_evaluations': results['total_evaluations'],
            'gp_data_size': results['gp_data_size'],
            'model_path': f"Experiments/{experiment_id}/models/{shape_name}_Cluster{best_cluster}_Gen{cluster_generation}/best_model.zip",
            'stats_path': f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Cluster{best_cluster}_Gen{cluster_generation}.pkl"
        }

        return best_info

    except FileNotFoundError:
        print("No surrogate results found. Run optimization first.")
        return None

def test_generation_best_surrogate(experiment_id, generation):
    """Test the best model from a specific generation"""
    try:
        with open(f"Experiments/{experiment_id}/evolution_data/generation_{generation}.pkl", 'rb') as f:
            gen_data = pickle.load(f)

        best_design = gen_data['best_individual']
        best_fitness = gen_data['best_fitness']

        print(f"Testing best design from generation {generation}")
        print(f"Design: {best_design}")
        print(f"Fitness: {best_fitness:.2f}")

        # Find which cluster this design belongs to
        codesign = SurrogateCoDesign(experiment_id)

        # Load cluster information from final results to get centroids
        with open(f"Experiments/{experiment_id}/evolution_data/final_results_surrogate.pkl", 'rb') as f:
            results = pickle.load(f)

        # Reconstruct clusters for finding nearest
        codesign.clusters = {}
        for cluster_id, info in results['clusters'].items():
            codesign.clusters[int(cluster_id)] = {'centroid': info['centroid']}

        nearest_cluster = codesign.find_nearest_cluster(best_design)

        # Load and test with the cluster's policy
        model_path = f"Experiments/{experiment_id}/models/{shape_name}_Cluster{nearest_cluster}_Gen0/best_model"
        model = PPO.load(model_path)

        test_env = DummyVecEnv(
            [lambda: Environment(vertex, training=False, render_mode="human", design_vector=best_design)])
        stats_path = f"Experiments/{experiment_id}/normalise_stats/vecnormalize_stats_{shape_name}_Cluster{nearest_cluster}_Gen0.pkl"

        if os.path.exists(stats_path):
            test_env = VecNormalize.load(stats_path, test_env)
        else:
            test_env = VecNormalize(test_env, norm_obs=True, norm_reward=False, training=False)

        test_env.training = False

        # Run test episodes
        for ep in range(3):
            obs = test_env.reset()
            done = False
            total_reward = 0.0
            print(f"Starting test episode {ep + 1}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = test_env.step(action)
                total_reward += reward
                test_env.render()

            print(f"Episode {ep + 1} finished with total reward {total_reward}")

        test_env.close()

    except FileNotFoundError:
        print(f"Generation {generation} data not found.")

def get_next_experiment_id():
    """Get the next experiment ID based on existing directories"""
    if not os.path.exists("Experiments"):
        os.makedirs("Experiments")
        return 1

    existing_ids = [int(f) for f in os.listdir("Experiments") if f.isdigit()]
    return max(existing_ids, default=0) + 1

if __name__ == "__main__":
    # Choose what to run:
    # Option 1: Run full surrogate co-design optimization
    # Option 2: Load and test best model after optimization
    # Option 3: Just get best model info without running anything
    # Option 4: Analyze surrogate results
    # Option 5: Test best model from a specific generation

    OPTION = 1 # Change this to 1, 2, 3, 4, or 5 as needed

    if OPTION == 1:
        # Run full surrogate co-design optimization
        experiment_id = get_next_experiment_id()
        print(f"Starting new surrogate co-design experiment with ID: {experiment_id}")

        codesign = SurrogateCoDesign(experiment_id)
        best_design, best_fitness, best_cluster = codesign.run_codesign_loop()

        print(f"\nOptimization complete!")
        print(f"Best design: {best_design}")
        print(f"Best fitness: {best_fitness:.2f}")
        print(f"Best cluster: {best_cluster}")

        analyze_surrogate_results(experiment_id)

    elif OPTION == 2:
        # Load and test best model from most recent experiment
        experiment_id = get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            print(f"Loading surrogate experiment ID: {experiment_id}")
            test_best_design_surrogate(experiment_id)

    elif OPTION == 3:
        # Just get best model info from most recent experiment
        experiment_id = get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            print(f"Analyzing surrogate experiment ID: {experiment_id}")
            info = get_best_surrogate_model_info(experiment_id)
            if info:
                print(f"Best design: {info['design']}")
                print(f"Best fitness: {info['fitness']:.2f}")
                print(f"Best cluster: {info['cluster']}")
                print(f"Total evaluations: {info['total_evaluations']}")
                print(f"GP dataset size: {info['gp_data_size']}")
                print(f"Model path: {info['model_path']}")
                print(f"Stats path: {info['stats_path']}")

            analyze_surrogate_results(experiment_id)

    elif OPTION == 4:
        # Analyze surrogate results
        experiment_id = get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            analyze_surrogate_results(experiment_id)

    elif OPTION == 5:
        # Test best model from each generation
        experiment_id = get_next_experiment_id() - 1
        if experiment_id < 1:
            print("No experiments found. Run optimization first (OPTION = 1).")
        else:
            generation_files = [f for f in os.listdir(f"Experiments/{experiment_id}/evolution_data")
                                if f.startswith("generation_") and f.endswith(".pkl")]
            generation_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))

            for gen_file in generation_files:
                generation = int(gen_file.split('_')[1].split('.')[0])
                if generation > 0:  # Skip generation 0 (initialization)
                    print(f"\n{'=' * 50}")
                    print(f"Testing best model from generation {generation}")
                    print('=' * 50)
                    test_generation_best_surrogate(experiment_id, generation)

                    # Ask user if they want to continue
                    response = input("\nPress Enter to continue to next generation, or 'q' to quit: ")
                    if response.lower() == 'q':
                        break