import random
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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

    # # Print the generated population
    # for idx, (design, fitness, success_rate) in population.items():
    #     print(f"Individual {idx+1} design: {design}, Fitness: {fitness}, Success Rate: {success_rate}")

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

def chebyshev(u, v):
    return np.max(np.abs(u - v))


def get_clusters(population, threshold=20, min_size=10, seed=None):
    """
    Greedy Chebyshev‐ball clustering with minimum cluster size:
      - Randomly pick a center among unclustered points
      - Group *all* points whose max‐coordinate distance ≤ threshold
      - If cluster size < min_size, discard it (but still remove points)
      - Remove those points and repeat
    """
    if seed is not None:
        np.random.seed(seed)

    # Extract N×D array from the population dict
    designs = np.array([list(ind[0]) for ind in population.values()])  # shape (N,5)
    points = designs.copy()

    clusters = []

    while points.size > 0:
        # Random center index
        idx = np.random.randint(len(points))
        center = points[idx]

        # Chebyshev distances to that center
        dists = np.max(np.abs(points - center), axis=1)

        # Pull in *all* points within threshold
        within_idx = np.where(dists <= threshold)[0]
        cluster = points[within_idx]

        if len(cluster) >= min_size:
            # Cluster is large enough, keep it
            clusters.append(cluster)
        # If cluster is too small, we discard it but still remove the points

        # Remove clustered points (whether we kept the cluster or not)
        points = np.delete(points, within_idx, axis=0)

    return clusters

def get_clusters_farthest_first(population, threshold=20, seed=None):
    """
    Optimized greedy Chebyshev‐ball clustering with:
      - 1st center chosen at random
      - subsequent centers chosen as the remaining point farthest (in L_inf) from all existing centers
      - after picking each center, absorb *all* points within `threshold` Chebyshev distance into that cluster
    """
    designs = np.array([list(ind[0]) for ind in population.values()])  # (N, 5)
    if seed is not None:
        np.random.seed(seed)

    n_points = len(designs)

    # Use boolean mask instead of index arrays for better performance
    unclustered = np.ones(n_points, dtype=bool)
    clusters = []
    centers = []

    # Pick first center at random
    remaining_indices = np.where(unclustered)[0]
    first = np.random.choice(remaining_indices)
    centers.append(designs[first])

    # Form its cluster and remove
    distances = np.max(np.abs(designs - centers[-1]), axis=1)
    cluster_mask = distances <= threshold
    clusters.append(designs[cluster_mask])
    unclustered &= ~cluster_mask  # Remove clustered points

    # Maintain minimum distances to all centers for each point
    min_distances = np.full(n_points, np.inf)
    min_distances[~cluster_mask] = distances[~cluster_mask]

    # Repeat: pick farthest‐first center among remaining
    while np.any(unclustered):
        # Find the unclustered point with maximum distance to nearest center
        farthest_idx = np.argmax(np.where(unclustered, min_distances, -np.inf))
        centers.append(designs[farthest_idx])

        # Compute distances to new center for all points
        new_distances = np.max(np.abs(designs - centers[-1]), axis=1)

        # Update minimum distances (only if new center is closer)
        min_distances = np.minimum(min_distances, new_distances)

        # Form cluster around new center
        cluster_mask = new_distances <= threshold
        clusters.append(designs[cluster_mask])

        # Remove clustered points
        unclustered &= ~cluster_mask

    return clusters

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

def main(experiment_id):

    base_width = 200
    left_finger_1 = 120
    right_finger_1 = 120
    left_finger_2 = 120
    right_finger_2 = 120

    # Parameters for population generation
    range = 10
    population_size = 100000

    # Parameters for clustering
    perturbation = 10
    cluster_threshold = perturbation // 2

    # Create the population based on the bounds defined by the max_difference
    bounds = get_bounds(base_width, left_finger_1, right_finger_1, left_finger_2, right_finger_2, range)
    population = generate_population(population_size, bounds)

    # Get the clusters
    clusters = get_clusters(population, threshold=cluster_threshold)
    # clusters = get_clusters_farthest_first(population, threshold=cluster_threshold)

    print(len(clusters), "clusters found")

    for idx, cluster in enumerate(clusters):
        print(f"Cluster {idx + 1} ({len(cluster)} items):")
        for d in cluster:
            print("  ", d)
        print()

def main_plot(experiment_id, ranges, max_perturbation=10):

    base_width = 200
    left_finger_1 = 120
    right_finger_1 = 120
    left_finger_2 = 120
    right_finger_2 = 120

    cluster_size = []

    # Parameters for population generation
    population_size = 100000

    for range in ranges:

        print(range)

        # Parameters for clustering
        perturbation = max_perturbation
        cluster_threshold = perturbation // 2

        # Create the population based on the bounds defined by the max_difference
        bounds = get_bounds(base_width, left_finger_1, right_finger_1, left_finger_2, right_finger_2, range)
        population = generate_population(population_size, bounds)

        # Get the clusters
        clusters = get_clusters(population, threshold=cluster_threshold)
        # clusters = get_clusters_farthest_first(population, threshold=cluster_threshold)

        cluster_size.append(len(clusters))

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(ranges, cluster_size, marker='o', linestyle='-', color='b')
    plt.title(f"Number of Clusters (> size 10) within perturbation of {max_perturbation} vs Population Bounds Range with population size {population_size}")
    plt.xlabel("Population Bounds Range")
    plt.ylabel("Number of Clusters")
    plt.xticks(ranges)
    plt.grid()
    os.makedirs(f'Experiments/{experiment_id}', exist_ok=True)
    plt.savefig(f"Experiments/{experiment_id}/clusters_vs_range.png")
    plt.show()


if __name__ == "__main__":

    experiment_id = get_next_experiment_id()
    # main(experiment_id)
    main_plot(experiment_id, ranges=list(range(5, 201, 5)), max_perturbation=20)