import time

import numpy as np


# Waterwheel Plant Algorithm (WPA)
def WPA(buckets, obj_func, lb, ub, max_iter):
    flow_rate = 0.2
    evaporation_rate = 0.01
    num_buckets, dim = buckets.shape[0], buckets.shape[1]

    # Initialize the buckets (solutions) randomly within bounds
    buckets = np.random.rand(num_buckets, dim)
    for i in range(dim):
        buckets[:, i] = lb[:, i] + (ub[:, i] - lb[:, i]) * buckets[:, i]

    # Evaluate initial buckets
    fitness = np.array([obj_func(bucket) for bucket in buckets])

    # Get the best solution
    best_solution = buckets[np.argmin(fitness)]
    best_fitness = np.min(fitness)
    Convergence = np.zeros(max_iter)
    ct = time.time()
    # Main loop (rotation of the waterwheel)
    for iteration in range(max_iter):
        # Simulate water flowing between buckets (solutions)
        for i in range(num_buckets):
            # Find a neighboring bucket (circular rotation on the wheel)
            next_bucket_idx = (i + 1) % num_buckets
            # Flow of water (solution adjustment) between neighboring buckets
            flow = flow_rate * (buckets[next_bucket_idx] - buckets[i])
            buckets[i] += flow

            # Ensure bounds are respected
            buckets[i] = np.clip(buckets[i], lb[i], ub[i])

        # Evaporation process (regenerate some buckets)
        for i in range(num_buckets):
            if np.random.rand() < evaporation_rate:
                r = np.random.rand(dim)
                buckets[i] = r * (ub[i] - lb[i]) + lb[i]

        # Evaluate the buckets again
        fitness = np.array([obj_func(bucket) for bucket in buckets])

        # Update the best solution if found
        current_best_idx = np.argmin(fitness)
        current_best_fitness = fitness[current_best_idx]
        if current_best_fitness < best_fitness:
            best_solution = buckets[current_best_idx]
            best_fitness = current_best_fitness
        Convergence[iteration] = best_fitness

        print(f"Iteration {iteration + 1}, Best fitness: {best_fitness}")
    ct = time.time() - ct
    return best_fitness, best_fitness, best_solution, ct
