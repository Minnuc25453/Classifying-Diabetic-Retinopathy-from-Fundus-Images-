import time

import numpy as np


# Function to perform the social tendency update (global influence)
def social_tendency(population, fitness):
    best_coyote_idx = np.argmin(fitness)
    best_coyote = population[best_coyote_idx]
    return best_coyote


# Function for social learning (local influence)
def social_learning(coyote, best_coyote, dim, sigma=0.1):
    return coyote + np.random.normal(0, sigma, dim) * (best_coyote - coyote)


# Function for movement
def movement(coyote, social_update, dim, lb, ub):
    new_coyote = coyote + social_update
    new_coyote = np.clip(new_coyote, lb, ub)  # Ensuring bounds are respected
    return new_coyote


# COA main function
def COA(population, fitness_function, lb, ub, max_iter):
    n_coyotes, dim = population.shape[0], population.shape[1]
    fitness = np.apply_along_axis(fitness_function, 1, population)

    best_fitness = np.min(fitness)
    best_position = population[np.argmin(fitness)]
    Convergence = np.zeros(max_iter)
    ct = time.time()
    for iteration in range(max_iter):
        for i in range(n_coyotes):
            # Social influence of the pack
            best_coyote = social_tendency(population, fitness)

            # Social learning and movement
            social_update = social_learning(population[i], best_coyote, dim)
            population[i] = movement(population[i], social_update, dim, lb[i], ub[i])

        fitness = np.apply_along_axis(fitness_function, 1, population)

        # Update global best
        current_best_fitness = np.min(fitness)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_position = population[np.argmin(fitness)]

        print(f"Iteration {iteration + 1}/{max_iter}, Best Fitness: {best_fitness}")
        Convergence[iteration] = best_fitness
    print("Best Solution:", best_position)
    print("Best Fitness:", best_fitness)
    ct = time.time() - ct
    return best_fitness, Convergence, best_position, ct




