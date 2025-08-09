import time
import numpy as np


# Define the Dollmaker Optimization Algorithm with bounds
def DOA(population, obj_func,  lb, ub,  num_iterations):
    population_size, num_vars = population.shape[0], population.shape[1]

    # Evaluate the initial population
    fitness = np.apply_along_axis(obj_func, 1, population)

    # Track the best solution
    best_fitness = np.min(fitness)
    best_solution = population[np.argmin(fitness)]

    Convergence = np.zeros(num_iterations)
    ct = time.time()

    # Main loop for the iterations
    for t in range(num_iterations):
        for i in range(population_size):
            # Phase 1: Pattern selection and sewing (exploration phase)
            r = np.random.rand(num_vars)  # Random pattern
            new_position_phase1 = population[i] + r * (best_solution - population[i])
            new_position_phase1 = np.clip(new_position_phase1, lb, ub)  # Ensure within bounds

            # Update ith member
            if obj_func(new_position_phase1)[i] < fitness[i]:
                population[i] = new_position_phase1[i]
                fitness[i] = obj_func(new_position_phase1)[i]

            # Phase 2: Beautifying the details of the doll (exploitation phase)
            beautify_factor = 0.5  # Example beautify factor
            new_position_phase2 = population[i] + beautify_factor * (population[i] - new_position_phase1)
            new_position_phase2 = np.clip(new_position_phase2, lb, ub)  # Ensure within bounds

            # Update ith member
            if obj_func(new_position_phase2)[i] < fitness[i]:
                population[i] = new_position_phase2[i]
                fitness[i] = obj_func(new_position_phase2)[i]

        # Update the best solution found so far
        if np.min(fitness) < best_fitness:
            best_fitness = np.min(fitness)
            best_solution = population[np.argmin(fitness)]

        Convergence[t] = best_fitness

    print("Best solution:", best_solution)
    print("Best fitness:", best_fitness)
    ct = time.time() - ct
    return best_fitness, Convergence, best_solution,  ct



