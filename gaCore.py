import numpy as np
from ypstruct import structure
from Constraints import constraint_violation, bound_violation

############################################## FUNCTIONS ##############################################

# INITIALIZE RANDOM POPULATION
def initialize_population(empty_individual, npop, varmin, varmax, nvar, costfunc, constraints_toll, constraint_functions):

    pop = empty_individual.repeat(npop)
    worst_valid_cost = 0

    for i in range(npop):
        pop[i].position = np.random.uniform(varmin, varmax, nvar) # fill population with npop random individuals
        pop[i].violation = constraints_violation(pop[i].position, constraint_functions)
        pop[i].cost = costfunc(pop[i].position)
        pop[i].valid = validity(pop[i].violation, constraints_toll)
        worst_valid_cost = worst_valid_cost_funct(worst_valid_cost, pop[i].valid, pop[i].cost)
        
    for j in range(npop):
        pop[j].fitness = fitness_funct(pop[j].cost, pop[j].violation, pop[j].valid, worst_valid_cost)

    return pop


# CALCULATE VIOLATIONS
def constraints_violation(x, constraint_functions):
    vals = constraint_functions(x)
    if len(vals) == 1:
        return vals[0]
    res = 0
    for val in vals:
        if val > 0:
            res += val
    return res


# CALCULATE VALIDITY
def validity(violation, constraints_toll):
    if violation >= constraints_toll:
        valid = False
    else:
        valid = True
    return valid

# CALCULATE WORST VALID COST
def worst_valid_cost_funct(worst_valid_cost, valid, cost):
    if valid == True:
        if cost > worst_valid_cost:
            worst_valid_cost = cost
    return worst_valid_cost


# CALCULATE FITNESS
def fitness_funct(cost, violation, valid, worst_valid_cost):
    if valid == False:
        fitness = worst_valid_cost + violation
    else:
        fitness = cost
    return fitness





""""
# CALCULATE FITNESS
def fitness(violation, cost, violation_tol):
    if violation <= violation_tol:
        fitness = cost

    else:





    






    def population_fitness(violation, obj_val, tol=1e-4):
    n = len(violation)
    is_feasible = np.zeros(n, dtype=boolean)
    fitness = np.empty(n)
    worst_obj = 0.
    for i in range(n):
        violation_i = violation[i]
        if violation_i <= tol:
            is_feasible[i] = True
            obj_val_i = obj_val[i]
            fitness[i] = obj_val_i
            if obj_val_i >= worst_obj:
                worst_obj = obj_val_i
        else:
            fitness[i] = violation_i
    fitness[~is_feasible] += worst_obj
    return fitness

    """

# PARENTING PROBABILITY
def prob_Boltzmann(pop,beta):
    fitness = np.array([x.fitness for x in pop])
    avg_fitness = np.mean(fitness)
    if avg_fitness != 0:
        fitness = fitness/avg_fitness
    probs = np.exp(-beta*fitness) # probability of each individual to be a parent
    return probs

# ROULETTE WHEEL SELECTION
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]

# CROSSOVER
def crossover(p1, p2, gamma=0.1):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    alpha = np.random.uniform(-gamma, 1+gamma, *c1.position.shape)
    c1.position = alpha*p1.position + (1-alpha)*p2.position
    c2.position = alpha*p2.position + (1-alpha)*p1.position
    return c1, c2

# MUTATION
def mutate(x, mu, sigma):
    y = x.deepcopy()
    flag = np.random.rand(*x.position.shape) <= mu
    ind = np.argwhere(flag)
    y.position[ind] += sigma*np.random.randn(*ind.shape)
    return y

# BOUNDARIES
def apply_bound(x, varmin, varmax):
    x.position = np.maximum(x.position, varmin)
    x.position = np.minimum(x.position, varmax)
    return x
