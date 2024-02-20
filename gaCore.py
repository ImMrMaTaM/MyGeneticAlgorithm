import numpy as np
from ypstruct import structure
from Constraints import constraint_violation, bound_violation

############################################## FUNCTIONS ##############################################

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

# ROULETTE WHEEL SELECTION
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]