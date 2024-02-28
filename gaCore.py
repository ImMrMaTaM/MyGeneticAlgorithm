import numpy as np
from ypstruct import structure


############################################## FUNCTIONS ##############################################

# CALCULATE VIOLATIONS
def constraints_violation(x, constraint_functions):
    vals = constraint_functions(x)
    if vals == None:
        return 0
    elif type(vals) == np.float64:
        if vals >= 0:
            return vals
        else:
            return 0
    else:
        res = 0
        for val in vals:
            if val >= 0:
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
    if valid == False:
        return worst_valid_cost
    elif valid == True:
        if cost > worst_valid_cost:
            worst_valid_cost = cost
    return worst_valid_cost


# CALCULATE FITNESS
def fitness_funct(cost, violation, valid, worst_valid_cost):
    if valid == False:
        fitness = np.maximum(worst_valid_cost + violation, cost)
    elif valid == True:
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

# WHOLE ARITHMETIC RECOMBINATION CROSSOVER (for continuous variables)
def WAR_crossover(p1, p2, gamma):
    alpha = np.random.uniform(-gamma, 1+gamma, np.shape(p1)) 
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1
    return c1, c2

# UNIFORM CROSSOVER (for discrete variables)
def uniform_crossover(p1, p2):
    crossover_vector = np.random.randint(2, size = np.size(p1))
    c1 = crossover_vector*p1 + (1-crossover_vector)*p2
    c2 = crossover_vector*p2 + (1-crossover_vector)*p1
    return c1, c2

# CROSSOVER 
def crossover(p1,p2,index_cont,index_disc,gamma):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    c1_cont, c2_cont = WAR_crossover(np.take(p1.position,index_cont), np.take(p2.position,index_cont), gamma)
    c1_disc, c2_disc = uniform_crossover(np.take(p1.position,index_disc), np.take(p2.position,index_disc))
    c1.position = np.concatenate((c1_cont,c1_disc))
    c2.position = np.concatenate((c2_cont,c2_disc))
    return c1, c2

# GAUSSIAN MUTATION (for continuous variables)
def gaussian_mutation(x, mu_cont, sigma):
    y = x
    flag = np.random.rand(np.size(x)) <= mu_cont
    ind = np.nonzero(flag)
    np.put(y,ind,np.take(y,ind)+sigma*np.random.randn(np.size(ind)))
    return y

# RANDOM MUTATION (for discrete variables)
def random_mutation(x, mu_disc, varmin_disc, varmax_disc):
    y = x
    flag = np.random.rand(np.size(y)) <= mu_disc
    ind = np.nonzero(flag)
    np.put(y,ind,np.take(np.random.randint(varmin_disc,varmax_disc), ind))
    return y
    
# MUTATION
def mutation (x, mu_cont, sigma, mu_disc, varmin_disc, varmax_disc, index_cont, index_disc):
    y = x.deepcopy()
    y_cont = gaussian_mutation(np.take(x.position, index_cont), mu_cont, sigma)
    y_disc = random_mutation(np.take(x.position, index_disc), mu_disc, varmin_disc, varmax_disc)
    y.position = np.concatenate((y_cont, y_disc))
    return y

# BOUNDARIES (for continuous variables)
def apply_bound(x, varmin_cont, varmax_cont, index_cont, index_disc):
    y = np.take(x.position, index_cont)
    y_min = np.maximum(y, varmin_cont)
    y_max = np.minimum(y_min, varmax_cont)
    y_new = np.concatenate((y_max,np.take(x.position, index_disc)))
    x.position = y_new
    return x