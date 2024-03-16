"""
This code contains all the functions used in the genetic algorithm (GA)
"""



# Import libraries
import numpy as np
from ypstruct import structure


############################################## INDIVIDUALS FUNCTIONS ##############################################
"""
These functions are used to calculate parameters related to the individuals in the population.
"""

# CALCULATE VIOLATIONS
"""
This function calculates how much a solution (x) violates the constraints of the problem.
x is the position of the individual in the search space.
constraint_functions is a function that returns the values of the constraints for the given position.
If the constraints are not violated, the function returns 0.
If the constraints are violated, the function returns the sum of the violations.
"""
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
"""
This function checks if the solution (x) is valid, i.e. if it violates the constraints of the problem.
It returns a boolean value, either True or False
"""
def validity(violation, constraints_toll):
    if violation >= constraints_toll:
        valid = False
    else:
        valid = True
    return valid

# CALCULATE WORST VALID COST
"""
This function calculates the worst valid cost found so far.
If the solution is not valid, it returns the worst valid cost found so far.
If the solution is valid, it returns the maximum between the worst valid cost found so far and the cost of the solution.
worst_cost_value is then used to calculate the fitness of the individuals.
"""
def worst_valid_cost_funct(worst_valid_cost, valid, cost):
    if valid == False:
        return worst_valid_cost
    elif valid == True:
        if cost > worst_valid_cost:
            worst_valid_cost = cost
    return worst_valid_cost


# CALCULATE FITNESS
"""
This function calculates the fitness of the individuals.
The fitness is the driving value of the genetic algorithm, it is the value that the algorithm tries to minimize.
If the solution is not valid, the fitness is the maximum between the worst valid cost found so far + the violation 
of the constraints and the cost.
This guarantees that all the valid solutions have a fitness value lower than the worst valid cost found so far.
If the solution is valid, the fitness is the cost of the solution.
"""
def fitness_funct(cost, violation, valid, worst_valid_cost):
    if valid == False:
        fitness = np.maximum(worst_valid_cost + violation, cost)
    elif valid == True:
        fitness = cost
    return fitness


############################################## GA FUNCTIONS ##############################################
"""
These functions are the ones used in the GA
For full explanation of the GA, please refer to the documentation (README)
"""
######################## PARENTS SELECTION #########################

# PARENTING PROBABILITY
"""
This function calculates the probability of each individual to be a parent.
By taking the entire population, it calculates the fitness of each individual and then the average fitness.
The probability of each individual to be a parent is then calculated as the exponential of the negative of 
the fitness of the individual divided by the average fitness.
"""

def prob_Boltzmann(pop,beta):
    fitness = np.array([x.fitness for x in pop])
    avg_fitness = np.mean(fitness)
    if avg_fitness != 0:
        fitness = fitness/avg_fitness
    probs = np.exp(-beta*fitness) # probability of each individual to be a parent
    return probs


# ROULETTE WHEEL SELECTION
"""
This function is used to simulate a roulette wheel to select which individuals will be selected as parents, 
using the probabilities computed above
"""
def roulette_wheel_selection(p):
    c = np.cumsum(p)
    r = sum(p)*np.random.rand()
    ind = np.argwhere(r <= c)
    return ind[0][0]


######################## CROSSOVER #########################

# WHOLE ARITHMETIC RECOMBINATION CROSSOVER (for continuous variables)
"""
This function is the crossover function for the continuous variables.
"""
def WAR_crossover(p1, p2, gamma):
    alpha = np.random.uniform(-gamma, 1+gamma, np.shape(p1)) 
    c1 = alpha*p1 + (1-alpha)*p2
    c2 = alpha*p2 + (1-alpha)*p1
    return c1, c2


# UNIFORM CROSSOVER (for discrete variables)
"""
This function is the crossover function for the discrete variables.
"""
def uniform_crossover(p1, p2):
    crossover_vector = np.random.randint(2, size = np.size(p1))
    c1 = crossover_vector*p1 + (1-crossover_vector)*p2
    c2 = crossover_vector*p2 + (1-crossover_vector)*p1
    return c1, c2


# CROSSOVER 
"""
This function is the crossover function for the entire population.
Combines the continuous and discrete variables.
"""
def crossover(p1,p2,index_cont,index_disc,gamma):
    c1 = p1.deepcopy()
    c2 = p1.deepcopy()
    c1_cont, c2_cont = WAR_crossover(np.take(p1.position,index_cont), np.take(p2.position,index_cont), gamma)
    c1_disc, c2_disc = uniform_crossover(np.take(p1.position,index_disc), np.take(p2.position,index_disc))
    c1.position = np.concatenate((c1_cont,c1_disc))
    c2.position = np.concatenate((c2_cont,c2_disc))
    return c1, c2


######################## MUTATION #########################

# GAUSSIAN MUTATION (for continuous variables)
"""
This function is the mutation function for the continuous variables when individual is valid.
Individual genes mutate with a probability mu, and the mutation is a gaussian noise with standard deviation sigma.
"""
def gaussian_mutation(x, mu, sigma):
    y = x
    flag = np.random.rand(np.size(x)) <= mu
    ind = np.nonzero(flag)
    np.put(y,ind,np.take(y,ind)+sigma*np.random.randn(np.size(ind)))
    return y


# RANDOM MUTATION (for continuous variables)
"""
This function is the mutation function for the continuous variables when individual is invalid.
Individual genes mutate with a probability mu, and the mutation is a random number between varmin and varmax.
"""
def random_mutation_cont(x, mu, varmin, varmax):
    y = x
    flag = np.random.rand(np.size(y)) <= mu
    ind = np.nonzero(flag)
    np.put(y,ind,np.take(np.random.uniform(varmin, varmax), ind))
    return y


# RANDOM MUTATION (for discrete variables)
"""
This function is the mutation function for the discrete variables.
Individual genes mutate with a probability mu, and the mutation is a random integer between varmin and varmax.
"""
def random_mutation_disc(x, mu, varmin, varmax):
    y = x
    flag = np.random.rand(np.size(y)) <= mu
    ind = np.nonzero(flag)
    np.put(y,ind,np.take(np.random.randint(varmin,varmax), ind))
    return y


# MUTATION
def mutation (x, mu_cont, sigma, mu_disc, varmin_cont, varmax_cont, varmin_disc, varmax_disc, index_cont, index_disc):
    y = x.deepcopy()
    if x.valid == False:
        y_cont = random_mutation_cont(np.take(x.position, index_cont), mu_cont, varmin_cont, varmax_cont)
        y_disc = random_mutation_disc(np.take(x.position, index_disc), mu_disc, varmin_disc, varmax_disc)
        y.position = np.concatenate((y_cont, y_disc))

    else:
        y_cont = gaussian_mutation(np.take(x.position, index_cont), mu_cont, sigma)
        y_disc = random_mutation_disc(np.take(x.position, index_disc), mu_disc, varmin_disc, varmax_disc)
        y.position = np.concatenate((y_cont, y_disc))

    return y


# ADAPTIVE MUTATION
def adaptive_mutation(x, mu_cont, mu_cont0, sigma, sigma0, mu_disc, mu_disc0, it_check, adaptmut_it):

    if x.valid == False: # if no valid individual is found, max mutation rate
        mu_cont_new = 1
        mu_disc_new = 1
        sigma_new = sigma0
  
    else: # if valid individual is found, adapt mutation rate and mutation range based on the current best solution
        
        if mu_cont == 1 and mu_disc == 1: # if values are maxed, bring them back to original values
            mu_cont_new = mu_cont0
            mu_disc_new = mu_disc0
            sigma_new = sigma0

        if it_check >= adaptmut_it: # if same best solution for adaptmut_it iterations, reduce mutation range and increase mutation rate
            sigma_new = sigma*0.95
            mu_cont_new = mu_cont*1.05
            mu_disc_new = mu_disc*1.05
            if mu_cont_new >= 0.8:
                mu_cont_new = 0.8
            if mu_disc_new >= 0.8:
                mu_disc_new = 0.8
            if sigma_new <= 0.3:
                sigma_new = 0.3

        else: # if best solution is changing, mutation range and mutation rate don't change
            mu_cont_new = mu_cont0
            sigma_new = sigma0
            mu_disc_new = mu_disc0

    return mu_cont_new, sigma_new, mu_disc_new


######################## BOUNDARIES ENFORCEMENT #########################

# BOUNDARIES (for continuous variables)
def apply_bound(x, varmin_cont, varmax_cont, index_cont, index_disc):
    y = np.take(x.position, index_cont) # take continuous portion of individual
    y_min = np.maximum(y, varmin_cont) # enforce lower bound
    y_max = np.minimum(y_min, varmax_cont) # enforce upper bound
    y_new = np.concatenate((y_max,np.take(x.position, index_disc))) # put back the discrete portion of individual
    x.position = y_new
    return x