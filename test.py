import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from ypstruct import structure
from gaCore import constraints_violation, validity, worst_valid_cost_funct, fitness_funct, prob_Boltzmann, roulette_wheel_selection, crossover, mutation, apply_bound
################################################################################

# COST FUNCTION TO MINIMIZE
def cost_function(x): 
    return -(100 - (x[0]-5)**2 - (x[1]-5)**2 - (x[2]-5)**2)/100

# CONSTRAINTS (form x^2+y^2-a <= 0)
def constraint_functions(x):
    const = (x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2 - 0.0625
    return const

# PROBLEM DEFINITION
costfunc = cost_function # define the problem's cost function
constraints = constraint_functions # define the problem's nonlinear constraints
constraints_toll = 1e-10 # define the problem's constraints tollerance
nvar_cont = 3 # define number of continuous variables in search space
nvar_disc = 3 # define number of discrete variables in search space
nvar = nvar_cont + nvar_disc # define total number of variables
index_cont = [0,1,2] # define the indexes of continuous variables
index_disc = [3,4,5] # define the indexes of discrete variables
varmin_cont = [0, 0, 0] # lower bound of continuous variables
varmax_cont = [10, 10, 10]# upper bound of continuous variables
varmin_disc = [1, 1, 1] # lower bound of discrete variables
varmax_disc = [9, 9, 9] # upper bound of discrete variables
varmin = varmin_cont + varmin_disc # lower bound of all variables
varmax = varmax_cont + varmax_disc # upper bound of all variables

# GA PARAMETERS
maxrep = 1 # maximum number of repetitions
stoprep = 3 # number of same solutions to stop repeating 
digits = 6 # accuracy of digits for a position being the same
maxit = 5 # maximum number of iterations 
stopit = 100 # number of repetitions of same optimum point before breaking
tollfitness = 1e-6 # fitness difference tollerance for breaking iterations
tollpos = 1e-6 # position difference tollerance for breaking iterations
npop = 10 # size of initial population
pc = 3 # proportion of children to main population
nc = int(np.round(pc*npop/2)*2) # number of children (always even)
beta = 0.3 # Boltzman constant for parent selection probability
gamma = 0.8 # parameter for whole arithmetic recombination crossover
mu_cont = 0.4 # mutation threshold for continuous variables
mu_disc = 0.4 # mutation threshold for discrete variables
sigma = 0.3 # standard deviation of gene mutation 


# INDIVIDUAL'S TEMPLATE
empty_individual = structure()
empty_individual.position = None
empty_individual.violation = None
empty_individual.cost = None
empty_individual.valid = None
empty_individual.fitness = None

# OUTPUTS' TEMPLATES
fitness = np.zeros([maxit,maxrep])
POS = np.zeros([maxrep,nvar]) # array with best position found at each repetition
FITNESS = np.zeros(maxrep) # array with best fitness found at each repetition
IT = np.zeros([maxit, maxrep]) # array with number of iterations performed at each repetition
n_rep = 0 # number of performed repetitions


############################################## REPETITIONS ##############################################
for rep in range(maxrep):
    
    # BEST INDIVIDUAL TEMPLATE FOUND AT CURRENT ITERATION
    bestsol = empty_individual.deepcopy() # best individual found at current iteration
    bestpos = np.empty([maxit,nvar]) # array with best position found at each iteration

    # INITIALIZE RANDOM POPULATION
    pop = empty_individual.repeat(npop)
    worst_valid_cost = 0

    for i in range(npop):
        pop[i].position = np.random.uniform(varmin_cont, varmax_cont, nvar_cont) # fill population with continuous variables
        pop[i].position = np.append(pop[i].position, np.random.randint(varmin_disc, varmax_disc, nvar_disc)) # fill population with discrete variables
        pop[i].violation = constraints_violation(pop[i].position, constraints)
        pop[i].cost = costfunc(pop[i].position)
        pop[i].valid = validity(pop[i].violation, constraints_toll)
        
    for j in range(npop):
        worst_valid_cost = worst_valid_cost_funct(worst_valid_cost, pop[j].valid, pop[j].cost)
        pop[j].fitness = fitness_funct(pop[j].cost, pop[j].violation, pop[j].valid, worst_valid_cost)
    
    # INITIALIZE ITERATION QUITTER
    it_check = 0


    for it in range(1, maxit):
            

        # CHILDREN GENERATION LOOP

        # 1 Calculate parenting probability
        probs = prob_Boltzmann(pop,beta) # probability of each individual to be a parent

        # 2 Initialize population of children
        popc = []

        for _ in range(nc//2):

            # 3 Parents selection(ROULETTE WHEEL)
            p1 = pop[roulette_wheel_selection(probs)]
            p2 = pop[roulette_wheel_selection(probs)]

            print(f"\n\n\n{p1.position}")
            print(p2.position)

            # 4 CROSSOVER
            c1, c2 = crossover(p1,p2,index_cont,index_disc,gamma)

            print(f"\n{c1.position}")
            print(c2.position)
            
            # 5 MUTATION
            c1 = mutation(c1, mu_cont, sigma, mu_disc, varmin_disc, varmax_disc, index_cont, index_disc)
            c2 = mutation(c2, mu_cont, sigma, mu_disc, varmin_disc, varmax_disc, index_cont, index_disc)

            print(f"\n{c1.position}")
            print(c2.position)

            

            ############################### MODIFY HERE #################################### 
            # 6 BOUNDARIES
            apply_bound(c1, varmin_cont, varmax_cont, index_cont, index_disc)
            apply_bound(c2, varmin_cont, varmax_cont, index_cont, index_disc)
            ############################### MODIFY HERE ####################################

            print(f"\n{c1.position}")
            print(c2.position)

            # 7 EVALUATE OFFSPRING (violation, cost, validity, fitness)
            c1.violation = constraints_violation(c1.position, constraints)
            c1.cost = costfunc(c1.position)
            c1.valid = validity(c1.violation, constraints_toll)
            c2.violation = constraints_violation(c2.position, constraints)
            c2.cost = costfunc(c2.position)
            c2.valid = validity(c2.violation, constraints_toll)

            # 8 GENERATE POPULATION OF CHILDREN
            popc.append(c1)
            popc.append(c2)
        
        # MERGE
        pop += popc # merge

        for k in range (len(pop)):
            worst_valid_cost = worst_valid_cost_funct(worst_valid_cost, pop[k].valid, pop[k].cost)
            pop[k].fitness = fitness_funct(pop[k].cost, pop[k].violation, pop[k].valid, worst_valid_cost)

        # SORT
        pop = sorted(pop, key=lambda x: x.fitness)

        # SELECT
        pop = pop[0:npop]

        # STORE BEST SOLUTION
        bestsol = pop[0].deepcopy() # update best individual
        bestpos[it] = bestsol.position
        fitness[it,rep] = bestsol.fitness 
        IT[it,rep] = it

        # PRINT ITERATION'S INFORMATIONS (# ITERATION AND BEST COST)
        print(worst_valid_cost)
        print("Iteration {}:  Best Fitness value = {}  Best Position = {}  Valid: {}".format(it, bestsol.fitness, bestsol.position, bestsol.valid))

        # CHECK FOR OPTIMUM REPETITION 
            
        diff_fitness = np.abs(fitness[it,rep] - fitness[it-1,rep])
        diff_pos = np.abs(np.sum((bestpos[it]-bestpos[it-1])))

        if diff_fitness < tollfitness and diff_pos < tollpos:
            it_check += 1
        else: 
            it_check = 0
    
        if it_check == stopit:
            break

    n_rep += 1
    POS[rep] = bestsol.position
    FITNESS[rep] = bestsol.fitness
    n = np.count_nonzero(np.all((np.around(POS[0:rep], decimals = digits) == np.around(bestsol.position, decimals = digits)), axis = 1))
    if n == stoprep:
        break
