import numpy as np
from ypstruct import structure
from gaCore import constraints_violation, validity, worst_valid_cost_funct, fitness_funct, prob_Boltzmann, roulette_wheel_selection, crossover, mutate, apply_bound

def run(problem, params):

    # EXTRACT PROBLEM INFORMATION 
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax
    constraints = problem.constraints
    constraints_toll = problem.constraints_toll

    # EXTRACT PROBLEM PARAMETERS
    maxrep = params.maxrep
    stoprep = params.stoprep
    digits = params.digits
    maxit = params.maxit
    stopit = params.stopit
    tollfitness = params.tollfitness
    tollpos = params.tollpos
    npop = params.npop
    beta = params.beta
    pc = params.pc
    nc = int(np.round(pc*npop/2)*2) # number of children (always even)
    gamma = params.gamma
    mu = params.mu
    sigma = params.sigma

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
            pop[i].position = np.random.uniform(varmin, varmax, nvar) # fill population with npop random individuals
            pop[i].violation = constraints_violation(pop[i].position, constraints)
            pop[i].cost = costfunc(pop[i].position)
            pop[i].valid = validity(pop[i].violation, constraints_toll)
        
        for j in range(npop):
            worst_valid_cost = worst_valid_cost_funct(worst_valid_cost, pop[j].valid, pop[j].cost)
            pop[j].fitness = fitness_funct(pop[j].cost, pop[j].violation, pop[j].valid, worst_valid_cost)
    
        # INITIALIZE ITERATION QUITTER
        it_check = 0
    
    #################################### ITERATIONS ####################################
        for it in range(1, maxit):
            

            # CHILDREN GENERATION LOOP

            # 1 Calculate parenting probability
            probs = prob_Boltzmann(pop,beta) # probability of each individual to be a parent

            # 2 Initialize population of children
            popc = []

            for _ in range(nc//2):

                # 3 Parents selection (RANDOM)
                #q = np.random.permutation(npop)
                #p1 = pop[q[0]]
                #p2 = pop[q[1]]

                # 3 Parents selection(ROULETTE WHEEL)
                p1 = pop[roulette_wheel_selection(probs)]
                p2 = pop[roulette_wheel_selection(probs)]
            
                # 4 CROSSOVER
                c1, c2 = crossover(p1, p2, gamma)

                # 5 MUTATION
                c1 = mutate(c1, mu, sigma)
                c2 = mutate(c2, mu, sigma)

                ############################### MODIFY HERE #################################### 
                # 6 BOUNDARIES
                apply_bound(c1, varmin, varmax)
                apply_bound(c2, varmin, varmax)
                ############################### MODIFY HERE ####################################


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

    print(pop)
    ################################################################################

    # OUTPUT
    out = structure()
    out.n_rep = n_rep
    out.fitness = fitness
    out.IT = IT
    out.POS = POS[0:n_rep]
    out.FITNESS = FITNESS[0:n_rep]
    return out
