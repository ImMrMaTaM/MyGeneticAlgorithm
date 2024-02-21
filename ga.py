import numpy as np
from ypstruct import structure
from gaCore import initialize_population, constraints_violation, validity, worst_valid_cost_funct, fitness_funct, prob_Boltzmann, roulette_wheel_selection, crossover, mutate, apply_bound

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
    empty_individual.valid = True
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
        pop = initialize_population(empty_individual, npop, varmin, varmax, nvar, costfunc, constraints_toll, constraints)
    
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
                worst_valid_child_cost = 0
                c1.violation = constraints_violation(c1.position, constraints)
                c1.cost = costfunc(c1.position)
                c1.valid = validity(c1.violation, constraints_toll)
                worst_valid_child_cost = worst_valid_cost_funct(worst_valid_child_cost, c1.valid, c1.cost)
                c2.violation = constraints_violation(c2.position, constraints)
                c2.cost = costfunc(c2.position)
                c2.valid = validity(c2.violation, constraints_toll)
                worst_valid_child_cost = worst_valid_cost_funct(worst_valid_child_cost, c2.valid, c2.cost)

                # 8 GENERATE POPULATION OF CHILDREN
                popc.append(c1)
                popc.append(c2)
            
            for k in range (len(popc)):
                popc[k].fitness = fitness_funct(popc[k].cost, popc[k].violation, popc[k].valid, worst_valid_child_cost)
        
            # MERGE
            pop += popc # merge

            """"
            # ELIMINATE DUPLICATES
            positions = np.array([x.position for x in pop])
            positions = np.unique(positions, axis = 0)

            leng = np.shape(positions)[0]
            add = npop+nc-leng

            while add > 0:
                positions = np.concatenate((positions, np.random.uniform(varmin, varmax, (add,nvar))), axis = 0)
                positions = np.unique(positions, axis = 0)
                leng = np.shape(positions)[0]
                add = npop+nc-leng

            new_pop = empty_individual.repeat(leng)
            for a in range(leng):
                new_pop[a].position = positions[a]
                new_pop[a].cost = costfunc(new_pop[a].position)
                pop = new_pop
            """

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
            print("Iteration {}:  Best Fitness value = {}  Best Position = {}".format(it, bestsol.fitness, bestsol.position))

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

    ################################################################################

    # OUTPUT
    out = structure()
    out.n_rep = n_rep
    out.fitness = fitness
    out.IT = IT
    out.POS = POS[0:n_rep]
    out.FITNESS = FITNESS[0:n_rep]
    return out
