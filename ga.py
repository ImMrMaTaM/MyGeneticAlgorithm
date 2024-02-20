import numpy as np
from ypstruct import structure
from gaCore import prob_Boltzmann, crossover, mutate, apply_bound, roulette_wheel_selection

def run(problem, params):

    # EXTRACT PROBLEM INFORMATION 
    costfunc = problem.costfunc
    nvar = problem.nvar
    varmin = problem.varmin
    varmax = problem.varmax

    # EXTRACT PROBLEM PARAMETERS
    maxrep = params.maxrep
    stoprep = params.stoprep
    digits = params.digits
    maxit = params.maxit
    stopit = params.stopit
    tollcost = params.tollcost
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
    empty_individual.cost = None

    # OUTPUTS' TEMPLATES
    costs = np.zeros([maxit,maxrep])
 
    POS = np.zeros([maxrep,nvar]) # array with best position found at each repetition
    COST = np.zeros(maxrep) # array with best cost found at each repetition
    IT = np.zeros([maxit, maxrep]) # array with number of iterations performed at each repetition
    n_rep = 0 # number of performed repetitions


    ############################################## REPETITIONS ##############################################
    for rep in range(maxrep):
    
        # BEST INDIVIDUAL TEMPLATE FOUND AT CURRENT ITERATION
        bestsol = empty_individual.deepcopy() # best individual found at current iteration
        bestsol.cost = np.inf # set best solution's cost to be plus infinity

        # ARRAY WITH BEST INDIVIDUAL OF EACH ITERATION FOR CURRENT REPETITION
        bestcost = np.empty(maxit) # array with best cost found at each iteration
        bestpos = np.empty([maxit,nvar]) # array with best position found at each iteration

        # INITIALIZE RANDOM POPULATION
        pop = empty_individual.repeat(npop)
        for i in range(npop):
            pop[i].position = np.random.uniform(varmin, varmax, nvar) # fill population with npop random individuals
            pop[i].cost = costfunc(pop[i].position) # calculate the cost of all individuals in the population
            if pop[i].cost < bestsol.cost:
                bestsol = pop[i].deepcopy() # calculate the actual best solution of the initialized population
                bestcost[0] = pop[i].cost
                bestpos[0] = pop[i].position
                costs[0,rep] = bestcost[0]
    
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

                # 6 BOUNDARIES
                apply_bound(c1, varmin, varmax)
                apply_bound(c2, varmin, varmax)

                # 7.1 EVALUATE FIRST OFFSPRING
                c1.cost = costfunc(c1.position)
                if c1.cost < bestsol.cost:
                    bestsol = c1.deepcopy() # update best individual

                # 7.2 EVALUATE SECOND OFFSPRING
                c2.cost = costfunc(c2.position)
                if c2.cost < bestsol.cost:
                    bestsol = c2.deepcopy() # update best individual

                # 8 GENERATE POPULATION OF CHILDREN
                popc.append(c1)
                popc.append(c2)
        
            # MERGE
            pop += popc # merge

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

            # SORT
            pop = sorted(pop, key=lambda x: x.cost)

            # SELECT
            pop = pop[0:npop]

            bestcost[it] = bestsol.cost # store best cost of iteration in the array
            bestpos[it] = bestsol.position # store best position of iteration in the array
            costs[it,rep] = bestcost[it] 
            IT[it,rep] = it

            # PRINT ITERATION'S INFORMATIONS (# ITERATION AND BEST COST)
            print("Iteration {}:  Best Cost = {}  Best Position = {}".format(it, bestcost[it], bestpos[it]))

            # CHECK FOR OPTIMUM REPETITION 
            diffcost = np.abs(bestcost[it]-bestcost[it-1])
            diffpos = np.abs(np.sum((bestpos[it]-bestpos[it-1])))

            if diffcost < tollcost and diffpos < tollpos:
                it_check += 1
            else: 
                it_check = 0
    
            if it_check == stopit:
                break

        n_rep += 1
        POS[rep] = bestsol.position
        COST[rep] = costfunc(POS[rep])
        n = np.count_nonzero(np.all((np.around(POS[0:rep], decimals = digits) == np.around(bestsol.position, decimals = digits)), axis = 1))

        

        if n == stoprep:
            break



    ################################################################################

    # OUTPUT
    out = structure()
    out.n_rep = n_rep

    out.costs = costs
    out.IT = IT
    out.POS = POS[0:n_rep]
    out.COST = COST[0:n_rep]
    return out


