import numpy as np
from ypstruct import structure

##########################################################################################
def cost(x): 
    return x[0]**2+x[1]**2

# PARENTING PROBABILITY
def prob_Boltzmann(pop,beta):
    costs = np.array([x.cost for x in pop])
    avg_cost = np.mean(costs)
    if avg_cost != 0:
        costs = costs/avg_cost
    probs = np.exp(-beta*costs) # probability of each individual to be a parent
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

##########################################################################################
costfunc = cost
nvar = 2
varmin = [-5,-5]
varmax = [5,5]
maxit = 100
npop = 10
beta = 2
pc = 2
nc = int(np.ceil(pc*npop/2)*2) # number of children (always even)
gamma = 0.1
mu = 0.1
sigma = 0.1
stop_it = 50
toll_cost = 1e-10
toll_pos = 1e-10

diffcost = []

##########################################################################################

# CREATE INDIVIDUAL TEMPLATE
empty_individual = structure()
empty_individual.position = None
empty_individual.cost = None

# BEST SOLUTION/COST
bestsol = empty_individual.deepcopy() # best individual found at current iteration
bestsol.cost = np.inf # set to infinity
bestcost = np.empty(maxit) # array with best cost found at each iteration
bestpos = np.empty([maxit,nvar]) # array with best position found at each iteration

# INITIALIZE POPULATION
pop = empty_individual.repeat(npop)
for i in range(npop):
    pop[i].position = np.random.uniform(varmin, varmax, nvar)
    pop[i].cost = costfunc(pop[i].position)
    if pop[i].cost < bestsol.cost:
        bestsol = pop[i].deepcopy()
        bestcost[0] = pop[i].cost
        bestpos[0] = pop[i].position

stop_it = 0

for it in range(1,maxit):


    # 1 Calculate parenting probability
    probs = prob_Boltzmann(pop,beta) # probability of each individual to be a parent
    # 2 Initialize population of children
    popc = []
    for _ in range(nc//2):
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




    # SORT
    pop = sorted(new_pop, key=lambda x: x.cost)

     # SELECT
    pop = pop[0:npop]


    bestcost[it] = bestsol.cost # store best cost of iteration in the array
    bestpos[it] = bestsol.position # store best position of iteration in the array

    # PRINT ITERATION'S INFORMATIONS (# ITERATION AND BEST COST)
    print("Iteration {}:  Best Cost = {}  Best Position = {}".format(it-1, bestcost[it-1], bestpos[it-1]))

    # CHECK FOR CONSTANT OPTIMUM AFTER STOP_IT ITERATIONS
    
    diffcost = np.abs(bestcost[it]-bestcost[it-1])
    diffpos = np.abs(np.sum((bestpos[it]-bestpos[it-1])))

    print("{}".format(diffpos))



    if diffcost < toll_cost and diffpos < toll_pos:
        stop_it += 1
    else: 
        stop_it = 0
    
    if stop_it == 10:
        break
    











    ########################################################################