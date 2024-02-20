import numpy as np
from ypstruct import structure

############################################## FUNCTIONS ##############################################

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