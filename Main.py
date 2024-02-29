import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from ypstruct import structure
import ga
import Test_functions_constrained as tfc
################################################################################

# COST FUNCTION TO MINIMIZE
def cost_function(x): 
    return -(100 - (x[0]-5)**2 - (x[1]-5)**2 - (x[2]-5)**2)/100

# CONSTRAINTS (form x^2+y^2-a <= 0)
def constraint_functions(x):
    const = (x[0]-x[3])**2 + (x[1]-x[4])**2 + (x[2]-x[5])**2 - 0.0625
    return const

# PROBLEM DEFINITION
problem = structure() # define the problem as a structure variable
problem.costfunc = cost_function # define the problem's cost function
problem.constraints = constraint_functions # define the problem's nonlinear constraints
problem.constraints_toll = 1e-10 # define the problem's constraints tollerance
problem.nvar_cont = 3 # define number of continuous variables in search space
problem.nvar_disc = 3 # define number of discrete variables in search space
problem.nvar = problem.nvar_cont + problem.nvar_disc # define total number of variables
problem.index_cont = [0, 1, 2] # define the indexes of continuous variables
problem.index_disc = [3, 4, 5] # define the indexes of discrete variables
problem.varmin_cont = [0, 0, 0] # lower bound of continuous variables
problem.varmax_cont = [10, 10, 10] # upper bound of continuous variables
problem.varmin_disc = [1, 1, 1] # lower bound of discrete variables
problem.varmax_disc = [9, 9, 9] # upper bound of discrete variables
problem.varmin = problem.varmin_cont + problem.varmin_disc # lower bound of all variables
problem.varmax = problem.varmax_cont + problem.varmax_disc # upper bound of all variables

# GA PARAMETERS
params = structure()
params.maxrep = 1 # maximum number of repetitions
params.stoprep = 3 # number of same solutions to stop repeating 
params.digits = 6 # accuracy of digits for a position being the same
params.maxit = 500 # maximum number of iterations 
params.stopit = 100 # number of repetitions of same optimum point before breaking
params.tollfitness = 1e-6 # fitness difference tollerance for breaking iterations
params.tollpos = 1e-6 # position difference tollerance for breaking iterations
params.npop = 300 # size of initial population
params.pc = 3 # proportion of children to main population
params.beta = 0.3 # Boltzman constant for parent selection probability
params.gamma = 0.8 # parameter for crossover
params.mu_cont = 0.3 # mutation threshold for continuous variables
params.mu_disc = 0.4 # mutation threshold for discrete variables
params.sigma = 0.3 # standard deviation of gene mutation 


# RUN GA
out = ga.run(problem, params)

#################################### PLOT RESULTS ####################################

best = np.inf
j = 0

for i in range(out.n_rep):
    print("\nSolution {} has position: {} and fitness {}".format(i+1,out.POS[i],out.FITNESS[i]))
    if out.FITNESS[i]<best:
       best = out.FITNESS[i]
    j = i
print("\n\n THE best solution found was number {} with position: {} and fitness: {}".format(j+1,out.POS[j],out.FITNESS[j]))

# COST - ITERATION
plt.figure()
for k in range (out.n_rep):
    plt.plot(out.IT[:,k][out.IT[:,k] != 0] , out.fitness[:,k][0:np.shape(out.IT[:,k][out.IT[:,k] != 0])[0]], label = "Repetition {}".format(k+1))

plt.xlabel('Number of iterations')
plt.ylabel('Best Fitness value')
plt.title('Genetic Algorithm (GA)')
plt.grid(True)
plt.legend()


# COST FUNCTION WITH FOUND MINIMA
# 1 VARIABLE
if problem.nvar == 1:
    X = np.linspace(problem.varmin,problem.varmax,100000)
    plt.figure()
    plt.plot(X,problem.costfunc(X), label = 'Cost Function')
    minima = plt.plot(out.POS,problem.costfunc(out.POS),'o', color = 'black', alpha = 0.6)
    minimum = plt.plot(out.POS[j],problem.costfunc(out.POS[j]),'o', color = 'red', alpha = 1)
    plt.xlim(problem.varmin, problem.varmax)
    plt.title('Cost Function')
    plt.grid(True)


# 2 VARIABLES
if problem.nvar == 2:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.linspace(problem.varmin[0],problem.varmax[0],1000)
    Y = np.linspace(problem.varmin[1],problem.varmax[1],1000)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y,problem.costfunc([X,Y]) , cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha = 0.3)

    minima = ax.scatter(out.POS[:,0],out.POS[:,1],problem.costfunc([out.POS[:,0],out.POS[:,1]]), 'o',color = 'black', alpha = 0.6, s = 1)
    minimum = ax.scatter(out.POS[j,0],out.POS[j,1],problem.costfunc([out.POS[j,0],out.POS[j,1]]), 'o',color = 'red', alpha = 1, s = 5)

    ax.set_xlim(problem.varmin[0],problem.varmax[0])
    ax.set_ylim(problem.varmin[1],problem.varmax[1])
    fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

