import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from ypstruct import structure
import ga
################################################################################

# COST FUNCTION TO MINIMIZE
def cost(x): 

    # return -20*np.exp(-0.2*np.sqrt(0.5*(x[0]**2+x[1]**2))) - np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))) + np.e + 20 # Akley function YES
    # return 100*np.sqrt(np.abs(x[1]-0.01*x[0]**2)) + 0.01*np.abs(x[0]+10) # Buckin function NO [-15,-3] [-5,3]
    # return -0.0001*(np.abs( np.sin(x[0])*np.sin(x[1])*np.exp(np.abs(100 -((np.sqrt(x[0]**2+x[1]**2))/(np.pi)))))+1)**0.1 # Cross-In-Tray function YES
    # return -((1+np.cos(12*np.sqrt(x[0]**2+x[1]**2)))/(0.5*(x[0]**2+x[1]**2)+2)) # Drop wave function YES
    # return -(x[1]+47)*np.sin(np.sqrt(np.abs(x[1]+0.5*x[0]+47))) - x[0]*np.sin(np.sqrt(np.abs(x[0]-(x[1]+47)))) # Eggholder function YES 
    # return ((np.sin(10*np.pi*x))/(2*x)) + (x-1)**4 # Gramacy & Lee function [0.5,2.5] YES
    # return -np.abs(np.sin(x[0])*np.cos(x[1])*np.exp(np.abs(1-((np.sqrt(x[0]**2+x[1]**2))/(np.pi)))))
    return 1 + (1/4000)*x[0]**2 + (1/4000)*x[1]**2 - np.cos(x[0])*np.cos(0.5*x[1]*np.sqrt(2)) # Griewank function [-600,600] YES
    # return 418.9829*2 - x[0]*np.sin(np.sqrt(np.abs(x[0]))) - x[1]*np.sin(np.sqrt(np.abs(x[1])))
    # return np.sin(3*np.pi*x[0])**2+(x[0]-1)**2*(1+np.sin(3*np.pi*x[1])*np.sin(3*np.pi*x[1]))+(x[1]-1)*(x[1]-1)*(1+np.sin(2*np.pi*x[1])*np.sin(2*np.pi*x[1])) # Levy function 13 [-10,10] YES
    # return (x[1] - (5.1/(4*np.pi**2))*x[0]**2 + (5/np.pi)*x[0] - 6)**2 + 10*(1-(1/(8*np.pi)))*np.cos(x[0]) + 10 # Branin function [-5, 0] [10,15] YES

# PROBLEM DEFINITION
problem = structure() # define the problem as a structure variable
problem.costfunc = cost # define the problem's cost function
problem.nvar = 2 # define number of variables in search space
problem.varmin = [-600, -600] # lower bound of variables
problem.varmax = [600,600] # upper bound of variables

# GA PARAMETERS
params = structure()

params.maxrep = 30 # maximum number of repetitions
params.stoprep = 10 # number of same solutions to stop repeating 
params.digits = 3 # accuracy of digits for a position being the same

params.maxit = 300 # maximum number of iterations 
params.stopit = 20 # number of repetitions of same optimum point before breaking
params.tollcost = 1e-5 # cost difference tollerance for breaking iterations
params.tollpos = 1e-5 # position difference tollerance for breaking iterations

params.npop = 100 # size of initial population
params.pc = 4 # proportion of children to main population
params.beta = 0.3 # Boltzman constant for parent selection probability
params.gamma = 0.8 # parameter for crossover
params.mu = 0.1 # mutation threshold
params.sigma = 0.2 # standard deviation of gene mutation 


# RUN GA
out = ga.run(problem, params)

#################################### PLOT RESULTS ####################################

best = np.inf
j = 0

for i in range(out.n_rep):
    print("\nSolution {} has position: {} and cost {}".format(i+1,out.POS[i],out.COST[i]))
    if out.COST[i]<best:
       best = out.COST[i]
    j = i
print("\n\n THE best solution found was number {} with position: {} and cost: {}".format(j+1,out.POS[j],out.COST[j]))

# COST - ITERATION
plt.figure()
for k in range (out.n_rep):
    plt.plot(out.IT[:,k][out.IT[:,k] != 0] , out.costs[:,k][0:np.shape(out.IT[:,k][out.IT[:,k] != 0])[0]], label = "Repetition {}".format(k+1))

plt.xlabel('Number of iterations')
plt.ylabel('Best Cost')
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

