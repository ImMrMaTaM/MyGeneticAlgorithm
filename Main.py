"""
This is the code to be run in order to solve the optimization problem using the Genetic Algorithm (GA)
implemented in the ga.py file.

"""

# Import libraries

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from ypstruct import structure

# Import other filess

import ga
import Test_functions_constrained as tfc

# Commands for LaTeX style plots

matplotlib.rcParams['mathtext.fontset'] = 'cm' # 'cm' or 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

matplotlib.rc('xtick', labelsize = 15)
matplotlib.rc('ytick', labelsize = 15)

# Cost function to be minimized

def cost_function(x): 
    return tfc.G10_function(x)

# Inequality constraints (form x ^ 2 + y ^ 2 - a <= 0)

def constraint_functions(x):
    return tfc.G10_constraints(x)

# Define the structure of the problem 

problem = structure() # define the problem as a structure variable
problem.costfunc = cost_function # define the problem's cost function
problem.constraints = constraint_functions # define the problem's nonlinear constraints
problem.constraints_toll = 1e-10 # define the problem's constraints tollerance
problem.nvar_cont = tfc.nvar_cont_G10 # define number of continuous variables in search space
problem.nvar_disc = tfc.nvar_disc_G10 # define number of discrete variables in search space
problem.nvar = problem.nvar_cont + problem.nvar_disc # define total number of variables
problem.index_cont = tfc.index_cont_G10 # define the indexes of continuous variables
problem.index_disc = tfc.index_disc_G10 # define the indexes of discrete variables
problem.varmin_cont = tfc.varmin_cont_G10 # lower bound of continuous variables
problem.varmax_cont = tfc.varmax_cont_G10 # upper bound of continuous variables
problem.varmin_disc = tfc.varmin_disc_G10 # lower bound of discrete variables
problem.varmax_disc = tfc.varmax_disc_G10 # upper bound of discrete variables

# Define parameters for the genetic algorithm

params = structure()
params.maxrep = 5 # maximum number of repetitions
params.stoprep = 3 # number of same solutions to stop repeating 
params.digits = 6 # accuracy of digits for a position being the same
params.maxit = 1000 # maximum number of iterations 
params.stopit = 100 # number of repetitions of same optimum point before breaking
params.tollfitness = 1 # fitness difference tollerance for breaking iterations
params.tollpos = 1 # position difference tollerance for breaking iterations
params.npop = 200 # size of initial population
params.pc = 3 # proportion of children to main population
params.beta = 0.3 # Boltzman constant for parent selection probability
params.gamma = 0.8 # parameter for crossover
params.adaptmut_it = 10 # number of iterations for adaptive mutation
params.mu_cont = 0.2 # mutation threshold for continuous variables
params.mu_disc = 0.2 # mutation threshold for discrete variables
params.sigma = 1 # standard deviation of gene mutation 

"""
Run the genetic algorithm by calling the script ga.py and passing the problem and parameters as arguments.
"""

out = ga.run(problem, params)

"""
Plot the results of the genetic algorithm.
"""

best = np.inf # Comparison value
j = 0 # Initialize the counter

# Display the information about the solutions

for i in range(out.n_rep):
    
    print("\nSolution {} has position: {} and fitness {}".format(i + 1, out.POS[i], out.FITNESS[i]))
    if out.FITNESS[i] < best:
        
       best = out.FITNESS[i]
       
    j = i
    
print("\n\n THE best solution found was number {} with position: {} and fitness: {}".format(j + 1, out.POS[j], out.FITNESS[j]))

# Plot cost - iteration

plt.figure(figsize = [8, 6])

for k in range (out.n_rep):
    
    plt.plot(out.IT[:, k][out.IT[:, k] != 0] , out.fitness[:, k][0 : np.shape(out.IT[:, k][out.IT[:, k] != 0])[0]], 
             label = "Repetition {}".format(k + 1))

plt.xlabel('Number of iterations', fontsize = 15)
plt.ylabel('Best Fitness value', fontsize = 15)
plt.title('Genetic Algorithm (GA)', fontsize = 20)
plt.grid(True)
plt.legend()


# COST FUNCTION WITH FOUND MINIMA
# 1 VARIABLE
if problem.nvar_cont == 1:
    X = np.linspace(problem.varmin_cont,problem.varmax_cont,100000)
    plt.figure()
    plt.plot(X,problem.costfunc(X), label = 'Cost Function')
    minima = plt.plot(out.POS,problem.costfunc(out.POS),'o', color = 'black', alpha = 0.6)
    minimum = plt.plot(out.POS[j],problem.costfunc(out.POS[j]),'o', color = 'red', alpha = 1)
    plt.xlim(problem.varmin, problem.varmax)
    plt.title('Cost Function')
    plt.grid(True)


# 2 VARIABLES
if problem.nvar_cont == 2:
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X = np.linspace(problem.varmin_cont[0],problem.varmax_cont[0],1000)
    Y = np.linspace(problem.varmin_cont[1],problem.varmax_cont[1],1000)
    X, Y = np.meshgrid(X, Y)

    surf = ax.plot_surface(X, Y,problem.costfunc([X,Y]) , cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha = 0.3)

    minima = ax.scatter(out.POS[:,0],out.POS[:,1],problem.costfunc([out.POS[:,0],out.POS[:,1]]), 'o',color = 'black', alpha = 0.6, s = 1)
    minimum = ax.scatter(out.POS[j,0],out.POS[j,1],problem.costfunc([out.POS[j,0],out.POS[j,1]]), 'o',color = 'red', alpha = 1, s = 5)

    ax.set_xlim(problem.varmin_cont[0],problem.varmax_cont[0])
    ax.set_ylim(problem.varmin_cont[1],problem.varmax_cont[1])
    fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

