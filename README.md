# FUNCTION MINIMIZATION WITH GENETIC ALGORITHM


## Project description 

This project aims at finding the global minimum of multi-variable functions using a genetic algorithm.

### Objective function $f(\vec{x})$ characteristics
- __Multi-variable__: $\vec{x}$ can contain any number of continuous and/or discrete variables. The type and number of variables must be specified.
- __Constrained__: any number of inequality constraints in the form $g_i(\vec{x})\leq0$ can be added.
<br />
Equality constraints such as $h(\vec{x})=0$ can be imitated using two inequality constraints as: $ g_i(\vec{x}) \leq 0 \leq g_j(\vec{x}) $
- __Bounded__: a lower and upper boundary must be specified for each variable.

### Genetic algorithm characteristics
- __Individuals__: each individual $k$ has 5 associated values: 
  - _position_: value of $\vec{x_k}=\left\{x_{k1},x_{k2},...\right\}$ 
  - _cost_: value of $f(\vec{x_k})$
  - _violation_: a measure of how much an individual is violating the constraints (equals to 0 if all constraints are satisfied)
  - _valid_: a boolean variable which is $True$ if the individual is satisfying all constraints and $False$ otherwise.
  - _fitness_: The value used to evaluate how important an individual is.<br /> 
    - If the individual is satisfying all constraints: $fit(\vec{x_k}) = cost(\vec{x_k})$ 

    - If any constraint isn't satisfied: $fit(\vec{x_k}) = \displaystyle\max_{\vec{x}\in valid}\{cost(\vec{x})\} + violat(\vec{x_k})$  

    This ensures that all valid indivuals have a lower fitness value than all invalid ones.

    <br />

- __Parents selection__: each individual $k$ has a certain probability $\Pi_k$ of being selected as a parent among the $n$ individuals of the population. This probability is calculated using __Boltzmann model__: $$\Pi_k=\exp\left(-\beta \frac {fit(\vec{x_k})}{\frac{1}{n}\sum_{i=1}^{n} fit(\vec{x_i})}\right)$$

    With $\beta$ being a tunable parameter. Note that:  $$if: \begin{cases}   \beta\to 0\:\Longrightarrow\:\Pi_k=\frac{1}{n}\:\forall k \\\beta\to\infty\:\Longrightarrow\: \Pi_k=\begin{cases}1\:\: \text{for best individual} \\0\:\: \text{otherwise}   \end{cases}  \end{cases}$$

    Parents are then chosen with the __roulette wheel selection method__.<br />
    All probabilities $\left( \Pi_1,\Pi_2,...,\Pi_n \right)$ are turned into cumulative probabilities  $\left( \Pi_{c,1},\Pi_{c,2},...,\Pi_{c,n}=1 \right)$
    <br />
    A number $r\in[0,1]$ is randomly picked and will fall into the cumulative probability of a certain individual. That individual is the selected parent.

    <br />

- __Crossover__: crossover is performed differently for continuous and discrete variables.
    - _Whole arithmetic recombination_ (continuous): $$\begin{array}{lcl} \vec{p_1} = \left\{ p_{11},p_{12},... \right\}\\\vec{p_2} = \left\{ p_{21},p_{22},...   \right\}\end{array} \to \begin{array}{lcl} c_{1i} = \alpha_i p_{1i}+(1-\alpha_i)p_{2i}\\c_{2i} = (1-\alpha_i) p_{1i}+\alpha_i p_{2i} \end{array}$$ 
    This allows a smoother transition from a generation to the next, at the cost of a slower convergence.
    <br />
    Parameter $\alpha_i$ is randomly picked for each gene: $\alpha_i \in\left[-\gamma,1+\gamma\right]$
    <br />
    $\gamma$ is a parameter appropriately tuned by the user. As $\gamma$ is selected to be larger, the children's genes will deviate further away from those of their parents. 
     <br />
    Whole arithmetic recombination can't be performed on discrete variables because multiplying them by $\alpha_i$ would turn them continuous.

    - _Uniform crossover_ (discrete): a random vector the same size as the parents is generated containing only 0 and 1. e.g. $\:\vec{v}=[0,1,1,0,0,1,...]$
      <br />
      $$\begin{array}{lcl} \vec{p_1} = \left\{ p_{11},p_{12},... \right\}\\\vec{p_2} = \left\{ p_{21},p_{22},...   \right\}\end{array} \to \begin{array}{lcl} \vec{c_1} = \vec{v} \cdot \vec{p_1} + (1-\vec{v}) \cdot \vec{p_2}\\\vec{c_2} = (1-\vec{v}) \cdot \vec{p_1}+\vec{v} \cdot \vec{p_2} \end{array}$$ 
      <br />
      This allows to maintain diversity while keeping the same variables throughout the generations.

      <br />

- __Mutation__: mutation is also performed differently for continuous and discrete variables.
<br />
  Each individual $\vec{x}$ has a certain probability of mutating its genes $\mu$.
  <br />
  A random vector the size of $\vec{x}$ is generated with values between $0$ and $1$.
  <br />
  All indices of elements in the random vector with value $<\mu$ are recorded and will be the elements that will undergo mutation in $\vec{x}$

  - _Gaussian mutation_ (continuous): 

    $$ x_i \to x_i+\sigma Z \:\:\text{with}\:\: Z\sim \mathcal{N}(0,1)$$
  
    Each mutating gene varies following a Gaussian distribution, driven by the standard deviation $\sigma$

  - _Random mutation_ (discrete):
    $$ x_i \to y_i \:\:\text{with}\:\: y_i \in \left\{x_{i,min}\: ,\:...\:,\:x_{i,max} \right\} $$
    This way each discrete gene is surely kept within its boundaries. 

  <br />
  Notice that mutation is adaptive, and varies depending on the current state of the GA.

  <br />

  - If the individual is invalid continuous variables mutate randomly, similarly to discrete variables. The mutation rate is also maximized: $\mu=1$ <br />
  This way the mutation search space is spread evenly and mutations occur much more frequently:it's easier to find feasible region.
  <br />

  - When a valid solution is found, continuous variables go back at mutating with the Gaussian mutation method and $\mu$ goes back to its original value. 
  <br />

  - If the same valid solution is being found after several iterations (number of consecutive iterations and tollerance are customizable), then values of $\mu$ and $\sigma$ change.<br /> $\mu$ gradually increases while $\sigma$ gradually decreases (both are bounded).<br />
  This allows an almost stuck algorithm to increase its searching frequency in the neighbourhood of the best solution found so far in an attempt at unstucking itself. <br />
  If a new solution is found, $\mu$ and $\sigma$ go back to their original value. 

  <br />

- __Boundaries enforcement__: continuous variables could step out of the boundaries due to crossover or mutation. To enforce those variables to stay within the boundaries the following operations are performed:
$$x_i=\max\left( x_i\:,\:x_{i,min}   \right)\:\: \text{to enforce lower bounds}$$
$$x_i=\min\left( x_i\:,\:x_{i,max}   \right)\:\: \text{to enforce upper bounds}$$

  
  


       
    

      

    





































