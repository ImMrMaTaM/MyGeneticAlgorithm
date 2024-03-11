import numpy as np
# When dealing with an equality constraint: write it as two inequality constraints:
eq_constraint_toll = 0.01
# h(x) = 0  ->  h(x) - 0.01 <= 0 and -0.01 - h(x) <= 0


########################################### G01 ###########################################
def G01_function(x):
    return 5*(x[0]+x[1]+x[2]+x[3]) - \
              5*(x[0]**2+x[1]**2+x[2]**2+x[3]**2) - \
                (x[0]+x[1]+x[2]+x[3]+x[4]+x[5]+x[6]+x[7]+x[8]+x[9]+x[10]+x[11]+x[12]) 

def G01_constraints(x):
    const1 = 2*x[0] + 2*x[1] + x[9] + x[10] - 10
    const2 = 2*x[0] + 2*x[2] + x[9] + x[10] - 10
    const3 = 2*x[1] + 2*x[2] + x[10] + x[11] - 10
    const4 = -8*x[0] + x[9]
    const5 = -8*x[1] + x[10]
    const6 = -8*x[2] + x[11]
    const7 = -2*x[3] - x[4] + x[9]
    const8 = -2*x[5] - x[6] + x[10]
    const9 = -2*x[7] - x[8] + x[11]
    return const1, const2, const3, const4, const5, const6, const7, const8, const9

nvar_G01 = 13
varmin_G01 = [0,0,0,0,0,0,0,0,0,0,0,0,0]
varmax_G01 = [1,1,1,1,1,1,1,1,1,100,100,100,1]

# I think it's correct (-19) but different from given answer (-15)

########################################### G02 ###########################################
def G02_function(x):
    return -np.abs( ( (np.cos(x[0])**4+np.cos(x[1])**4+np.cos(x[2])**4+np.cos(x[3])**4+np.cos(x[4])**4+np.cos(x[5])**4+np.cos(x[6])**4+np.cos(x[7])**4+np.cos(x[8])**4+np.cos(x[9])**4+np.cos(x[10])**4\
                +np.cos(x[11])**4+np.cos(x[12])**4+np.cos(x[13])**4+np.cos(x[14])**4+np.cos(x[15])**4+np.cos(x[16])**4+np.cos(x[17])**4+np.cos(x[18])**4+np.cos(x[19])**4)\
                    -2*(np.cos(x[0])**2*np.cos(x[1])**2*np.cos(x[2])**2*np.cos(x[3])**2*np.cos(x[4])**2*np.cos(x[5])**2*np.cos(x[6])**2*np.cos(x[7])**2*np.cos(x[8])**2*np.cos(x[9])**2*np.cos(x[10])**2\
                        *np.cos(x[11])**2*np.cos(x[12])**2*np.cos(x[13])**2*np.cos(x[14])**2*np.cos(x[15])**2*np.cos(x[16])**2*np.cos(x[17])**2*np.cos(x[18])**2*np.cos(x[19])**2))/ \
                            (np.sqrt(1*x[0]**2 + 2*x[1]**2 + 3*x[2]**2 + 4*x[3]**2 + 5*x[4]**2 + 6*x[5]**2 + 7*x[6]**2 + 8*x[7]**2 + 9*x[8]**2 + 10*x[9]**2 + \
                                11*x[10]**2 + 12*x[11]**2 + 13*x[12]**2 + 14*x[13]**2 + 15*x[14]**2 + 16*x[15]**2 + 17*x[16]**2 + 18*x[17]**2 + 19*x[18]**2 + 20*x[19]**2) ) )
                                             
def G02_constraints(x):
    const1 = 0.75 - np.prod(x)
    const2 = np.sum(x) - 7.5*20
    return const1, const2

nvar_cont_G02 = 20
nvar_disc_G02 = 0
index_cont_G02 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
index_disc_G02 = []
varmin_cont_G02 = 0
varmax_cont_G02 = 10
varmin_disc_G02 = 0
varmax_disc_G02 = 1

# YES

########################################### G03 ###########################################     
def G03_function(x):
    return -(np.sqrt(10))**10 * np.prod(x)

def G03_constraints(x):
    const1 = np.sum(np.power(x,2)) - 1 - eq_constraint_toll
    const2 = - np.sum(np.power(x,2)) + 1 - eq_constraint_toll
    return const1, const2

nvar_G03 = 10
varmin_G03 = 0
varmax_G03 = 1

# GOOD results considering we are approximating an equality constraint (difficulty in choosing the right bounds (0.99-1.01))

########################################### G04 ###########################################
def G04_function(x):
    return 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4] + 37.293239*x[0] - 40792.141

def G04_constraints(x):
    const1 = 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4] - 92
    const2 = -85.334407 - 0.0056858*x[1]*x[4] - 0.0006262*x[0]*x[3] + 0.0022053*x[2]*x[4]
    const3 = 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2 - 110
    const4 = -80.51249 - 0.0071317*x[1]*x[4] - 0.0029955*x[0]*x[1] - 0.0021813*x[2]**2 + 90
    const5 = 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3] - 25
    const6 = -9.300961 - 0.0047026*x[2]*x[4] - 0.0012547*x[0]*x[2] - 0.0019085*x[2]*x[3] + 20
    return const1, const2, const3, const4, const5, const6

nvar_G04 = 5
varmin_G04 = [78,33,27,27,27]
varmax_G04 = [102,45,45,45,45]

# YES

########################################### G05 ###########################################

def G05_function(x):
    return 3*x[0] + 0.000001*x[0]**3 + 2*x[1] + (0.000002/3)*x[1]**3

def G05_constraints(x):
    const1 = -x[3] + x[2] - 0.55
    const2 = -x[2] + x[3] - 0.55
    const3 = 1000*np.sin(-x[2]-0.25) + 1000*np.sin(-x[3]-0.25) + 894.8 - x[0] - eq_constraint_toll
    const4 = - 1000*np.sin(-x[2]-0.25) - 1000*np.sin(-x[3]-0.25) - 894.8 + x[0] - eq_constraint_toll
    const5 = 1000*np.sin(x[2]-0.25) + 1000*np.sin(x[2]-x[3]-0.25) + 894.8 - x[1] - eq_constraint_toll
    const6 = -1000*np.sin(x[2]-0.25) - 1000*np.sin(x[2]-x[3]-0.25) - 894.8 + x[1] - eq_constraint_toll
    const7 = 1000*np.sin(x[3]-0.25) + 1000*np.sin(x[3]-x[2]-0.25) + 1294.8 - eq_constraint_toll
    const8 = -1000*np.sin(x[3]-0.25) - 1000*np.sin(x[3]-x[2]-0.25) - 1294.8 - eq_constraint_toll
    return const1, const2, const3, const4, const5, const6, const7, const8

nvar_G05 = 4
varmin_G05 = [0,0,-0.55,-0.55]
varmax_G05 = [1200,1200,0.55,0.55]

# NO (difficult to work with multiple equality constraints, as no feasible answer is found)

########################################### G06 ###########################################
def G06_function(x):
    return (x[0]-10)**3 + (x[1]-20)**3

def G06_constraints(x):
    const1 = -(x[0]-5)**2 - (x[1]-5)**2 + 100 
    const2 = (x[0]-6)**2 + (x[1]-5)**2 - 82.81
    return const1, const2

nvar_cont_G06 = 2
nvar_disc_G06 = 0
index_cont_G06 = [0, 1]
index_disc_G06 = []
varmin_cont_G06 = [13,0]
varmax_cont_G06 = [100,100]
varmin_disc_G06 = 0
varmax_disc_G06 = 1

# YES 

########################################### G07 ###########################################
def G07_function(x):
    return x[0]**2 + x[1]**2 + x[0]*x[1] - 14*x[0] - 16*x[1] + (x[2]-10)**2\
        + 4*(x[3]-5)**2 + (x[4]-3)**2 + 2*(x[5]-1)**2 + 5*x[6]**2\
            + 7*(x[7]-11)**2 + 2*(x[8]-10)**2 + (x[9]-7)**2 + 45

def G07_constraints(x):
    const1 = -105 + 4*x[0] + 5*x[1] - 3*x[6] + 9*x[7]
    const2 = 10*x[0] - 8*x[1] - 17*x[6] + 2*x[7]
    const3 = -8*x[0] + 2*x[1] + 5*x[8] - 2*x[9] - 12
    const4 = 3*(x[0]-2)**2 + 4*(x[1]-3)**2 + 2*x[2]**2 - 7*x[3] - 120
    const5 = 5*x[0]**2 + 8*x[1] + (x[2]-6)**2 - 2*x[3] - 40
    const6 = x[0]**2 + 2*(x[1]-2)**2 - 2*x[0]*x[1] + 14*x[4] - 6*x[5]
    const7 = 0.5*(x[0]-8)**2 + 2*(x[1]-4)**2 + 3*x[4]**2 - x[5] - 30
    const8 = -3*x[0] + 6*x[1] + 12*(x[8]-8) - 7*x[9]
    return const1, const2, const3, const4, const5, const6, const7, const8

nvar_G07 = 10
varmin_G07 = -10
varmax_G07 = 10

# YES 

########################################### G08 ###########################################
def G08_function(x):
    return -(np.sin(2*np.pi*x[0])**3 * np.sin(2*np.pi*x[1]))/((x[0]**3)*(x[0]+x[1]))

def G08_constraints(x):
    const1 = x[0]**2 - x[1] + 1
    const2 = 1 - x[0] + (x[1]-4)**2
    return const1, const2

nvar_G08 = 2
varmin_G08 = 0
varmax_G08 = 10

# NO (problem when dividing by zero)

########################################### G09 ###########################################
def G09_function(x):
    return (x[0]-10)**2 + 5*(x[1]-12)**2 + x[2]**4 + 3*(x[3]-11)**2 +\
        10*x[4]**6 + 7*x[5]**2 + x[6]**4 - 4*x[5]*x[6] - 10*x[5] - 8*x[6]

def G09_constraints(x):
    const1 = -127 + 2*x[0]**2 + 3*x[1]**4 + x[2] + 4*x[3]**2 + 5*x[4]
    const2 = -282 + 7*x[0] + 3*x[1] + 10*x[2]**2 + x[3] - x[4]
    const3 = -196 + 23*x[0] + x[1]**2 + 6*x[5]**2 - 8*x[6]
    const4 = 4*x[0]**2 + x[1]**2 - 3*x[0]*x[1] + 2*x[2]**2 + 5*x[5] - 11*x[6]
    return const1, const2, const3, const4

nvar_G09 = 7
varmin_G09 = -10
varmax_G09 = 10

# YES 

########################################### G10 ###########################################
def G10_function(x):
    return x[0] + x[1] + x[2]

def G10_constraints(x):
    const1 = -1 + 0.0025*(x[3]+x[5])
    const2 = -1 + 0.0025*(x[4]+x[6]-x[3])
    const3 = -1 + 0.01*(x[7]-x[4])
    const4 = -x[0]*x[5] + 833.33252*x[3] + 100*x[0] - 83333.333
    const5 = -x[1]*x[6] + 1250*x[4] + x[1]*x[3] - 1250*x[3]
    const6 = -x[2]*x[7] + 1250000 + x[2]*x[4] - 2500*x[4]
    return const1, const2, const3, const4, const5, const6

nvar_cont_G10 = 8
nvar_disc_G10 = 0
index_cont_G10 = [0, 1, 2, 3, 4, 5, 6, 7]
index_disc_G10 = []
varmin_cont_G10 = [100,1000,1000,10,10,10,10,10]
varmax_cont_G10 = [10000,10000,10000,1000,1000,1000,1000,1000]
varmin_disc_G10 = 0
varmax_disc_G10 = 1

# NO (can't find feasible region)

########################################### G11 ###########################################
def G11_function(x):
    return x[0]**2 + (x[1]-1)**2

def G11_constraints(x):
    const1 = x[1] - x[0]**2 - eq_constraint_toll
    const2 = x[0]**2 - x[1] - eq_constraint_toll
    return const1, const2

nvar_G11 = 2
varmin_G11 = [-1,-1]
varmax_G11 = [1,1]

# YES 

########################################### G12 ###########################################


########################################### G13 ###########################################
def G13_function(x):
    return np.exp(x[0]*x[1]*x[2]*x[3]*x[4])

def G13_constraints(x):
    const1 = x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2 + x[4]**2 - 10 -  eq_constraint_toll
    const2 = - x[0]**2 - x[1]**2 - x[2]**2 - x[3]**2 - x[4]**2 + 10 -  eq_constraint_toll
    const3 = x[1]*x[2] - 5*x[3]*x[4] - eq_constraint_toll
    const4 = - x[1]*x[2] + 5*x[3]*x[4] - eq_constraint_toll
    const5 = x[0]**3 + x[1]**3 + 1 - eq_constraint_toll
    const6 = - x[0]**3 - x[1]**3 - 1 - eq_constraint_toll
    return const1, const2, const3 ,const4, const5, const6

nvar_G13 = 5
varmin_G13 = [-2.3, -2.3, -3.2, -3.2, -3.2]
varmax_G13 = [2.3, 2.3, 3.2, 3.2, 3.2]

# NO

########################################### G14 ###########################################


########################################### G15 ###########################################
def G15_function(x):
    return 1000 - x[0]**2 - 2*x[1]**2 - x[2]**2 - x[0]*x[1] - x[0]*x[2]

def G15_constraints(x):
    const1 = x[0]**2 + x[1]**2 + x[2]**2 - 25 - eq_constraint_toll
    const2 = -x[0]**2 - x[1]**2 - x[2]**2 + 25 - eq_constraint_toll
    const3 = 8*x[0] + 14*x[1] + 7*x[2] - 56 - eq_constraint_toll
    const4 = -8*x[0] - 14*x[1] - 7*x[2] + 56 - eq_constraint_toll
    return const1, const2, const3 ,const4

nvar_cont_G15 = 3
nvar_disc_G15 = 0
index_cont_G15 = [0, 1, 2]
index_disc_G15 = []
varmin_cont_G15 = [0,0,0]
varmax_cont_G15 = [10,10,10]
varmin_disc_G15 = 0
varmax_disc_G15 = 1

# NO

########################################### G18 ###########################################
def G18_function(x):
    return -0.5*(x[0]*x[3] - x[1]*x[2] + x[2]*x[8] - x[4]*x[8] + x[4]*x[7] - x[5]*x[6]) 

def G18_constraints(x):
    const1 = x[2]**2 + x[3]**2 - 1 
    const2 = x[8]**2 - 1
    const3 = x[4]**2 + x[5]**2 - 1
    const4 = x[0]**2 + (x[1]-x[8])**2 - 1
    const5 = (x[0]-x[4])**2 + (x[1]-x[5])**2 - 1
    const6 = (x[0]-x[6])**2 + (x[1]-x[7])**2 - 1
    const7 = (x[2]-x[4])**2 + (x[3]-x[5])**2 - 1
    const8 = (x[2]-x[6])**2 + (x[3]-x[7])**2 - 1
    const9 = x[6]**2 + (x[7]-x[8])**2 - 1
    const10 = x[1]*x[2] - x[0]*x[3]
    const11 = -x[2]*x[8] 
    const12 = x[4]*x[8]
    const13 = x[5]*x[6] - x[4]*x[7]
    return const1, const2, const3 ,const4, const5, const6, const7, const8, const9, const10, const11, const12, const13

nvar_G18 = 9
varmin_G18 = [-10, -10, -10, -10, -10, -10, -10, -10, 0]
varmax_G18 = [10, 10, 10, 10, 10, 10, 10, 10, 20]

# YES

