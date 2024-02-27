import numpy as np
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

nvar_G02 = 20
varmin_G02 = 0
varmax_G02 = 10

# NO (very difficult to find the global minimum)

########################################### G03 ###########################################     
def G03_function(x):
    return -(np.sqrt(10))**10 * np.prod(x)

def G03_constraints(x):
    const1 = np.sum(np.power(x,2)) - 1.01
    const2 = 0.99 - np.sum(np.power(x,2))
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