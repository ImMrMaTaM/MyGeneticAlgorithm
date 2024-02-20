import numpy as np
from ypstruct import structure

############################################## CONSTRAINTS ##############################################
def constraint_violation(x, constraint_functions):
    vals = constraint_functions(x)
    if len(vals) == 1:
        return vals[0]
    res = 0
    for val in vals:
        if val > 0:
            res += val
    return res


"""def constraint_violation(x, lin_lhs, lin_rhs, nonlinear):
    lin_vio = linear_constraint_violation(lin_lhs, lin_rhs, x)  # compute linear inequality violation
    non_vio = nonlinear_constraints(x.T[0], nonlinear)  # compute non-linear inequality violation
    return lin_vio + non_vio """

############################################## BOUNDARIES ##############################################

def bound_violation(x, varmin, varmax):
    n = len(x)
    res = 0.
    for i in range(n):
        val = x[i]
        if val < varmin[i]:
            res += varmin[i] - val
        if val > varmax[i]:
            res += val - varmax[i]
    return res

