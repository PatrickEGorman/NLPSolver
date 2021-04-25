from sympy import *


# Generates the hession matrix from list of variables and expression
def hessian(vars, expr):
    H = zeros(len(vars), len(vars))
    for i1 in range(len(vars)):
        for i2 in range(len(vars)):
            H[i1, i2] = diff(expr,vars[i1], vars[i2])
    return H

# Determines the critical points from list of variables and expression
def crit_points(vars, expr):
    gradient = [0]
    for x in vars:
        gradient.append(diff(expr,x))
    return solve(gradient, vars)
