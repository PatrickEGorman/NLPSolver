from sympy import *

from .util import hessian, crit_points


class NLPSolver(object):

    def __init__(self, vars, objective, constraint, tolerance):
        self.x = symbols('x y z')
        self.u = 0.1
        try:
            self.objective = parse_expr(objective)
        except TypeError:
            print("Invalid objective function")
        try:
            self.constraint = parse_expr(constraint)
        except TypeError:
            print("Invalid constraint function")
        self.min = False
        self.max = False
        self.analyze_hessian()
        self.tolerance = float(tolerance)
        self.p = self.constraint ** 2

    def solve(self):
        penalty_function = self.objective + self.u * self.p
        crit = crit_points(self.x, penalty_function)
        val = self.objective.subs(crit)
        diff = self.p.subs(crit)
        if self.u * diff > self.tolerance:
            print("Objective value " + str(val) + " at points " + str(crit))
            print(str(self.u * diff) + ">" + str(self.tolerance) + " continuing iterations" + "for u=" + str(self.u))
            self.u = self.u * 10
            self.solve()
        else:
            print(str(self.u * diff) + "<=" + str(self.tolerance) + " ceasing iterations")
            print("Final objective value " + str(val) + " at points " + str(crit) + "for u=" + str(self.u))
            if self.min and self.max:
                print("Point is a saddle point")
            elif self.min:
                print("Point is a minimum")
            elif self.max:
                print("Point is a maximum")
            else:
                print("Point is unknown")

    def analyze_hessian(self):
        h = hessian(self.x, self.objective)
        self.min = False
        self.max = False
        for eig in h.eigenvals():
            if eig > 0:
                self.min = True
            if eig < 0:
                self.max = True
