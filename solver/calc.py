from sympy import *

from .util import hessian, crit_points


# NLP Solver Object
class NLPSolver(object):

    # Initialize class variables
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

    # Recursive function to iterate through penalty functions until tol achieved
    def solve(self):
        penalty_function = self.objective + self.u * self.p
        crit = crit_points(self.x, penalty_function)
        val = self.objective.subs(crit)
        diff = self.p.subs(crit)
        # Iterate until tolerance is reached if function doesn't have a saddle point
        if self.u * diff > self.tolerance and not (self.min and self.max):
            print("Objective value " + str(val) + " at points " + str(crit))
            print(str(self.u * diff) + ">" + str(self.tolerance) + " continuing iterations" + "for u=" + str(self.u))
            self.u = self.u * 10
            self.solve()
        else:  # Once tol achieved output final values
            print(str(self.u * diff) + "<=" + str(self.tolerance) + " ceasing iterations")
            print("Final objective value " + str(val) + " at points " + str(crit) + "for u=" + str(self.u))
            # If mixed eigenvalues function has a saddle point
            if self.min and self.max:
                print("Function has a saddle point.  Penalty method not effective.")
            elif self.min:
                print("Point is a minimum")
            elif self.max:
                print("Point is a maximum")
            else:
                print("Point is unknown")

    # Analyze hessian to see if function has max, min, or saddle
    def analyze_hessian(self):
        h = hessian(self.x, self.objective)
        self.min = False
        self.max = False
        for eig in h.eigenvals():
            if eig > 0:
                self.min = True
            if eig < 0:
                self.max = True
            # If both true, function has a saddle