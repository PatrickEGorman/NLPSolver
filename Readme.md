# NLPSolver

The following application uses the penalty method to approximate a nonlinear program.

## Input

The program takes in a second degree objective function in terms of variables x,y,z

It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0

Finally, it takes a tolerance level for the approximation

It is important to note that the recognized code for an exponential is ```**``` and not ```^```.  
The program is only guaranteed to work with second degree polynomials with a min or max in terms of x, y and z
variables, and a linear constraint equal to 0.

## Output

It prints out a calculated extreme point within a specified tolerance

It also prints out whether the point is a max or a min

## Running The App

First enter the command ```pip install -r requirements.txt``` in a shell window while in the main directory.

Then enter ```python main.py```

Answer prompts for polynomial, constraint, and tolerance making sure it's in proper form and a polynomial this app can
handle.

## Example run

``` 
python main.py
The following application uses the penalty method to approximate a nonlinear program
The program takes in a second degree objective function in terms of variables x,y,z
It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0
Finally, it takes a tolerance level for the approximation
--------------------------------------------------------------
It prints out a calculated extreme point within a specified tolerance
It also prints out whether the point is a max or a min
Input Objective Function: x**2+5*x+2*y**2-5*y+2*z**2-3*z+x*y+3         
Input Constraint: 0=x+y+y-5
Input tolerance: .001
Objective value -12.1325716603570 at points {x: -3.37209302325581, y: 2.44186046511628, z: 0.750000000000000}
1.21687398593835>0.001 continuing iterationsfor u=0.1
Objective value -8.52197542533081 at points {x: -2.82608695652174, y: 3.26086956521739, z: 0.750000000000000}
1.70132325141777>0.001 continuing iterationsfor u=1.0
Objective value -5.03453422496325 at points {x: -2.54491017964072, y: 3.68263473053892, z: 0.750000000000000}
0.322707877657859>0.001 continuing iterationsfor u=10.0
Objective value -4.44485375122607 at points {x: -2.50466708151836, y: 3.74299937772246, z: 0.750000000000000}
0.0348506398383903>0.001 continuing iterationsfor u=100.0
Objective value -4.38202663843236 at points {x: -2.50046854501156, y: 3.74929718248266, z: 0.750000000000000}
0.00351255084568576>0.001 continuing iterationsfor u=1000.0
0.000351531740301812<=0.001 ceasing iterations
Final objective value -4.37570307886012 at points {x: -2.50004687294931, y: 3.74992969057604, z: 0.750000000000000}for u=10000.0
Point is a minimum
DONE
```

## Code

```python
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

print("The following application uses the penalty method to approximate a nonlinear program")
print("The program takes in a second degree objective function in terms of variables x,y,z")
print("It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0")
print("Finally, it takes a tolerance level for the approximation")
print("--------------------------------------------------------------")
print("It prints out a calculated extreme point within a specified tolerance")
print("It also prints out whether the point is a max or a min")

obj = input("Input Objective Function: ")
constraint = input("Input Constraint: 0=")
tol = input("Input tolerance: ")

nlpsolve = NLPSolver(['x', 'y', 'z'], objective=obj, constraint=constraint, tolerance=tol)
nlpsolve.solve()
print("DONE")
```

## Requirements

```
mpmath==1.2.1
sympy==1.7.1```