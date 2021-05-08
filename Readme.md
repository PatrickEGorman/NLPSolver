# NLPSolver

The following application uses the penalty method to approximate a nonlinear program.

## Input

The program takes in a second degree objective function in terms of variables x,y,z.  It works best
with second degree functions that have a max or a min.  It may or may not work with higher degree
functions.  It can detect if a second order function has a saddle point but will likely not get
the correct location of the saddle.

It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0.  This program
has only been tested with first order constraints so performance with higher order constraints is not
guaranteed.

Finally, it takes a tolerance level for the approximation.  The program will run until this tolerance 
is achieved.

The program does not take in a value for maximize or minimize because it adaptively detects whether a point is a max,
a min or a saddle.

It is important to note that the recognized code for an exponential is ```**``` and not ```^```.  In the case
of 2 terms being multiplied together a ```*``` symbol is required.  The program will not recognize any variables
other than x,y, and z.  It can run with 2 out of 3 variables, but due to some issue in the libraries used, it
doesn't seem to do well with only 1 out of 3 variables.
The program is only guaranteed to work with second degree polynomials with a min or max in terms of x, y and z
variables, and a linear constraint equal to 0.

## Output

It prints out a calculated extreme point within a specified tolerance and all steps taken to calculate the extreme
point as well as the value of the penalty coefficient u.

It also prints out whether the point is a max or a min.  It can detect if a function has a saddle point, but is not accurate in determining 
the actual location of the saddle.

## Running The App

First enter the command ```pip install -r requirements.txt``` in a shell window while in the main directory.

Then enter ```python main.py```

Answer prompts for polynomial, constraint, and tolerance making sure it's in proper form and a polynomial this app can
handle.

## Example runs

Here the program is shown to calculate a maximum of a 2 variable second degree function.
```
python main.py
The following application uses the penalty method to approximate a nonlinear program
The program takes in a second degree objective function in terms of variables x,y,z
It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0
Finally, it takes a tolerance level for the approximation
--------------------------------------------------------------
It prints out a calculated extreme point within a specified tolerance
It also prints out whether the point is a max or a min
Input Objective Function: -x**2-y**2
Input Constraint: 0=x+y-2
Input tolerance: 0.01
Objective value -0.125000000000000 at points {x: -0.250000000000000, y: -0.250000000000000}
0.625000000000000>0.01 continuing iterationsfor u=0.1
Objective value -8.00000000000000 at points {y: 2.00000000000000, x: 2.00000000000000}
4.00000000000000>0.01 continuing iterationsfor u=1.0
Objective value -2.21606648199446 at points {x: 1.05263157894737, y: 1.05263157894737}
0.110803324099723>0.01 continuing iterationsfor u=10.0
Objective value -2.02015100628772 at points {x: 1.00502512562814, y: 1.00502512562814}
0.0101007550314382>0.01 continuing iterationsfor u=100.0
0.00100100075050061<=0.01 ceasing iterations
Final objective value -2.00200150100063 at points {x: 1.00050025012506, y: 1.00050025012506}for u=1000.0
Point is a maximum
DONE

```

Here the program is shown to calculate the minimum of a 3 variable second degree function.
```
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

Here the program does not correctly get the location of a saddle point of a second degree function, but does accurately
determine that a saddle point exists due to properties of the Hessians eigenvalues.
```
pInput Objective Function: x**2-y**2
Input Constraint: 0=x+y-5
Input tolerance: 0.1
Objective value 0 at points {x: 0.500000000000000, y: -0.500000000000000}
2.50000000000000>0.1 continuing iterationsfor u=0.1
Objective value 0 at points {x: 5.00000000000000, y: -5.00000000000000}
25.0000000000000>0.1 continuing iterationsfor u=1.0
Objective value 0 at points {x: 50.0000000000000, y: -50.0000000000000}
250.000000000000>0.1 continuing iterationsfor u=10.0
Objective value 0 at points {x: 500.000000000000, y: -500.000000000000}
2500.00000000003>0.1 continuing iterationsfor u=100.0
Objective value 0 at points {x: 5000.00000000000, y: -5000.00000000000}
25000.0000000044>0.1 continuing iterationsfor u=1000.0
Objective value 0 at points {x: 50000.0000000000, y: -50000.0000000000}
250000.000000696>0.1 continuing iterationsfor u=10000.0
Objective value 0 at points {x: 500000.000000000, y: -500000.000000000}
2499999.99998098>0.1 continuing iterationsfor u=100000.0
Objective value 0 at points {x: 5000000.00000000, y: -5000000.00000000}
25000000.0002817>0.1 continuing iterationsfor u=1000000.0
Objective value 0 at points {x: 50000000.0000000, y: -50000000.0000000}
250000000.251965>0.1 continuing iterationsfor u=10000000.0
Objective value 0 at points {x: 500000000.000000, y: -500000000.000000}
2500000030.63495>0.1 continuing iterationsfor u=100000000.0
Objective value 0 at points {x: 5000000000.00000, y: -5000000000.00000}
25000004087.0187>0.1 continuing iterationsfor u=1000000000.0
Objective value 0 at points {x: 50000000000.0000, y: -50000000000.0000}
250000041320.187>0.1 continuing iterationsfor u=10000000000.0
Objective value 0 at points {x: 500000000000.000, y: -500000000000.000}
2500000413651.87>0.1 continuing iterationsfor u=100000000000.0
Objective value 0 at points {x: 5000000000000.00, y: -5000000000000.00}
25004445226649.9>0.1 continuing iterationsfor u=1000000000000.0
Objective value 0 at points {x: 50000000000000.0, y: -50000000000000.0}
249600520792536.>0.1 continuing iterationsfor u=10000000000000.0
Objective value 0 at points {x: 500000000000000., y: -500000000000000.}
2.49600520792581e+15>0.1 continuing iterationsfor u=100000000000000.0
0<=0.1 ceasing iterations
Final objective value -y**2 + 25.0*(1 - 0.2*y)**2 at points {x: 5.0 - y}for u=1000000000000000.0
Function has a saddle point
DONE

```

## Code

Here is the code.  If you are going to run this, you are going to at least need sympy and possibly
mpmath (not sure about mpmath).  Simply try ```pip install sympy``` and ```pip install mpmath```.
The project was written in python3.  If ```python main.py``` is not working try  ```pip3 install ...```
```python3 main.py```

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
        if self.u * diff > self.tolerance: # repeate until tol achieved
            print("Objective value " + str(val) + " at points " + str(crit))
            print(str(self.u * diff) + ">" + str(self.tolerance) + " continuing iterations" + "for u=" + str(self.u))
            self.u = self.u * 10
            self.solve()
        else: #Once tol achieved output final values
            print(str(self.u * diff) + "<=" + str(self.tolerance) + " ceasing iterations")
            print("Final objective value " + str(val) + " at points " + str(crit) + "for u=" + str(self.u))
            # If mixed eigenvalues function has a saddle point
            if self.min and self.max:
                print("Function has a saddle point")
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

print("The following application uses the penalty method to approximate a nonlinear program")
print("The program takes in a second degree objective function in terms of variables x,y,z")
print("It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0")
print("Finally, it takes a tolerance level for the approximation")
print("--------------------------------------------------------------")
print("It prints out a calculated extreme point within a specified tolerance")
print("It also prints out whether the point is a max or a min")

# Take in user input
obj = input("Input Objective Function: ")
constraint = input("Input Constraint: 0=")
tol = input("Input tolerance: ")

# Pass input to solver and solve
nlpsolve = NLPSolver(['x', 'y', 'z'], objective=obj, constraint=constraint, tolerance=tol)
nlpsolve.solve()
print("DONE")
```

## Requirements

I'm not sure whether mpmath is required by sympy is definitely required.  This program
was build using python 3.8.  Use either ```pip install package_name``` or ```pip3 install package_name```
to install these packages.

```
mpmath==1.2.1
sympy==1.7.1```
