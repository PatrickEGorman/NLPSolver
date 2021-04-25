# NLPSolver

The following application uses the penalty method to approximate a nonlinear program
## Input
The program takes in a second degree objective function in terms of variables x,y,z

It also takes in a single constraint of the form g(x,y,z)-b which is assumed to equal 0

Finally, it takes a tolerance level for the approximation

It is important to note that the recognized code for an exponential is "**" and not "^".  
The program is only guaranteed to work with second degree polynomials with a min or max in terms of x, y and z variables, and a linear constraint equal to 0.
## Output
It prints out a calculated extreme point within a specified tolerance

It also prints out whether the point is a max or a min

## Running The App
First enter the command ```pip install -r requirements.txt``` in a shell window while in the main directory.

Then enter ```python main.py```

Answer prompts for polynomial, constraint, and tolerance making sure it's in proper form and a polynomial this app can handle