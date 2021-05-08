from solver.calc import NLPSolver

if __name__ == '__main__':
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


