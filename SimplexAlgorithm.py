import numpy as np
from scipy.optimize import linprog
import time

def run_simplex(n, m):
    # Objective function: minimize sum(x1 + x2 + ... + xn)
    c = [1] * n

    # Start with the original known constraints for n = 4, m = 2
    if n == 4 and m == 2:
        A = [[1, 2, -1, -1], [-1, -2, 1, 1], [-1, -5, 2, 3], [1, 5, -2, -3]]
        b = [1, -1, 1, -1]
    else:
        # Generate a known feasible solution
        feasible_x = np.random.uniform(0.5, 2.0, size=n)

        # Generate random A matrix
        A = np.random.uniform(-2, 2, size=(m, n))

        # Generate b to ensure feasibility based on the known solution
        b = np.dot(A, feasible_x)

        # Perturb b to ensure it's less than or equal to the feasible values for inequality constraints
        b += np.random.uniform(0.1, 0.5, size=m)

    # Solve the problem with inequality constraints (A_ub and b_ub)
    start_time = time.time()
    result = linprog(c, A_ub=A, b_ub=b, bounds=(0, None), method='highs')
    end_time = time.time()

    # Display results
    if result.success:
        print(f"For n = {n}, m = {m}:")
        print(f"Optimal Objective Value: {result.fun}")
        print(f"Optimal Solution: {result.x}")
    else:
        print(f"For n = {n}, m = {m}: No feasible solution found.")

    print(f"Time Taken: {end_time - start_time} seconds\n")

# Test for various values of n and m
n_values = [4, 10, 20, 30, 40, 50]
m_values = [2, 6, 10, 14]

# Run the simplex for all combinations of n and m
for n in n_values:
    for m in m_values:
        run_simplex(n, m)