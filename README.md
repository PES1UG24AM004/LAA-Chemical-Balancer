# Chemical Equation Balancer: A Linear Algebra Approach

A graphical Python application that leverages core linear algebra concepts—specifically Null Space and Least Squares Approximation—to balance chemical equations and recover stoichiometric ratios from noisy sensor data.

This project was developed as a miniproject for the Linear Algebra and its Applications (LAA) coursework.

## The Mathematics

This project is divided into two primary modules, demonstrating different applications of linear algebra in computational chemistry.

### Act 1: Exact Balancing via Null Space

In a perfectly balanced chemical equation, the Law of Conservation of Mass dictates that the number of atoms for each element must be equal on both the reactant and product sides. We can represent this as a homogeneous system of linear equations:

Ax = 0

- A: The atom count matrix. Rows represent individual elements, and columns represent molecules. Reactant counts are positive, and product counts are negative.
- x: The vector of unknown stoichiometric coefficients required to balance the equation.

By computing the null space of matrix A (utilizing `scipy.linalg.null_space`), the application identifies the basis vector x that perfectly satisfies the homogeneous system. This floating-point vector is then algorithmically scaled into minimal whole integers to conform to standard chemical notation.

### Act 2: Real-World Sensor Data via Least Squares

In physical chemistry experiments, sensor readings for molecule concentrations are inherently noisy. Multiple sensor readings generate an overdetermined system with no exact solution:

Ax ≈ b

- A: A matrix of noisy sensor readings for each molecule across multiple experimental trials.
- b: The total sum of the readings for each trial.

Because an exact intersection does not exist, the application finds the vector x that minimizes the squared error ||Ax - b||^2. The algorithm computes the Least Squares solution by determining the normal equations:

x = (A^T A)^-1 A^T b

This approach allows the system to successfully recover the true underlying stoichiometric ratios despite randomized measurement noise.

## Technical Stack & Libraries

- NumPy: Utilized for matrix manipulations, dynamic array handling, and executing the least squares approximation (`numpy.linalg.lstsq`).
- SciPy: Utilized for extracting the exact basis vectors of the homogeneous systems (`scipy.linalg.null_space`).
- Tkinter: The standard GUI library for Python, used to construct the interactive graphical interface.

## Team Members

- Aarav Yuval (PES1UG24AM004)
- A Ravi Teja (PES1UG24AM001)
- Ahan A Mysore (PES1UG24AM019)
