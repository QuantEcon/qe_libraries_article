---
title: 'QuantEcon'
tags:
  - Python
  - Julia
  - economics
authors:
  - name: ABC
    affiliation: 1
  - name: DEF
    affiliation: 2
  - name: XYZ
    affiliation: 3
affiliations:
  - name: University1
    index: 1
  - name: University2
    index: 2
  - name: University3
    index: 3
date: 13 December 2023
bibliography: paper.bib
---

# Summary

QuantEcon is a NumFOCUS fiscally sponsored project dedicated to development and documentation of modern open source computational tools for economics, econometrics, and decision making.

# QuantEcon Capabilities

This section gives a basic introduction of QuantEcon and it's usage. The `quantecon` python library consists of the following major modules:

- Game Theory (`game_theory`)
- Markov Chains (`markov`)
- Optimization algorithms (`optimize`)
- Random generation utilities (`random`)

The library also has some other submodules containing utility functions, implementation of kalman filters, tools for directed graphs, algorithm for solving linear quadratic control, etc.

## Markov Chains

The `quantecon.markov` module deals with the computations related to the markov chains.

This module contains a class `MarkovChain` which represents finite-state discrete-time Markov chain.

```python
>>> P = np.array([
...     [0.1, 0.9],
...     [0.0, 1.0]
... ])
>>> mc = qe.MarkovChain(P)
```

The `MarkovChain` object consists of many useful information like:

- List of stationary distributions

```python
>>> mc.stationary_distributions
array([[0., 1.]])
```

- Indicates whether the Markov chain is irreducible
```python
>>> mc.is_irreducible
False
```

- Indicates whether the Markov chain is aperiodic
```python
>>> mc.is_aperiodic
True
```

- List of communication classes
```python
>>> mc.communication_classes
[array([1]), array([0])]
```

- Simulate time series of state transitions
```python
>>> mc.simulate_indices(5)
array([0, 1, 1, 1, 1])
```

The `MarkovChain` object is also capable of computing many different features like the number of the recurrent classes, period of the Markov chain, list of cyclic classes, etc.

It is also possible to construct a `MarkovChain` object using approximation:

- `tauchen`: discretized version of the linear Gaussian AR(1) process using Tauchen's method.

$$
  y_t = \mu + \rho y_{t-1} + \epsilon_t
$$

```python
>>> tauchen_mc = qe.markov.tauchen(n=4, rho=0.5, sigma=0.5, mu=0., n_std=3)
>>> tauchen_mc.state_values
array([-1.73205081, -0.57735027,  0.57735027,  1.73205081])
```

- `rouwenhorst`: estimates a stationary AR(1) process. The Rouwenhorst approximation uses the recursive defintion for approximating a distribution.


$$
y_t = \mu + \rho y_{t-1} + \varepsilon_t
$$

```python
>>> rhorst_mc = qe.markov.rouwenhorst(n=4, rho=0.5, sigma=0.5, mu=0.)
>>> rhorst_mc.state_values
array([-1.        , -0.33333333,  0.33333333,  1.        ])
```

The `quantecon.markov` module can also be used for solving discrete dynamic programs (also known as Markov decision processes) with finite states and actions.

The `DiscreteDP` class currently implements the following solution algorithms:

- value iteration
- policy iteration
- modified policy iteration
- linear programming

```python
>>> s_indices = [0, 0, 1]  # State indices
>>> a_indices = [0, 1, 0]  # Action indices
>>> R = [5, 10, -1]  # Rewards
>>> Q = [(0.5, 0.5), (0, 1), (0, 1)]  # Transition probability
>>> beta = 0.95  # Discount factor
>>> ddp = qe.markov.DiscreteDP(R, Q, beta, s_indices, a_indices)
```

- Solve using *value iteration* method

```python
>>> res = ddp.solve(method='value_iteration', v_init=[0, 0], epsilon=0.01)
>>> res.sigma  # (Approximate) optimal policy function
array([0, 0])
>>> res.v  # (Approximate) optimal value function
array([ -8.5665053 , -19.99507673])
```

- Solve using *policy iteration* method

```python
>>> res = ddp.solve(method='policy_iteration', v_init=[0, 0])
>>> res.sigma  # Optimal policy function
array([0, 0])
>>> res.v  # Optimal value function
array([ -8.57142857, -20.        ])
```

Similary, we can also solve using the other two methods -  *modified policy iteration* and *linear programming* by changing the *method* name in `ddp.solve`.

## Optimize

The `optimize` module provides various routines to tackle the optimization problems.

### Linear Programming Solver

This module contains a linear programming solver based on the simplex
method - `linprog_simplex`, which helps to solve the following optimization problem.

$$
    \begin{aligned}
    \min_{x} \ & c^T x \\
    \mbox{subject to } \ & A_{ub} x \leq b_{ub}, \\
    & A_{eq} x = b_{eq}, \\
    & l \leq x \leq u \\
    \end{aligned}
$$

The following snippet solves the [Klee-Minty ](https://www.math.ubc.ca/~israel/m340/kleemin3.pdf) problem.

```python
>>> c = [100, 10, 1]
>>> A_ub = [[1, 0, 0],
...         [20, 1, 0],
...         [200, 20, 1]]
>>> b_ub = [1, 100, 10000]
>>> c, A_ub, b_ub = map(np.asarray, [c, A_ub, b_ub])
>>> qe.optimize.linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
SimplexResult(x=array([    0.,     0., 10000.]), lambd=array([0., 0., 1.]), fun=10000.0, success=True, status=0, num_iter=9)
```

### Scalar Maximization

The `optimize` module implements the Nelder-Mead algorithm for maximizing a scalar-valued function with one or more variables.

```python
>>> @njit
... def rosenbrock(x):
...     return -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0])**2)
...
>>> x0 = np.array([-2, 1])
>>> qe.optimize.nelder_mead(rosenbrock, x0)
results(x=array([0.99999814, 0.99999756]), fun=-1.6936258239463265e-10, success=True, nit=110, final_simplex=array([[0.99998652, 0.9999727 ],
       [1.00000218, 1.00000301],
       [0.99999814, 0.99999756]]))
```

There's also another scalar maximization function - `brentq_max` which
maximizes the function within the given bounded intervals and
returns maximizer value, maximum value attained and some extra information related to convergence and number of iterations.

```python
>>> @njit
... def f(x):
...     return -(x + 2.0)**2 + 1.0
...
>>> qe.optimize.brent_max(f, -3, 2) # x, max_value_of_f, extra_info
(-2.0, 1.0, (0, 6))
```

### Root Finding

It comprises of all the routines that finds the root of the given function.
Presently, quantecon has the following implementations:

- bisect
- brentq
- newton
- newton_halley
- newton_secant

The following snippet uses `brentq` to find the root of the function $f$
in the interval $(-1, 2)$.

```python
>>> @njit
... def f(x):
...     return np.sin(4 * (x - 1/4)) + x + x**20 - 1
...
>>> qe.optimize.brentq(f, -1, 2)
results(root=0.40829350427935973, function_calls=12, iterations=11, converged=True)
```

# References
