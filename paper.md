---
title: 'QuantEcon'
tags:
  - Python
  - Economics
authors:
  - name: Quentin Batista
    affiliation: 5
  - name: Chase Coleman
    affiliation: 4
  - name: Oyamad Diasuke
    affiliation: 2
  - name: Yuya Furusawa
    affiliation: 6
  - name: Shu Hu
    affiliation: 1
  - name: Smit Lunagariya
    affiliation: 3
  - name: Spencer Lyon
    affiliation: 4
  - name: Matthew McKay
    affiliation: 1
  - name: Thomas J. Sargent
    affiliation: 4
  - name: Zejin Shi
    affiliation: 7
  - name: John Stachurski
    affiliation: 1
  - name: Natasha Watkins
    affiliation: 1
  - name: Humphrey Yang
    affiliation: 1
  - name: Hengcheng Zhang
    affiliation: 1
affiliations:
  - name: The Australian National University
    index: 1
  - name: University of Tokyo
    index: 2
  - name: Indian Institute of Technology (BHU), Varanasi
    index: 3
  - name: New York University
    index: 4
  - name: MIT
    index: 5
  - name: Crop.inc
    index: 6
  - name: University of Arizona
    index: 7
date: 13 December 2023
bibliography: paper.bib
nocite: |
  @*
---

# Summary

Economics has traditionally relied on tractable mathematical models, diagrams,
and simple regression methods to analyze and understand economic phenomena.
However, in recent decades economists have increasingly shifted towards more
computationally challenging problems, involving large numbers of heterogeneous
agents and complex nonlinear interactions.

[QuantEcon.py](https://github.com/quantecon/QuantEcon.py) is an open-source
software library that helps to support this shift
towards more computational intensive research in the field of economics.  First
released in 2014, [QuantEcon.py](https://github.com/quantecon/QuantEcon.py)
has been under continuous development for around
9 years. The library includes a wide range of functions for economic analysis,
including numerical methods, data visualization, estimation, and dynamic
programming. The library includes a number of fundamental algorithms used in
high performance computational economics. In this article we review the key
features of the the library.

# Statement of Need

Economists use a variety of economic, statistical and mathematical models as
building blocks for constructing larger and more fully-featured models. Some of
these are relatively unique to economics and finance. For example, many
macroeconomic and financial models include a stochastic volatility component,
since asset markets often exhibit bursts of volatility. Other building blocks
involve optimization routines, such as firms that maximize present value given
estimated time paths for profits and interest rates.  Firms modeled in this way
are then plugged into larger models that contain households, banks and other
economic agents.

[QuantEcon.py](https://github.com/quantecon/QuantEcon.py) focuses on
supplying building blocks for constructing economic
models that are fast, efficient and simple to modify.  This encourages code
re-use across the economics community, without enforcing particular model
structure through a top-down development process.


# Implementation Choices


In terms of software systems and architecture, [QuantEcon.py](https://github.com/quantecon/QuantEcon.py) is built on
top of standard libraries such as [NumPy](https://numpy.org) and [SciPy](https://scipy.org), while also heavily leveraging
[Numba](https://numba.pydata.org) for just-in-time (JIT) code acceleration, combined with automatic parallelization and
caching when possible.  ([Numba](https://numba.pydata.org) is a just-in-time (JIT) compiler for Python first
developed by Continuum Analytics that can generate optimized LLVM machine code at
run-time.)  JIT-based acceleration is essential to QuantEcon's strategy of
providing code for computational economics that is performant, portable and easy
to modify.

For installation and maintenance ease, QuantEcon maintainers restrict
contributions to depend on libraries available in [Anaconda](https://www.anaconda.com/).


# Status

[QuantEcon.py](https://github.com/quantecon/QuantEcon.py) is released under the
open-source MIT License and is partly
maintained and supported by QuantEcon, a NumFOCUS fiscally sponsored project
dedicated to development and documentation of modern open source computational
tools for economics, econometrics, and decision making.

[QuantEcon.py](https://github.com/quantecon/QuantEcon.py) is available through
the [Python Package Index](https://pypi.org/project/quantecon/):

```bash
pip install quantecon
```

or through `conda`:

```bash
conda install -c conda-forge quantecon
```

# Capabilities

This section gives a basic introduction of `quantecon` and it's usage. The
`quantecon` python library consists of the following major modules:

- Game Theory (`game_theory`)
- Markov Chains (`markov`)
- Optimization algorithms (`optimize`)
- Random generation utilities (`random`)

The library also has some other submodules containing utility functions and
miscellaneous tools such as implementations of kalman filters, tools for directed
graphs, algorithm for solving linear quadratic control, etc.


## Game Theory

The `game_theory` submodule provides efficient implementation of state-of-the-art
algorithms for computing Nash equilibria of normal form games,
the Lemke-Howson algorithm, the McLennan-Tourky algorithm and
several learning/evolutionary dynamics algorithms, such as
fictitious play (and its stochastic version),
best response dynamics (and its stochastic version),
local interaction dynamics, and logit response dynamics.

It can also compute all mixed Nash equilibria of a 2-player (non-degenerate)
normal form game by support enumeration and vertex enumeration respectively.

```python
>>> import quantecon as qe
>>> import numpy as np
```

```python
>>> bimatrix = [[(1, 1), (-1, 0)],
...             [(-1, 0), (1, 0)],
...             [(0, 0), (0, 0)]]
>>> g = qe.game_theory.NormalFormGame(bimatrix)
>>> qe.game_theory.support_enumeration(g)
[(array([1., 0., 0.]), array([1., 0.])), (array([0., 1., 0.]), array([0., 1.]))]
>>> qe.game_theory.vertex_enumeration(g)
[(array([1., 0., 0.]), array([1., 0.]))]
```

The following snippet computes mixed Nash equilibria of a
2-player normal form game by the Lemke-Howson algorithm.

```python
>>> bimatrix = [[(3, 3), (3, 2)],
...             [(2, 2), (5, 6)],
...             [(0, 3), (6, 1)]]
>>> g = qe.game_theory.NormalFormGame(bimatrix)
>>> qe.game_theory.lemke_howson(g, init_pivot=1)
(array([0.        , 0.33333333, 0.66666667]), array([0.33333333, 0.66666667]))
```

Similarly, it can also compute mixed Nash equilibria of an
N-player normal form game by applying the imitation
game algorithm by McLennan and Tourky to the best response correspondence.

```python
>>> N = 3
>>> v = 2
>>> payoff_array = np.empty((2,)*N)
>>> payoff_array[0, :] = 1
>>> payoff_array[1, :] = 0
>>> payoff_array[1].flat[0] = v
>>> g = qe.game_theory.NormalFormGame((qe.game_theory.Player(payoff_array), ) * N)
>>> res = qe.game_theory.mclennan_tourky(g, epsilon=1e-5)
>>> res[0]
array([0.70710754, 0.29289246])
>>> res[1]
array([0.70710754, 0.29289246])
>>> res[2]
array([0.70710754, 0.29289246])
```

## Markov Chains

The `quantecon.markov` module deals with the computations related to markov chains.

This module contains a class `MarkovChain` which represents finite-state discrete-time Markov chains.

```python
>>> P = np.array([
...     [0.1, 0.9],
...     [0.0, 1.0]
... ])
>>> mc = qe.MarkovChain(P)
```

The `MarkovChain` object provides access to useful information such as:

- List of stationary distributions
```python
>>> mc.stationary_distributions
array([[0., 1.]])
```

- Indication of whether the Markov chain is irreducible
```python
>>> mc.is_irreducible
False
```

- Indication of whether the Markov chain is aperiodic
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

The `MarkovChain` object is also capable of computing many different features
like the number of the recurrent classes, period of the Markov chain, list of
cyclic classes, etc.

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

- `rouwenhorst`: estimates a stationary AR(1) process. The Rouwenhorst approximation
  uses the recursive defintion for approximating a distribution.

$$
y_t = \mu + \rho y_{t-1} + \varepsilon_t
$$

```python
>>> rhorst_mc = qe.markov.rouwenhorst(n=4, rho=0.5, sigma=0.5, mu=0.)
>>> rhorst_mc.state_values
array([-1.        , -0.33333333,  0.33333333,  1.        ])
```

The `quantecon.markov` module can also be used for solving discrete dynamic
programs (also known as Markov decision processes) with finite states and actions.

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

Similary, we can also solve using using *modified policy iteration*
and *linear programming* by changing the *method* name in `ddp.solve`.

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
>>> res = qe.optimize.linprog_simplex(c, A_ub=A_ub, b_ub=b_ub)
>>> res.x, res.fun, res.success
(array([    0.,     0., 10000.]), 10000.0, True)
```

### Scalar Maximization

The `optimize` module implements the Nelder-Mead algorithm for maximizing a scalar-valued function with one or more variables.

```python
>>> from numba import njit
>>> @njit
... def rosenbrock(x):
...     return -(100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0])**2)
...
>>> x0 = np.array([-2, 1])
>>> res = qe.optimize.nelder_mead(rosenbrock, x0)
>>> res.x, res.fun, res.success
(array([0.99999814, 0.99999756]), -1.6936258239463265e-10, True)
```

There's also the scalar maximization function - `brentq_max` which
maximizes the function within the given bounded intervals and
returns maximizer value, maximum value attained and some additional
information related to convergence and number of iterations.

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

## Miscellaneous tools

The library also contains some other tools that help in tackling problems
like linear quadratic optimal control, analyzing dynamic linear economies,
discrete lyapunov equation, etc. The brief overview of some of these routines
is given below:

### Matrix equations

The function `solve_discrete_lyapunov` helps to compute the solution of
the discrete lyapunov equation given by:

$$
AXA' - X + B = 0
$$

```python
>>> A = np.ones((2, 2)) * .5
>>> B = np.array([[.5, -.5], [-.5, .5]])
>>> qe.solve_discrete_lyapunov(A, B)
array([[ 0.5, -0.5],
       [-0.5,  0.5]])
```

Similarly, the function `solve_discrete_riccati` computes the solution of
the discrete-time algebraic Riccati equation:

$$
X = A'XA - (N + B'XA)'(B'XB + R)^{-1}(N + B'XA) + Q
$$

### LQ Control

The library has a class `LQ` for analyzing linear quadratic optimal
control problems of either the infinite horizon form or the finite horizon form.

```python
>>> Q = np.array([[0., 0.], [0., 1]])
>>> R = np.array([[1., 0.], [0., 0]])
>>> RF = np.eye(2) * 100
>>> A = np.ones((2, 2)) * .95
>>> B = np.ones((2, 2)) * -1
>>> beta = .95
>>> T = 1
>>> lq_mat = qe.LQ(Q, R, A, B, beta=beta, T=T, Rf=RF)
>>> lq_mat
Linear Quadratic control system
  - beta (discount parameter)       : 0.95
  - T (time horizon)                : 1
  - n (number of state variables)   : 2
  - k (number of control variables) : 2
  - j (number of shocks)            : 1
```

### Graph Tools

The library contains a class `DiGraph` to represent directed graphs
and provides information about the graph structure such as strong
connectivity, speriodicity, cyclic components, etc.

```python
>>> adj_matrix = [[1, 0, 1], [1, 0, 1], [1, 1, 1]]
>>> node_labels = np.array(['a', 'b', 'c'])
>>> g = qe.DiGraph(adj_matrix, node_labels=node_labels)
>>> g
Directed Graph:
  - n(number of nodes): 3
```

- Check if the graph is strongly connected

```python
>>> g.is_strongly_connected
True
```

- Find the period of the graph

```python
>>> g.period
1
```

- Find the cyclic components

```python
>>> g.cyclic_components
[array(['a', 'b', 'c'], dtype='<U1')]
```


# Future Work

QuantEcon aims to extend its current implementation to other backend
libraries like JAX or other GPU providing libraries to utilize the
modern computing systems and provide faster execution speeds.

The [documentation is available on readthedocs](https://quanteconpy.readthedocs.io/en/latest/)

# References
