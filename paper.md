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
# References
