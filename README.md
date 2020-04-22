# Test Functions for Optimisation in n Dimensions
This is a small test suite for optimisation algorithms that work on n-dimensional problems. I have developing machine learning assisted design of experiments for setting optimisation on industrial manufacturing equipment. To test the algorithms and gauge the limits of the approach, I wanted to apply them to simulated problems.<br>

## Function Properties
The functions included in this suite generally satisfy the following properties.<br>
The function f, maps an n-dimensional hypercybe to a real number

f: x e [a,b]^n -> y e R

The function is continuous in the hypercube, it is difficult to approximate, and possibly have multiple treacherous local extrema. The function may have several global extrema, but assume minimisation problems and will not have global minima on the edges of the hypercube. 

## Sampling for Simulation
In general terms, the problems assume a function that maps a normalised range of n number of parameters to a real number, the target variable. So, a model of the system tries to approximate a function f

f: x e [0,1]^n -> y e R

And our eventual target is to find the x that minimises y.<br>
An added challenge when sampling for the real system is noise. Sampling the target variable at the same settings multiple times will yield different values. We can, however, reasonably assume gaussian noise, such that the average of multiple samples will tend towards the true value. For problems that sample computer models (e.g. a finite element model) this is not an issue and the same set of parameters will always yield the same output value. This aspect has been implemented as gaussian noise that can be switched on and off depeinding on the problem at hand.
# How to Use the Test Suite
```python
import numpy as np
from opt_test_functions import OptTestFunction
myTestFunction = OptTestFunction(function="Michalewicz", 
                                 noise=True, 
                                 noise_stddev=1, 
                                 xmin=0, 
                                 xmax=1)
x = np.random.random((6,30)) # 6 dimensions, 30 examples
y = myTestFunction.get(x)
```
# Implemented Functions
## Rastrigin
## Michalewicz
## Zakharov
## Styblinski-Tang