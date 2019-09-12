# fmbqm
This fmbqm package expands the application of annealers.

`fmbqm.FMBQM` class inherits from [`dimod.BQM`](https://docs.ocean.dwavesys.com/projects/dimod/en/latest/reference/bqm/binary_quadratic_model.html#dimod.BinaryQuadraticModel)
and its parameters are trained on data provided by users. So when considering a minimization task,
it does not expect the target function to have an equivalent representation as an Ising or QUBO model.

## Install
On the root of the project, run

```bash
$ python setup.py install
```

## Example

We try to minimize a function:

```python
def two_complement(x, scaling=True):
    '''
    Evaluation function for binary array
    of two's complement representation.

    example (when scaling=False):
    [0,0,0,1] => 1
    [0,0,1,0] => 2
    [0,1,0,0] => 4
    [1,0,0,0] => -8
    [1,1,1,1] => -1
    '''
    val, n = 0, len(x)
    for i in range(n):
        val += (1<<(n-i-1)) * x[i] * (1 if (i>0) else -1)
    return val * (2**(1-n) if scaling else 1)
```
This is an evaluator of two's complement representation, while its output is scaled to [-1,1].

We fix the input length to 16, and generate initial dataset of size 5 for training.

```python
import numpy as np

xs = np.random.randint(2, size=(5,16))
ys = np.array([two_complement(x) for x in xs])
```

Based on the dataset, train a FMBQM.

```python
import fmbqm
model = fmbqm.FMBQM.from_data(xs, ys)
```

We use simulated annealing from `dimod` package here, to solve the trained BQM.

```python
import dimod
sa_sampler = dimod.samplers.SimulatedAnnealingSampler()
```

We repeat taking 3 samples and updating the BQM for 15 times (total: 45 samples).

```python
for _ in range(15):
    res = sa_sampler.sample_qubo(model.to_qubo()[0], num_reads=3)
    xs = np.r_[xs, res.record['sample']]
    ys = np.r_[ys, [two_complement(x) for x in res.record['sample']]]
    model.train(xs, ys)
```

Then, the history of the sampling is plotted like this.

```python
import matplotlib.pyplot as plt
plt.plot(ys, 'o')
plt.xlabel('Selection ID')
plt.ylabel('value (scaled)')
plt.ylim([-1.0,1.0])
```
![image](https://user-images.githubusercontent.com/15908202/64800217-205ed100-d5c1-11e9-8d29-b2d13bcb0e53.png)

We can see that the sampler is trying to take near optimal binary array.
