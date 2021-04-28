# fmqa
The fmqa package provides a trainable binary quadratic model `FMBQM`.
In combination with annealing solvers, it enables optimization of
a black-box function in a data-driven way.
This could expand the application of annealing solvers.

A common way of solving a combinatorial optimization problem is to encode
the objective function into a binary quadratic model (BQM),
where the user has to set parameters of the BQM beforehand.
However, our `fmqa.FMBQM` class can automatically learn the parameters based
on a dataset provided by users.
This is an ideal approach when the user can evaluate the objective function
on any given input, but has no knowledge about the analytical form of it.

The `FMBQM` class inherits from [`dimod.BQM`](https://docs.ocean.dwavesys.com/projects/dimod/en/latest/reference/bqm/binary_quadratic_model.html#dimod.BinaryQuadraticModel)
of D-Wave Ocean Tools, so the basic usage of `FMBQM` has many in common with that of `BQM`.
For the functions specific to `FMBQM`, such as how to train the model,
please refer to the example code below.

## Install
On the root of the project, run

```bash
$ python setup.py install
```

## Example
For an example use of the package, we try to minimize this function:

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

Based on the dataset, train a FMBQM model.

```python
import fmqa
model = fmqa.FMBQM.from_data(xs, ys)
```

We use simulated annealing from `dimod` package here to solve the trained model.

```python
import dimod
sa_sampler = dimod.samplers.SimulatedAnnealingSampler()
```

We repeat taking 3 samples at once and updating the model for 15 times
(45 samples taken in total).

```python
for _ in range(15):
    res = sa_sampler.sample(model, num_reads=3)
    xs = np.r_[xs, res.record['sample']]
    ys = np.r_[ys, [two_complement(x) for x in res.record['sample']]]
    model.train(xs, ys)
```

Then, the history of the sampling looks like this.

```python
import matplotlib.pyplot as plt
plt.plot(ys, 'o')
plt.xlabel('Selection ID')
plt.ylabel('value (scaled)')
plt.ylim([-1.0,1.0])
plt.show()
```
![image](https://user-images.githubusercontent.com/15908202/64800217-205ed100-d5c1-11e9-8d29-b2d13bcb0e53.png)

We can see that the sampling go down to near optimal as the dataset grows.

## License

The fmqa package is licensed under the MIT "Expat" License.

## Citation

If you use this package in your work, please cite:

```
@article{PhysRevResearch.2.013319,
  title = {Designing metamaterials with quantum annealing and factorization machines},
  author = {Kitai, Koki and Guo, Jiang and Ju, Shenghong and Tanaka, Shu and Tsuda, Koji and Shiomi, Junichiro and Tamura, Ryo},
  journal = {Phys. Rev. Research},
  volume = {2},
  issue = {1},
  pages = {013319},
  numpages = {10},
  year = {2020},
  month = {Mar},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevResearch.2.013319},
  url = {https://link.aps.org/doi/10.1103/PhysRevResearch.2.013319}
}

```

