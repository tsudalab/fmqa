"""
Trainable Binary Quadratic Model based on Factorization Machine (FMBQM)
"""

import numpy as np
import mxnet as mx
from mxnet import nd
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.vartypes import Vartype, as_vartype
from dimod import BQM
from itertools import combinations

from .factorization_machine import FactorizationMachine, SparseFactorizationMachine

__all__ = [
    "mergeBQM",
    "BitConstraintBQM",
    "FactorizationMachineBinaryQuadraticModel",
    "FMBQM",
]

def mergeBQM(bqms, scales=None):
    """
    Merge all BQMs contained in a list.
    """
    if scales == None:
        scales = np.ones(len(bqms))
    if bqms[0].vartype == Vartype.SPIN:
        h, J, o = {}, {}, 0.0
        for bqm, scale in zip(bqms, scales):
            _h, _J, _o = bqm.to_ising()
            for k in _h.keys():
                h[k] = h.get(k, 0.0) + _h[k] * scale
            for k in _J.keys():
                J[k] = J.get(k, 0.0) + _J[k] * scale
            o += _o * scale
        return BinaryQuadraticModel.from_ising(h, J, o)
    elif bqms[0].vartype == Vartype.BINARY:
        Q, o = {}, 0.0
        for bqm, scale in zip(bqms, scales):
            _Q, _o = bqm.to_qubo()
            for k in _Q.keys():
                Q[k] = Q.get(k, 0.0) + _Q[k] * scale
            o += _o * scale
        return BinaryQuadraticModel.from_qubo(Q, o)

class BitConstraintBQM(BinaryQuadraticModel):
    '''

    Examples:
    Here demonstrates applying a 2-hot constraint on a BQM.
    >>> import numpy as np
    >>> import neal
    >>> import dimod
    >>> import fmqa
    >>> qubo = dimod.BQM.from_qubo(np.array([
        [1, -1,  0,  1],
        [0, -1,  0, -1],
        [0,  0, -1,  0], 
        [0,  0,  0, -1]
    ]))
    >>> sa = neal.SimulatedAnnealingSampler()
    >>> sa.sample(qubo, num_reads=1)
    SampleSet(rec.array([([0, 1, 1, 1], -4., 1)], ..., 'BINARY')
    >>> constraint = fmqa.BitConstraintBQM.n_hot([0,1,2,3], 2)
    >>> constraint_qubo = fmqa.mergeBQM([qubo, constraint])
    >>> sa.sample(constraint_qubo, num_reads=1)
    SampleSet(rec.array([([0, 1, 0, 1], -5., 1)], ..., 'BINARY')

    '''

    def __init__(self, *args, **kwargs):
        super(BitConstraintBQM, self).__init__(*args, **kwargs)

    @classmethod
    def n_hot(cls, region, n, vartype="SPIN"):
        '''
        Generate a BQM instance encoding n-hot constraint (out of bits listed in region).
        '''
        N = len(region)
        vartype = as_vartype(vartype)
        assert vartype in [Vartype.SPIN, Vartype.BINARY]
        assert n >= 0
        assert N >= n
        if vartype == Vartype.SPIN:
            r = 2/(N-2*n) if abs(N-2*n) > 2 else 1.0
            h = {i: N-2*n * r for i in region}
            J = {k: 1.0 * r for k in map(tuple, map(sorted, combinations(region, 2)))}
            return cls.from_ising(h, J)
        elif vartype == Vartype.BINARY:
            r = min(1.0/abs(n-0.5), 1.0)
            Q = {(i, i): (0.5-n) * r for i in region}
            for k in map(tuple, map(sorted, combinations(region, 2))):
                Q[k] = 1.0 * r
            return cls.from_qubo(Q)

    @classmethod
    def one_hot(cls, region, vartype="SPIN"):
        return cls.n_hot(region, 1, vartype)

class FactorizationMachineBinaryQuadraticModel(BinaryQuadraticModel):
    """FMBQM: Trainable BQM based on Factorization Machine

    Args:
        input_size (int):
            The dimension of input vector.
        vartype (dimod.vartypes.Vartype):
            The type of input vector.
        act (string, optional):
            Name of activation function applied on FM output: "identity", "sigmoid", or "tanh".
    """

    def __init__(self, input_size, vartype, act="identity", nodelist=[], edgelist=[], ctx=mx.cpu(), **kwargs):
        # Initialization of BQM
        init_linear = {i: 0.0  for i in range(input_size)}
        init_quadratic  = {}
        init_offset = 0.0
        self.ctx = ctx
        super().__init__(init_linear,  init_quadratic, init_offset, vartype, **kwargs)
        if len(edgelist) == 0:
            self.fm = FactorizationMachine(input_size, act=act, **kwargs)
        else:
            self.fm = SparseFactorizationMachine(input_size, act=act, nodelist=nodelist, edgelist=edgelist, **kwargs)

    def to_qubo(self):
        return self._fm_to_qubo()

    def to_ising(self):
        return self._fm_to_ising()

    @classmethod
    def from_data(cls, x, y, act="identity", num_epoch=1000, learning_rate=1.0e-2, schedule=None, gpu=False, factorization_size=None, **kwargs):
        """Create a binary quadratic model by FM regression model trained on the provided data.

        Args:
            x (ndarray, int):
                Input vectors of SPIN/BINARY.
            y (ndarray, float):
                Target values.
            act (string, optional):
                Name of activation function applied on FM output: "identity", "sigmoid", or "tanh".
            num_epoch (int, optional):
                The number of epoch for training FM model.
            learning_rate (int, optional):
                Learning rate for FM's optimizer.
            **kwargs:

        Returns:
            :class:`.FactorizationMachineBinaryQuadraticModel`
        """
        if np.all((x == 0) | (x == 1)):
            vartype = Vartype.BINARY
        elif np.all((x == -1) | (x == 1)):
            vartype = Vartype.SPIN
        else:
            raise ValueError("input data should BINARY or SPIN vectors")

        input_size = x.shape[-1]
        ctx = mx.gpu() if gpu else mx.cpu()
        if factorization_size == None:
            if x.shape[0] > 10:
                cv_fold = 3
                blocks = np.random.permutation(x.shape[0])[:x.shape[0]//cv_fold*cv_fold].reshape((3,-1))
                sizes = np.logspace(np.log10(3),np.log10(x.shape[1]),5,dtype=int)
                cverrors = np.zeros(len(sizes))
                for n in range(len(sizes)):
                    _fmbqm = cls(input_size, vartype, act, factorization_size=sizes[n], ctx=ctx, **kwargs)
                    for i in range(cv_fold):
                        train_idx = np.concatenate([blocks[:i], blocks[i+1:]]).flatten()
                        test_idx = blocks[i]
                        _fmbqm.train(x[train_idx], y[train_idx], num_epoch, learning_rate, gpu=gpu, schedule=schedule, init=True)
                        cverrors[n] += np.sum((_fmbqm.predict(x[test_idx]) - y[test_idx]) ** 2) ** .5
                factorization_size = sizes[np.argmin(cverrors)]
            else:
                factorization_size = 8

        fmbqm = cls(input_size, vartype, act, factorization_size=factorization_size, ctx=ctx, **kwargs)
        fmbqm.train(x, y, num_epoch, learning_rate, gpu=gpu, schedule=schedule, init=True)
        return fmbqm

    def train(self, x, y, num_epoch=1000, learning_rate=1.0e-2, gpu=False, schedule=None, init=False):
        """Train FM regression model on the provided data.

        Args:
            x (ndarray, int):
                Input vectors of SPIN/BINARY.
            y (ndarray, float):
                Target values.
            num_epoch (int, optional):
                The number of epoch for training FM model.
            learning_rate (int, optional):
                Learning rate for FM's optimizer.
            init (bool, optional):
                Initialize or not before training.
        """
        ctx = mx.gpu() if gpu else mx.cpu()
        if init:
            self.fm.init_params(ctx=ctx)
        self._check_vartype(x)
        x, y = nd.array(x, ctx=ctx), nd.array(y, ctx=ctx)
        if self.vartype == Vartype.SPIN:
            self.fm.train(x, y, num_epoch, learning_rate, schedule=schedule)
            h, J, b = self._fm_to_ising()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = h.get(i, 0)
                for j in range(i+1, self.fm.input_size):
                    self.quadratic[(i,j)] = J.get((i,j), 0)
        elif self.vartype == Vartype.BINARY:
            self.fm.train(2*x-1, y, num_epoch, learning_rate, schedule=schedule)
            Q, b = self._fm_to_qubo()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = Q.get((i,i), 0)
                for j in range(i+1, self.fm.input_size):
                    self.quadratic[(i,j)] = Q.get((i,j), 0)

    def predict(self, x):
        """Predict target value by trained model.

        Args:
            x (ndarray, int):
                Input vectors of SPIN/BINARY.

        Returns:
            :obj:`numppy.ndarray`: Predicted values.
        """
        self._check_vartype(x)
        x = nd.array(x, ctx=self.ctx)
        if len(x.shape) == 1:
            x = nd.expand_dims(x, axis=0)
        if self.vartype == Vartype.SPIN:
            return self.fm(x).asnumpy()
        elif self.vartype == Vartype.BINARY:
            return self.fm(2*x-1).asnumpy()

    def _check_vartype(self, x):
        if (self.vartype is Vartype.BINARY) and np.all((1 == x) | (0 == x)) or \
           (self.vartype is Vartype.SPIN) and np.all((1 == x) | (-1 == x)):
            return
        raise ValueError("input data should be of type", self.vartype)

    def _fm_to_ising(self, scaling=True):
        """Convert trained model into Ising parameters.

        Args:
            scaling (bool, optional):
                Flag for automatic scaling.

        """
        b, h, J = self.fm.get_bhQ()
        if isinstance(h, dict):
            if scaling:
                values = list(h.values()) + list(J.values())
                scaling_factor = np.max(np.abs(values))
                b /= scaling_factor
                for k in h.keys():
                    h[k] /= scaling_factor
                for k in J.keys():
                    J[k] /= scaling_factor
            return h, J, b
        elif scaling:
            scaling_factor = max(np.max(np.abs(h)), np.max(np.abs(J)))
            b /= scaling_factor
            h /= scaling_factor
            J /= scaling_factor
        return {key: h[key] for key in range(len(h))}, {key: J[key] for key in zip(*J.nonzero())}, b

    def _fm_to_qubo(self, scaling=True):
        """Convert trained model into QUBO parameters.

        Args:
            scaling (bool, optional):
                Flag for automatic scaling.
        """
        b, h, Q = self.fm.get_bhQ()
        if isinstance(h, dict):
            b = b - np.sum(list(h.values())) + np.sum(list(Q.values()))
            h = {k: 2 * (h[k] - sum([Q[p] for p in Q.keys() if k in p])) for k in h.keys()}
            Q = {k: 4 * Q[k] for k in Q.keys() if Q[k] != 0}
            for k in h.keys():
                Q[k, k] = h[k]
            if scaling:
                scaling_factor = np.max(np.abs(list(Q.values())))
                b /= scaling_factor
                for k in Q.keys():
                    Q[k] /= scaling_factor
            return Q, b
        b = b - np.sum(h) + np.sum(Q)
        h = 2 * (h - np.sum(Q, axis=0) - np.sum(Q, axis=1))
        Q = 4 * Q
        Q[np.diag_indices(len(Q))] = h
        if scaling:
            scaling_factor = np.max(np.abs(Q))
            b /= scaling_factor
            Q /= scaling_factor
        # Conversion from full matrix to dict
        Q_dict = {key: Q[key] for key in zip(*Q.nonzero())}
        for i in range(len(Q)):
            Q_dict[(i,i)] = Q[i,i]

        return Q_dict, b

FMBQM = FactorizationMachineBinaryQuadraticModel

