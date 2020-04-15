"""
Trainable Binary Quadratic Model based on Factorization Machine (FMBQM)
"""

import numpy as np
from mxnet import nd
from dimod.binary_quadratic_model import BinaryQuadraticModel
from dimod.vartypes import Vartype, as_vartype
from dimod import BQM
from itertools import combinations

from .factorization_machine import FactorizationMachine

__all__ = [
    "mergeBQM",
    "BitConstraintBQM",
    "FactorizationMachineBinaryQuadraticModel",
    "FMBQM",
]

def mergeBQM(bqms, scales=None):
    """
    Merges all BQMs contained in a list.
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
    Here demonstrates applying several 2-hot constraint on FMBQM
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

    def __init__(self, input_size, vartype, act="identity", **kwargs):
        # Initialization of BQM
        init_linear = {i: 0.0  for i in range(input_size)}
        init_quadratic  = {}
        init_offset = 0.0
        super().__init__(init_linear,  init_quadratic, init_offset, vartype, **kwargs)
        self.fm = FactorizationMachine(input_size, act=act, **kwargs)
        self.constraints = []

    def to_qubo(self):
        return self._fm_to_qubo()

    def to_ising(self):
        return self._fm_to_ising()

    @classmethod
    def from_data(cls, x, y, act="identity", num_epoch=1000, learning_rate=1.0e-2, **kwargs):
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
        fmbqm = cls(input_size, vartype, act, **kwargs)
        fmbqm.train(x, y, num_epoch, learning_rate, init=True)
        return fmbqm

    def train(self, x, y, num_epoch=1000, learning_rate=1.0e-2, init=False):
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
        if init:
            self.fm.init_params()
        self._check_vartype(x)
        self.fm.train(x, y, num_epoch, learning_rate)
        if self.vartype == Vartype.SPIN:
            h, J, b = self._fm_to_ising()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = h[i]
                for j in range(i+1, self.fm.input_size):
                    self.quadratic[(i,j)] = J.get((i,j), 0)
        elif self.vartype == Vartype.BINARY:
            Q, b = self._fm_to_qubo()
            self.offset = b
            for i in range(self.fm.input_size):
                self.linear[i] = Q[(i,i)]
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
        x = nd.array(x)
        if len(x.shape) == 1:
            x = nd.expand_dims(x, axis=0)
        return self.fm(x).asnumpy()

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
        if self.vartype is Vartype.BINARY:
            b = b + np.sum(h)/2 + np.sum(J)/4
            h = (2*h + np.sum(J, axis=0) + np.sum(J, axis=1))/4.0
            J = J/4.0
        if scaling:
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
        if self.vartype is Vartype.SPIN:
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

