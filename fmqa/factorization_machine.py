"""
Factorization Machine implemented with MXNet Gluon
"""

__all__ = [
    "FactorizationMachineBinaryQuadraticModel", "FMBQM"
]

import numpy as np
import mxnet as mx
from   mxnet import nd
from   mxnet import gluon

def triu_mask(input_size, F=nd):
    """Generate a square matrix with its upper trianguler elements being 1 and others 0.
    """
    mask = F.expand_dims(F.arange(input_size), axis=0)
    return (F.transpose(mask) < mask) * 1.0

def VtoQ(V, F=nd):
    """Calculate interaction strength by inner product of feature vectors.
    """
    input_size = V.shape[1]
    Q = F.dot(F.transpose(V), V) # (d,d)
    return Q * triu_mask(input_size, F)

class QuadraticLayer(gluon.nn.HybridBlock):
    """A neural network layer which applies quadratic function on the input.

    This class defines train() method for easy use.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_params(self, initializer=mx.init.Normal()):
        """Initialize all parameters

        Args:
            initializer(mx.init.Initializer):
                MXNet initializer object. [Default=mxnet.init.Normal()]
        """
        self.initialize(initializer, force_reinit=True)

    def train(self, x, y, num_epoch=100, learning_rate=1.0e-2):
        """Training of the regression model using Adam optimizer.
        """
        x, y = nd.array(x), nd.array(y)
        batchsize = x.shape[0]
        if None == self.trainer:
            self.trainer = gluon.Trainer(self.collect_params(), "adam", {"learning_rate": learning_rate})
        else:
            self.trainer.set_learning_rate(learning_rate)
        for epoch in range(num_epoch):
            with mx.autograd.record():
                output = self(x)
                loss = nd.mean((y - output)**2)
            loss.backward()
            self.trainer.step(batchsize, ignore_stale_grad=True)

    def get_bhQ(self):
        raise NotImplementedError()

class FactorizationMachine(QuadraticLayer):
    """Factorization Machine as a neural network layer.

    Args:
        input_size (int):
            The dimension of input value.
        factorization_size (int (<=input_size)):
            The rank of decomposition of interaction terms.
        act (string, optional):
            Name of activation function applied on FM output: "identity", "sigmoid", or "tanh". (default="identity")
        **kwargs:
    """

    def __init__(self, input_size, factorization_size=8, act="identity", **kwargs):
        super().__init__(**kwargs)
        self.factorization_size = factorization_size
        self.input_size = input_size
        self.trainer = None
        with self.name_scope():
            self.h = self.params.get("h", shape=(input_size,), dtype=np.float32)
            if factorization_size > 0:
                self.V = self.params.get("V", shape=(factorization_size, input_size), dtype=np.float32)
            else:
                self.V = self.params.get("V", shape=(1, input_size), dtype=np.float32) # dummy V
            self.bias = self.params.get("bias", shape=(1,), dtype=np.float32)
        self.act = act

    def hybrid_forward(self, F, x, h, V, bias):
        """Forward propagation of FM.

        Args:
          x: input vector of shape (N, d).
          h: linear coefficient of lenth d.
          V: matrix of shape (k, d).
        """
        if self.factorization_size <= 0:
            return bias + F.dot(x, h)
        Q = VtoQ(V, F) # (d,d)
        Qx = F.FullyConnected(x, weight=Q, bias=None, no_bias=True, num_hidden=self.input_size)
        act = {"identity": F.identity, "sigmoid": F.sigmoid, "tanh": F.tanh}[self.act]
        return act(bias + F.dot(x, h) +  F.sum(x*Qx, axis=1))

    def get_bhQ(self):
        """Returns linear and quadratic coefficients.
        """
        V = nd.zeros(self.V.shape) if self.factorization_size == 0 else self.V.data()
        return self.bias.data().asscalar(), self.h.data().asnumpy(), VtoQ(V, nd).asnumpy()

