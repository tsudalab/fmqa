import unittest
import numpy as np

from fmqa import FMBQM
class TestModel(unittest.TestCase):
    def test_prediction_accuracy_binary(self):
        np.random.seed(12345)
        xs = np.random.randint(2, size=(20, 32))
        ys = np.sum(xs[:,1:] * xs[:, :1], axis=1) + xs[:,0] * xs[:,-1]
    
        fmbqm = FMBQM.from_data(xs, ys)
        for i in range(20):
            self.assertAlmostEqual(fmbqm.predict(xs[i])[0], ys[i], places=3)

    def test_prediction_accuracy_spin(self):
        np.random.seed(12345)
        xs = np.random.randint(2, size=(20, 32)) * 2 - 1
        ys = np.sum(xs[:,1:] * xs[:, :1], axis=1) + xs[:,0] * xs[:,-1]
    
        fmbqm = FMBQM.from_data(xs, ys)
        for i in range(20):
            self.assertAlmostEqual(fmbqm.predict(xs[i])[0], ys[i], places=3)

if __name__ == "__main__":
    unittest.main()
