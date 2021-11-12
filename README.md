# SpectralFlow
Tensorflow implementation of stochastic spectral-sums estimators

Requirements: Tensorflow, Tensorflow Probability (only for Rademacher samples)

Demo can be found [here](notebooks/StochasticChebyshev.ipynb), in a nutshell, using an
expansion to 50th degree of the log function:
```python
import spectralflow as sf
import tensorflow as tf

def op(x):
    # where A is some linear operator of size [32,32]
    return tf.tensordot(x, A, axes=[[-1],[-1]])

logdet = sf.chebyshev_logdet(op, shape=[128, 32], deg=50)
```
Note that this code is not yet fully validated, *a utiliser a vos risques et perils ;-)*

## Spectral Sum by Stochastic Chebyshev Approximation

This method is a reimplementation of Han et al. 2017, 2018, also used in
Ramesh & Lecun 2018:

  - https://arxiv.org/pdf/1606.00942.pdf
  - https://arxiv.org/pdf/1802.06355.pdf
  - https://arxiv.org/pdf/1806.00499.pdf
