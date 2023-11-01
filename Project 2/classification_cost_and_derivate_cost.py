import jax.numpy as jnp


def CostCrossEntropy(target):
    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func