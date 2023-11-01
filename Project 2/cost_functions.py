import jax.numpy as jnp


def CostCrossEntropy(target):
    def func(X):
        return -(1.0 / target.size) * jnp.sum(target * jnp.log(X + 10e-10))

    return func

def analytic_derivate_CostCross(activation_output, target):
    dCW_da = (activation_output - target) / (activation_output * (1 - activation_output))
    return dCW_da

