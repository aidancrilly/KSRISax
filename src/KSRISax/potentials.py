import jax.numpy as jnp

def CoulombPotential(grid, Z):
    """Compute the Coulomb potential for a nucleus of charge Z on the given grid."""
    V_coulomb = -Z / jnp.maximum(grid.x, 1e-10)
    return V_coulomb