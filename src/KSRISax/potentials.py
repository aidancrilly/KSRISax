import jax.numpy as jnp

def CoulombPotential(grid, Z):
    """Compute the Coulomb potential for a nucleus of charge Z on the given grid."""
    V_coulomb = -Z / jnp.maximum(grid.xc, 1e-10)
    return V_coulomb

def LDA_exchange(n, grid):
    # See https://en.wikipedia.org/wiki/Local-density_approximation
    return - (3 / jnp.pi)**(1/3) * jnp.maximum(n,1e-30)**(1/3)