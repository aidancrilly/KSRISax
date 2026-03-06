import jax
import jax.numpy as jnp
import equinox as eqx
import abc

def CoulombPotential(grid, Z):
    """Compute the Coulomb potential for a nucleus of charge Z on the given grid."""
    V_coulomb = -Z / jnp.maximum(grid.xc, 1e-10)
    return V_coulomb

class ExchangeCorrelation(eqx.Module):

    @abc.abstractmethod
    def energy(self, n, grid):
        raise NotImplementedError

    def potential(self, n, grid):
        v_xc = self.energy(n,grid) + n * jax.vmap(jax.grad(self.energy,argnums=0),in_axes=[0,None])(n, grid)
        return v_xc

class ZeroXC(ExchangeCorrelation):

    def energy(self, n, grid):
        return jnp.zeros_like(n)

class LDA_exchange(ExchangeCorrelation):

    def energy(self, n, grid):
        # See https://en.wikipedia.org/wiki/Local-density_approximation
        return - (3/4) * (3 / jnp.pi)**(1/3) * jnp.maximum(n,1e-30)**(1/3)
