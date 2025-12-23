from KSRISax.poisson import PoissonSolver
from KSRISax.grid import Grid
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

def test_PoissonSolver():
    grid = Grid.create(0.0, 100.0, 500)

    PS = PoissonSolver(grid=grid)

    rho = jnp.where(grid.xc < 10.0, 1.0, 0.0) / (4.0/3.0 * jnp.pi * 10.0**3)
    V_H = PS.solve(rho, V_gauge=1.0/grid.xb[-1])

    V_H_expected = jnp.where(grid.xc < 10.0,
                             0.5 * (3 - grid.xc**2/100.0) / 10.0,
                             1.0/grid.xc)

    assert V_H.shape == (grid.Nx,)
    assert jnp.allclose(V_H, V_H_expected, atol=1e-3)