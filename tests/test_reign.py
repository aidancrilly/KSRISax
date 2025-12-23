from KSRISax.reign import KohnShamSolver
from KSRISax.grid import Grid
from KSRISax.potentials import CoulombPotential
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

def test_KohnShamEigen():
    grid = Grid.create(1e-8, 100.0, 500)
    V_ext = CoulombPotential(grid, Z=1.0)
    V_H = jnp.zeros_like(grid.x)
    V_xc = jnp.zeros_like(grid.x)

    KSS = KohnShamSolver(grid=grid)

    eigvals, eigvecs = KSS.EigenSolve(0, V_ext, V_H, V_xc)

    assert eigvals.shape == (500,)
    assert eigvecs.shape == (500, 500)

    bound_states = eigvals[eigvals < 0]
    for n, energy in enumerate(bound_states, start=1):
        expected_energy = -0.5 / n**2
        assert jnp.isclose(energy, expected_energy, atol=1e-2)
