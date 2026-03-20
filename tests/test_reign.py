from KSRISax.reign import KohnShamSolver
from KSRISax.grid import *
from KSRISax.potentials import CoulombPotential
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)

def test_KohnShamEigen():
    grid = LinearGrid.create(0.0, 100.0, 500)
    V_ext = CoulombPotential(grid, Z=1.0)
    V_H = jnp.zeros_like(grid.xc)
    V_xc = jnp.zeros_like(grid.xc)

    KSS = KohnShamSolver(grid=grid)

    eigvals, eigvecs = KSS.EigenSolve(0, V_ext, V_H, V_xc)

    assert eigvals.shape == (grid.Nx,)
    assert eigvecs.shape == (grid.Nx, grid.Nx)

    assert jnp.all(jnp.isclose(jnp.abs(eigvecs[:,0]),jnp.exp(-grid.xc)*grid.xc/jnp.sqrt(jnp.pi),rtol=1e-2,atol=1e-2))

    bound_states = eigvals[eigvals < 0]
    print(bound_states)
    for n, energy in enumerate(bound_states, start=1):
        expected_energy = -0.5 / n**2
        assert jnp.isclose(energy, expected_energy, atol=1e-2)

    grid = LogarithmicGrid.create(1e-4, 100.0, 500)
    V_ext = CoulombPotential(grid, Z=1.0)
    V_H = jnp.zeros_like(grid.xc)
    V_xc = jnp.zeros_like(grid.xc)

    KSS = KohnShamSolver(grid=grid)

    eigvals, eigvecs = KSS.EigenSolve(0, V_ext, V_H, V_xc)

    assert eigvals.shape == (grid.Nx-1,)
    assert eigvecs.shape == (grid.Nx, grid.Nx-1)

    assert jnp.all(jnp.isclose(jnp.abs(eigvecs[:,0]),jnp.exp(-grid.xc)*grid.xc/jnp.sqrt(jnp.pi),rtol=1e-2,atol=1e-2))

    bound_states = eigvals[eigvals < 0]
    print(bound_states)
    for n, energy in enumerate(bound_states, start=1):
        expected_energy = -0.5 / n**2
        assert jnp.isclose(energy, expected_energy, atol=1e-2)


def test_KohnShamEigen_extended_grid():
    Nextend = 100

    # Linear grid with extension
    grid = LinearGrid.create(0.0, 100.0, 500, Nextend=Nextend)
    assert grid.Nextend == Nextend
    assert grid.xc_ext.shape == (grid.Nx + Nextend,)
    assert grid.xb_ext.shape == (grid.Nx + Nextend + 1,)
    assert grid.vol_ext.shape == (grid.Nx + Nextend,)
    # Extended grid matches original up to Nx
    assert jnp.allclose(grid.xc_ext[:grid.Nx], grid.xc)
    assert jnp.allclose(grid.xb_ext[:grid.Nx + 1], grid.xb)

    V_ext = CoulombPotential(grid, Z=1.0)
    V_H = jnp.zeros_like(grid.xc)
    V_xc = jnp.zeros_like(grid.xc)

    KSS = KohnShamSolver(grid=grid)
    eigvals, eigvecs = KSS.EigenSolve(0, V_ext, V_H, V_xc)

    # eigvecs are returned on the original grid
    assert eigvecs.shape == (grid.Nx, grid.Nx + Nextend)
    # More eigenvalues due to larger solve domain
    assert eigvals.shape == (grid.Nx + Nextend,)

    # 1s orbital should still be correct on original grid
    assert jnp.all(jnp.isclose(jnp.abs(eigvecs[:,0]), jnp.exp(-grid.xc)*grid.xc/jnp.sqrt(jnp.pi), rtol=1e-2, atol=1e-2))

    bound_states = eigvals[eigvals < 0]
    for n, energy in enumerate(bound_states, start=1):
        expected_energy = -0.5 / n**2
        assert jnp.isclose(energy, expected_energy, atol=1e-2)

    # Logarithmic grid with extension
    grid = LogarithmicGrid.create(1e-4, 100.0, 500, Nextend=Nextend)
    assert grid.Nextend == Nextend
    assert grid.xc_ext.shape == (grid.Nx + Nextend,)
    assert grid.xb_ext.shape == (grid.Nx + Nextend + 1,)
    assert grid.vol_ext.shape == (grid.Nx + Nextend,)
    # Extended grid matches original up to Nx
    assert jnp.allclose(grid.xc_ext[:grid.Nx], grid.xc)
    assert jnp.allclose(grid.xb_ext[:grid.Nx + 1], grid.xb)

    V_ext = CoulombPotential(grid, Z=1.0)
    V_H = jnp.zeros_like(grid.xc)
    V_xc = jnp.zeros_like(grid.xc)

    KSS = KohnShamSolver(grid=grid)
    eigvals, eigvecs = KSS.EigenSolve(0, V_ext, V_H, V_xc)

    # eigvecs are returned on the original grid (Nx rows)
    assert eigvecs.shape == (grid.Nx, grid.Nx + Nextend - 1)
    assert eigvals.shape == (grid.Nx + Nextend - 1,)

    # 1s orbital should still be correct on original grid
    assert jnp.all(jnp.isclose(jnp.abs(eigvecs[:,0]), jnp.exp(-grid.xc)*grid.xc/jnp.sqrt(jnp.pi), rtol=1e-2, atol=1e-2))

    bound_states = eigvals[eigvals < 0]
    for n, energy in enumerate(bound_states, start=1):
        expected_energy = -0.5 / n**2
        assert jnp.isclose(energy, expected_energy, atol=1e-2)
