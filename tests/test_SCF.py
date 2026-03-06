from KSRISax.poisson import PoissonSolver
from KSRISax.reign import KohnShamSolver
from KSRISax.potentials import CoulombPotential, ZeroXC
from KSRISax.SCF import SelfConsistentFieldSolver
from KSRISax.grid import *
import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)

def test_SCF_iteration():
    grid = LinearGrid.create(0.0, 10.0, 500)

    KSS = KohnShamSolver(grid=grid)
    PS = PoissonSolver(grid=grid)

    SCFS = SelfConsistentFieldSolver(
        grid=grid,
        KohnShamSolver=KSS,
        PoissonSolver=PS,
        ExternalPotential=lambda g: CoulombPotential(g, Z=1.0),
        XC=ZeroXC(),
        max_iterations=50,
        convergence_threshold=1e-2)
    
    n_initial = jnp.zeros_like(grid.xc)
    n_final, aux = SCFS.scf_iteration(n_initial,args={'N':1,'T':0.01})

    ground_state  = jnp.exp(-grid.xc)
    ground_state_norm = jnp.sum((ground_state**2) * grid.vol)

    assert jnp.allclose(grid.xc**2 * n_final, grid.xc**2 * (ground_state**2) / ground_state_norm, atol=1e-2)

    n_SCF, _ = SCFS(N=jnp.array(1.0), T=jnp.array(0.01), n_initial=n_initial)

    assert jnp.sum(grid.xc*n_SCF*grid.vol) > jnp.sum(grid.xc*n_final*grid.vol)
    # Will need more iterations to pass
    assert jnp.isclose(jnp.sum(grid.xc*n_SCF*grid.vol), 2.18, rtol=1e-1)

    grid = LogarithmicGrid.create(1e-2, 10.0, 500)

    KSS = KohnShamSolver(grid=grid)
    PS = PoissonSolver(grid=grid)

    SCFS = SelfConsistentFieldSolver(
        grid=grid,
        KohnShamSolver=KSS,
        PoissonSolver=PS,
        ExternalPotential=lambda g: CoulombPotential(g, Z=1.0),
        XC=ZeroXC(),
        max_iterations=50,
        convergence_threshold=1e-2)
    
    n_initial = jnp.zeros_like(grid.xc)
    n_final, aux = SCFS.scf_iteration(n_initial,args={'N':1,'T':0.01})

    ground_state  = jnp.exp(-grid.xc)
    ground_state_norm = jnp.sum((ground_state**2) * grid.vol)

    assert jnp.allclose(grid.xc**2 * n_final, grid.xc**2 * (ground_state**2) / ground_state_norm, atol=1e-2)

    n_SCF, _ = SCFS(N=jnp.array(1.0), T=jnp.array(0.01), n_initial=n_initial)

    assert jnp.sum(grid.xc*n_SCF*grid.vol) > jnp.sum(grid.xc*n_final*grid.vol)
    # Will need more iterations to pass
    assert jnp.isclose(jnp.sum(grid.xc*n_SCF*grid.vol), 2.18, rtol=1e-1)