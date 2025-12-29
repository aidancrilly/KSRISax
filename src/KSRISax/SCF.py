import equinox as eqx
import optimistix as opt
import jax
import jax.numpy as jnp
from typing import Callable
from KSRISax.chem import find_chemical_potential_w_KSstates

class SelfConsistentFieldSolver(eqx.Module):
    grid: eqx.Module
    KohnShamSolver: eqx.Module
    PoissonSolver: eqx.Module
    ExternalPotential: Callable
    ExchangeCorrelationPotential: Callable
    max_iterations: int = eqx.field(static=True)
    convergence_threshold: float = eqx.field(static=True)

    def scf_iteration(self, n, args):
        V_ext = self.ExternalPotential(self.grid)
        V_H = self.PoissonSolver.solve(n, V_gauge = 0.0)#-V_ext[-1])
        V_xc = self.ExchangeCorrelationPotential(n, self.grid)

        l = 0
        degeneracy = 2 * (2 * l + 1)

        eigvals, eigvecs = self.KohnShamSolver.EigenSolve(l, V_ext, V_H, V_xc)

        mu, occ = find_chemical_potential_w_KSstates(eigvals, degeneracies=degeneracy, V=jnp.sum(self.grid.vol), N=args['N'], T=args['T'])

        n_new = jnp.sum(((eigvecs / self.grid.xc[:, jnp.newaxis])**2) * occ[jnp.newaxis, :], axis=1)

        # Normalise
        n_new = args['N']/jnp.sum(n_new*self.grid.vol) * n_new

        aux = {
            'eigvals': eigvals,
            'mu': mu,
            'occ': occ
        }

        return n_new, aux

    @jax.jit
    def __call__(self, N, T):
        n_initial = jnp.zeros_like(self.grid.xc)

        solver = opt.Newton(rtol=self.convergence_threshold, atol=1e-8, norm = opt.max_norm)
        fp = opt.fixed_point(fn=self.scf_iteration, solver = solver, y0 = n_initial, args = {'N' : N, 'T' : T}, max_steps = self.max_itertions, has_aux=True)

        n_final = fp.value
        return n_final, fp.aux