import equinox as eqx
import optimistix as opt
import jax
import jax.numpy as jnp
from typing import Callable
from KSRISax.chem import *

class SelfConsistentFieldSolver(eqx.Module):
    grid: eqx.Module
    KohnShamSolver: eqx.Module
    PoissonSolver: eqx.Module
    ExternalPotential: Callable
    ExchangeCorrelationPotential: Callable
    max_iterations: int = eqx.field(static=True)
    convergence_threshold: float = eqx.field(static=True)
    L_max: int = eqx.field(static=True, default=0)

    def scf_iteration(self, n, args):
        V_ext = self.ExternalPotential(self.grid)
        V_H = self.PoissonSolver.solve(n, V_gauge = 0.0)#-V_ext[-1])
        V_xc = self.ExchangeCorrelationPotential(n, self.grid)

        eigvals = []
        eigvecs = []
        degen = []
        for l in range(self.L_max+1):
            degeneracy = 2 * (2 * l + 1)

            _eigvals, _eigvecs = self.KohnShamSolver.EigenSolve(l, V_ext, V_H, V_xc)

            eigvals.append(_eigvals)
            eigvecs.append(_eigvecs)
            degen.append(degeneracy * jnp.ones_like(_eigvals))

        eigvals = jnp.concatenate(eigvals)
        eigvecs = jnp.concatenate(eigvecs, axis=1)
        degen   = jnp.concatenate(degen)

        V_tot = jnp.sum(self.grid.vol)

        mu, occ = find_chemical_potential_w_freecontinuum(eigvals, degeneracies=degen, V=V_tot, N=args['N'], T=args['T'])

        # Calculate new density
        # Bound state contribution
        n_new = jnp.sum(((eigvecs / self.grid.xc[:, jnp.newaxis])**2) * occ['state_occ'][jnp.newaxis, :], axis=1)
        # Free state contribution
        n_new += occ['free_occ'] / V_tot

        # Normalise
        n_new = args['N']/jnp.sum(n_new*self.grid.vol) * n_new

        aux = {
            'eigvals': eigvals,
            'mu': mu,
            'occ': occ
        }

        return n_new, aux
    
    def __call__(self, N, T):
        n_initial = jnp.zeros_like(self.grid.xc)

        solver = opt.Newton(rtol=self.convergence_threshold, atol=1e-8, norm = opt.max_norm)
        fp = opt.fixed_point(fn=self.scf_iteration, solver = solver, y0 = n_initial, args = {'N' : N, 'T' : T}, max_steps = self.max_iterations, has_aux=True, throw = False)

        n_final = fp.value
        return n_final, fp.aux