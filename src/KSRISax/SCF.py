import equinox as eqx
import optimistix as opt
import jax
import jax.numpy as jnp
from typing import Callable
from KSRISax.chem import *
from KSRISax.potentials import ExchangeCorrelation

class SelfConsistentFieldSolver(eqx.Module):
    grid: eqx.Module
    KohnShamSolver: eqx.Module
    PoissonSolver: eqx.Module
    ExternalPotential: Callable
    XC: ExchangeCorrelation
    max_iterations: int = eqx.field(static=True)
    convergence_threshold: float = eqx.field(static=True)
    L_max: int = eqx.field(static=True, default=0)
    FPI_damping: float = eqx.field(static=True, default=0.95)
    verbose: bool = False

    def scf_iteration(self, n, args):
        V_ext = self.ExternalPotential(self.grid)
        V_H = self.PoissonSolver.solve(n)
        V_H += -V_ext[-1]-V_H[-1]
        V_xc = self.XC.potential(n, self.grid)

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
        n_new = jnp.sum(self.KohnShamSolver.compute_normalised_densities(eigvecs) * occ['state_occ'][jnp.newaxis, :], axis=1)
        # Free state contribution (uniform)
        n_new += occ['free_occ'] / V_tot

        # Normalise
        n_new = args['N']/jnp.sum(n_new*self.grid.vol) * n_new

        aux = {
            'eigvals': eigvals,
            'eigvecs': eigvecs,
            'V_H' : V_H,
            'mu': mu,
            'occ': occ
        }

        if self.verbose:
            jax.debug.print('{E}', E = jnp.sum(eigvals * occ['state_occ']))
            jax.debug.print('{MSE}', MSE = jnp.mean((n_new-n)**2))

        return n_new, aux
    
    def __call__(self, N, T, n_initial):

        solver = opt.FixedPointIteration(rtol=self.convergence_threshold, atol=1e-8, norm = opt.max_norm, damp = self.FPI_damping)
        fp = opt.fixed_point(fn=self.scf_iteration, solver = solver, y0 = n_initial, args = {'N' : N, 'T' : T}, max_steps = self.max_iterations, has_aux=True, throw = False)

        n_final = fp.value

        # Compute other energy terms
        scf_result = fp.aux

        energies = scf_result['eigvals']
        occupancies = scf_result['occ']['state_occ']
        U_KS = jnp.sum(energies * occupancies)

        U_H = 0.5 * jnp.sum(scf_result['V_H'] * n_final * self.grid.vol)

        U_xc = jnp.sum(self.XC.energy(n_final, self.grid) * n_final * self.grid.vol)

        v_xc_integral = jnp.sum(self.XC.potential(n_final, self.grid) * n_final * self.grid.vol)

        U_bound = U_KS - U_H + U_xc - v_xc_integral

        jax.debug.print('{U_H} {U_xc} {v_xc_integral}',U_H=U_H, U_xc=U_xc, v_xc_integral=v_xc_integral)

        scf_result = scf_result | {'U_bound' : U_bound}

        return n_final, scf_result