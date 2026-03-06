import jax.numpy as jnp
import equinox as eqx
from KSRISax.poisson import PoissonSolver
from KSRISax.reign import KohnShamSolver
from KSRISax.potentials import CoulombPotential, LDA_exchange
from KSRISax.SCF import SelfConsistentFieldSolver
from KSRISax.grid import LogarithmicGrid
from KSRISax.chem import free_entropy_integral, bound_entropy_calc
from FDint_JAX import fermi_dirac_integral_three_half


class Thermodynamics(eqx.Module):
    N: float = eqx.field(static=True)
    rmin: float = eqx.field(static=True, default=1e-2)
    Nr: int = eqx.field(static=True,default=500)
    SCF_max_iterations: int = eqx.field(static=True, default=10)
    SCF_convergence_threshold: float = eqx.field(static=True, default=1e-4)
    SCF_L_max: int = eqx.field(static=True, default=0)
    SCF_damping: float = eqx.field(static=True, default=0.99)

    def _solve_SCF(self, V, T, n_initial):
        # Set up
        R = (3*V/(4*jnp.pi))**(1/3)
        grid = LogarithmicGrid.create(self.rmin, R, self.Nr)

        KSS = KohnShamSolver(grid=grid)
        PS = PoissonSolver(grid=grid)

        SCFS = SelfConsistentFieldSolver(
            grid=grid,
            KohnShamSolver=KSS,
            PoissonSolver=PS,
            ExternalPotential=lambda g: CoulombPotential(g, Z=self.N),
            XC=LDA_exchange(),
            max_iterations=self.SCF_max_iterations,
            convergence_threshold=self.SCF_convergence_threshold,
            L_max=self.SCF_L_max,
            FPI_damping=self.SCF_damping,
            verbose=True)

        # Run SCF to get energies, degeneracies, and chemical potential
        n_SCF, scf_result = SCFS(self.N, T, n_initial)

        scf_result['n_SCF'] = n_SCF

        return scf_result

    def _calc_EoS(self, V, T, n_initial):
        scf_result = self._solve_SCF(V, T, n_initial)
        mu = scf_result['mu']

        # Internal energy
        U_bound = scf_result['U_bound']
        U_free = ((jnp.sqrt(2) * V * T**(5/2)) / (jnp.pi**2)) * (3 * jnp.sqrt(jnp.pi) / 4) * fermi_dirac_integral_three_half(mu/T)
        U = U_bound + U_free

        # Number of free electrons
        Z = scf_result['occ']['free_occ']

        # Entropy
        S_bound = bound_entropy_calc(scf_result['eigvals'], scf_result['degen'], mu, T)
        S_free = self.N*V/(jnp.sqrt(2)*jnp.pi**2)*free_entropy_integral(mu,T)
        S = S_bound + S_free

        # Helmholtz free energy
        F = U - T*S

        # Pressure
        P_free = (2/3) * U_free
        P = scf_result['P_KS'] + P_free

        return U, (P, F, Z, S, mu)

    def nograd_call(self, V, T, n_initial):

        U, (P, F, Z, S, mu) = eqx.filter_jit(self._calc_EoS)(V, T, n_initial)

        thermo_dict = {
            'U': U,
            'Cv': None,
            'P' : P,
            'F' : F,
            'Z' : Z,
            'S' : S,
            'mu': mu,
        }

        return thermo_dict

    def grad_call(self, V, T, n_initial):

        (U, Cv), (P, F, Z, S, mu) = eqx.filter_jit(eqx.value_and_grad(self._calc_EoS,argnums=1,has_aux=True))(V, T, n_initial)

        thermo_dict = {
            'U': U,
            'Cv': Cv,
            'P' : P,
            'F' : F,
            'Z' : Z,
            'S' : S,
            'mu': mu,
        }

        return thermo_dict
