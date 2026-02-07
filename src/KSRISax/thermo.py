import jax
import jax.numpy as jnp
import equinox as eqx
from KSRISax.poisson import PoissonSolver
from KSRISax.reign import KohnShamSolver
from KSRISax.potentials import CoulombPotential, LDA_exchange
from KSRISax.SCF import SelfConsistentFieldSolver
from KSRISax.grid import LogarithmicGrid, LinearGrid
from FDint_JAX import fermi_dirac_integral_three_half


class Thermodynamics(eqx.Module):   
    N: float = eqx.field(static=True)
    rmin: float = eqx.field(static=True, default=1e-2)
    Nr: int = eqx.field(static=True,default=500)
    SCF_max_iterations: int = eqx.field(static=True, default=10)
    SCF_convergence_threshold: float = eqx.field(static=True, default=1e-4)
    SCF_L_max: int = eqx.field(static=True, default=0)
    SCF_damping: float = eqx.field(static=True, default=0.99)

    def __call__(self, V, T, n_initial):
        """
        U = U_bound + U_free

        U_bound = sum over all bounded n,l states (E * degeneracy factor * occupancy)

        U_free = int_0^+inf (density of state * E * occupancy)
            = sqrt(2) * V * T^5/2 / pi^2 * F_3/2 (mu/T)

        """
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
            ExchangeCorrelationPotential=lambda n, g: LDA_exchange(n, g),
            max_iterations=self.SCF_max_iterations,
            convergence_threshold=self.SCF_convergence_threshold,
            L_max=self.SCF_L_max,
            FPI_damping=self.SCF_damping,
            verbose=True)

        # Run SCF to get energies, degeneracies, and chemical potential
        n_SCF, scf_result = SCFS(self.N, T, n_initial)

        # bound internal energy
        energies = scf_result['eigvals']
        occupancies = scf_result['occ']['state_occ']
        U_bound = jnp.sum(energies * occupancies)
        
        # free internal energy
        mu = scf_result['mu']
        gamma_factor = 3 * jnp.sqrt(jnp.pi) / 4
        U_free = ((jnp.sqrt(2) * V * T**(5/2)) / (jnp.pi**2)) * gamma_factor * fermi_dirac_integral_three_half(mu/T)

        # total internal energy
        U = U_bound + U_free

        thermo_dict = {
            'n_SCF': n_SCF,
            'U_bound': U_bound,
            'U_free': U_free,
            'U_total': U,
            'mu': mu,
            'energies': energies,
            'u_nl': scf_result['eigvecs'],
            'occupancies': occupancies
        }

        return thermo_dict
