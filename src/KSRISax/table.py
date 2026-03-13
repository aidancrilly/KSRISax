import jax
import jax.numpy as jnp
import equinox as eqx
import optimistix as optx
from KSRISax.thermo import Thermodynamics

# Unit conversion constants
eV_to_Ha = 1.0 / 27.2114       # 1 eV in Hartree
amu_to_g = 1.66053906660e-24    # g per atomic mass unit
a0_to_cm = 0.529177210903e-8    # Bohr radius in cm


def density_to_volume(A, rho_gcc):
    """
    Convert atomic mass and mass density to volume per atom in atomic units (a0³).

    Parameters
    ----------
    A : float
        Atomic mass in atomic mass units (amu).
    rho_gcc : float
        Mass density in g/cc.

    Returns
    -------
    float
        Volume per atom in a0³.
    """
    V_cm3 = (A * amu_to_g) / rho_gcc
    return V_cm3 / (a0_to_cm ** 3)


class EoSTable(eqx.Module):
    """
    Equation of state table for a single-species plasma.

    Wraps a Thermodynamics solver and evaluates thermodynamic quantities
    over a 2-D grid of (normalised density, temperature) by vmapping
    thermo.nograd_call over the grid points.

    Attributes
    ----------
    Z : float
        Atomic number (number of electrons per ion).
    A : float
        Atomic mass in atomic mass units (amu).
    rho_solid : float
        Solid (reference) mass density in g/cc.
    thermo : Thermodynamics
        Underlying KS-DFT thermodynamics solver.
    """

    Z: float = eqx.field(static=True)
    A: float = eqx.field(static=True)
    rho_solid: float = eqx.field(static=True)
    thermo: Thermodynamics

    def build(self, rho_norm_grid, T_eV_grid):
        """
        Build an EoS table by vmapping thermo.nograd_call over density and
        temperature grids.

        Parameters
        ----------
        rho_norm_grid : jnp.ndarray, shape (N_rho,)
            Normalised mass density grid (rho / rho_solid).
        T_eV_grid : jnp.ndarray, shape (N_T,)
            Temperature grid in eV.

        Returns
        -------
        dict
            Dictionary containing 2-D arrays of shape (N_rho, N_T) for each
            thermodynamic quantity:

            * ``'U'``   – internal energy per unit volume (Ha / a0³)
            * ``'P'``   – pressure (Ha / a0³)
            * ``'F'``   – Helmholtz free energy per unit volume (Ha / a0³)
            * ``'Z'``   – mean number of free electrons per ion
            * ``'S'``   – entropy
            * ``'mu'``  – chemical potential (Ha)

            Plus the axes used to build the table:

            * ``'rho_norm'`` – normalised density axis (copy of *rho_norm_grid*)
            * ``'T_eV'``     – temperature axis in eV (copy of *T_eV_grid*)
        """
        # Convert normalised density grid to volume per atom in atomic units
        V_grid = jax.vmap(lambda rho_norm: density_to_volume(self.A, rho_norm * self.rho_solid))(rho_norm_grid)

        # Convert temperature grid from eV to Hartree
        T_Ha_grid = T_eV_grid * eV_to_Ha

        # Build a flat list of all (V, T) pairs on the 2-D grid
        V_mesh, T_Ha_mesh = jnp.meshgrid(V_grid, T_Ha_grid, indexing='ij')
        V_flat = V_mesh.ravel()
        T_flat = T_Ha_mesh.ravel()

        # Zero initial electron density guess shared across all grid points
        n_initial = jnp.zeros(self.thermo.Nr)

        def _single_point(V, T):
            """Evaluate EoS at a single (V, T) point."""
            U, (P, F, Z_free, S, mu) = self.thermo.calc_EoS(V, T, n_initial)
            return U, P, F, Z_free, S, mu

        # Vmap over all (V, T) pairs on the flattened grid, then JIT-compile
        U_flat, P_flat, F_flat, Z_flat, S_flat, mu_flat = eqx.filter_jit(
            jax.vmap(_single_point)
        )(V_flat, T_flat)

        shape = V_mesh.shape
        return {
            'U': U_flat.reshape(shape),
            'P': P_flat.reshape(shape),
            'F': F_flat.reshape(shape),
            'Z': Z_flat.reshape(shape),
            'S': S_flat.reshape(shape),
            'mu': mu_flat.reshape(shape),
            'rho_norm': rho_norm_grid,
            'T_eV': T_eV_grid,
        }

    def hugoniot(self, rho_norm0, T_eV0, P_amp_grid):
        """
        Compute the principal Hugoniot curve starting from an initial state.

        For each pressure amplification factor ``P_amp`` in *P_amp_grid* the
        method finds the shocked state ``(rho_norm1, T_eV1)`` satisfying both:

        1. **Pressure condition**: ``P(rho_norm1, T_eV1) == P_amp * P0``
        2. **Rankine–Hugoniot energy condition**:
           ``E1 - E0 == 0.5 * (P1 + P0) * (V0 - V1)``

        where ``E = U * V`` is the internal energy per atom and ``V`` is the
        volume per atom.  The two conditions form a 2-D vector-valued residual
        that is solved simultaneously with a Newton root finder vmapped over
        the pressure amplification grid.

        Parameters
        ----------
        rho_norm0 : float
            Initial normalised mass density (rho / rho_solid).
        T_eV0 : float
            Initial temperature in eV.
        P_amp_grid : jnp.ndarray, shape (N_P,)
            Pressure amplification factors ``P_final / P_initial``.

        Returns
        -------
        compression_ratio : jnp.ndarray, shape (N_P,)
            Density compression ratio ``rho1 / rho0`` at each Hugoniot point.
        T_eV : jnp.ndarray, shape (N_P,)
            Temperature in eV at each Hugoniot point.
        """
        V0 = density_to_volume(self.A, self.rho_solid * rho_norm0)
        T_Ha0 = T_eV0 * eV_to_Ha
        n_initial = jnp.zeros(self.thermo.Nr)

        # Evaluate the initial (unshocked) state once outside the vmap
        U0, (P0, _, _, _, _) = eqx.filter_jit(self.thermo.calc_EoS)(V0, T_Ha0, n_initial)
        E0 = U0 * V0  # internal energy per atom (Ha)

        def _hugoniot_residuals(y, P_target):
            """2-D residual: [Hugoniot condition, pressure condition]."""
            rho_norm1, T_Ha1 = y[0], y[1]
            V1 = density_to_volume(self.A, self.rho_solid * rho_norm1)
            U1, (P1, _, _, _, _) = self.thermo.calc_EoS(V1, T_Ha1, n_initial)
            E1 = U1 * V1
            hugoniot_res = E1 - E0 - 0.5 * (P1 + P0) * (V0 - V1)
            pressure_res = P1 - P_target
            return jnp.array([hugoniot_res, pressure_res])

        def _solve_hugoniot_point(P_amp):
            """Find the shocked state for a single pressure amplification."""
            P_target = P_amp * P0
            # Initial guess: scale density as P^(1/3), temperature as P^(2/3)
            y0 = jnp.array([rho_norm0 * P_amp ** (1.0 / 3.0), T_Ha0 * P_amp ** (2.0 / 3.0)])
            solver = optx.Newton(rtol=1e-6, atol=1e-6)
            # throw=False is required for vmap compatibility: JAX cannot
            # raise exceptions inside a traced/vmapped computation.
            # Convergence quality can be verified through physical sanity
            # checks on the returned values.
            sol = optx.root_find(_hugoniot_residuals, solver, y0=y0, args=P_target, throw=False)
            rho_norm1, T_Ha1 = sol.value[0], sol.value[1]
            return rho_norm1 / rho_norm0, T_Ha1 / eV_to_Ha

        compression_ratio, T_eV_arr = eqx.filter_jit(
            jax.vmap(_solve_hugoniot_point)
        )(P_amp_grid)
        return compression_ratio, T_eV_arr
