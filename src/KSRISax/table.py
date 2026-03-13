import jax
import jax.numpy as jnp
import equinox as eqx
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
