from KSRISax.table import EoSTable, density_to_volume, amu_to_g, a0_to_cm
from KSRISax.thermo import Thermodynamics
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)


def test_density_to_volume():
    # Solid aluminium: A = 26.982 amu, rho = 2.7 g/cc
    A_Al = 26.982
    rho_Al = 2.7
    V = density_to_volume(A_Al, rho_Al)

    # Verify against manual calculation using the same constants
    V_expected = (A_Al * amu_to_g) / rho_Al / (a0_to_cm ** 3)
    assert np.isclose(float(V), V_expected, rtol=1e-6)
    assert V > 0

    # Doubling density should halve volume
    V2 = density_to_volume(A_Al, 2.0 * rho_Al)
    assert np.isclose(float(V2), float(V) / 2.0, rtol=1e-6)

    # Doubling atomic mass should double volume
    V3 = density_to_volume(2.0 * A_Al, rho_Al)
    assert np.isclose(float(V3), float(V) * 2.0, rtol=1e-6)


def test_EoSTable_build():
    # Use hydrogen with minimal SCF settings for a fast test
    therm = Thermodynamics(
        N=1,
        rmin=1e-3,
        Nr=50,
        SCF_max_iterations=5,
        SCF_L_max=0,
        SCF_convergence_threshold=1e-2,
        SCF_damping=0.5,
        verbose=False,
    )
    table = EoSTable(Z=1, A=1.008, rho_solid=1.0, thermo=therm)

    rho_norm_grid = jnp.array([0.5, 1.0])
    T_eV_grid = jnp.array([1.0, 10.0])

    result = table.build(rho_norm_grid, T_eV_grid)

    N_rho = len(rho_norm_grid)
    N_T = len(T_eV_grid)

    # Check output shape for every thermodynamic quantity
    for key in ('U', 'P', 'F', 'Z', 'S', 'mu'):
        assert result[key].shape == (N_rho, N_T), f"Expected shape ({N_rho}, {N_T}) for '{key}', got {result[key].shape}"

    # Axis arrays should be returned unchanged
    assert jnp.allclose(result['rho_norm'], rho_norm_grid)
    assert jnp.allclose(result['T_eV'], T_eV_grid)

    # Physical sanity checks
    assert jnp.all(result['P'] > 0), "Pressure should be positive"
    assert jnp.all(result['Z'] >= 0), "Free electron count should be non-negative"
    assert jnp.all(result['Z'] <= 1.0 + 1e-4), "Free electrons per H atom cannot exceed 1"

    # Higher temperature at the same density should give higher pressure
    assert jnp.all(result['P'][:, 1] > result['P'][:, 0]), "Pressure should increase with temperature"
