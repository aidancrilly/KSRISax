from KSRISax.table import EoSTable, density_to_volume, amu_to_g, a0_to_cm
from KSRISax.thermo import DFTThermodynamics
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
    therm = DFTThermodynamics(
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


def test_hugoniot():
    # Use hydrogen with minimal SCF settings for a fast test
    therm = DFTThermodynamics(
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

    rho_norm0 = 1.0
    T_eV0 = 5.0
    P_amp_grid = jnp.array([2.0, 5.0, 10.0])

    compression_ratio, T_eV_arr = table.hugoniot(rho_norm0, T_eV0, P_amp_grid)

    # Shape checks
    assert compression_ratio.shape == (len(P_amp_grid),)
    assert T_eV_arr.shape == (len(P_amp_grid),)

    # Physical sanity: higher pressure → higher compression and temperature
    assert jnp.all(compression_ratio > 1.0), "Shock should compress: rho1 > rho0"
    assert jnp.all(T_eV_arr > T_eV0), "Shock should heat: T1 > T0"
    assert jnp.all(jnp.diff(compression_ratio) > 0), "Compression should increase with P_amp"
    assert jnp.all(jnp.diff(T_eV_arr) > 0), "Temperature should increase with P_amp"

if __name__ == '__main__':
    N = 13

    therm = DFTThermodynamics(
        N=N,
        rmin=1e-5,
        Nr=200,
        SCF_max_iterations=200,
        SCF_L_max=3,
        SCF_convergence_threshold=1e-6,
        SCF_damping=0.1,
        verbose=False
        )
    table = EoSTable(Z=13, A=27.0, rho_solid=2.7, thermo=therm)

    rho_norm0 = 1.0
    T_eV0 = 10.0
    P_amp_grid = jnp.geomspace(1.0,1e1,5)

    compression_ratio, T_eV_arr = table.hugoniot(rho_norm0, T_eV0, P_amp_grid)

    import matplotlib.pyplot as plt

    plt.semilogy(compression_ratio, P_amp_grid)

    plt.show()
