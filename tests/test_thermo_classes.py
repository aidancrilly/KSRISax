from KSRISax.thermo import Thermodynamics, IdealFermiGasThermodynamics, DFTThermodynamics
from KSRISax.chem import find_free_chemical_potential, fermi_energy
import jax.numpy as jnp
import jax

jax.config.update('jax_enable_x64', True)


def test_thermodynamics_is_abstract():
    """Thermodynamics cannot be instantiated directly."""
    import pytest
    with pytest.raises(TypeError):
        Thermodynamics(N=1.0)


def test_ideal_fermi_gas_is_thermodynamics():
    """IdealFermiGasThermodynamics inherits from Thermodynamics."""
    ideal = IdealFermiGasThermodynamics(N=13.0)
    assert isinstance(ideal, Thermodynamics)


def test_dft_is_thermodynamics():
    """DFTThermodynamics inherits from Thermodynamics and initialises ideal_fermi_gas internally."""
    dft = DFTThermodynamics(N=13.0)
    assert isinstance(dft, Thermodynamics)
    assert isinstance(dft.ideal_fermi_gas, IdealFermiGasThermodynamics)
    assert dft.ideal_fermi_gas.N == 13.0


def test_find_free_chemical_potential():
    """Chemical potential for free electron gas converges to Fermi energy at low T."""
    N = 13.0
    V = 100.0
    T = 0.01

    mu = find_free_chemical_potential(V, N, T)
    E_F = fermi_energy(V, N)

    # At low T, mu should be close to the Fermi energy
    assert jnp.isclose(mu, E_F, rtol=0.1)


def test_ideal_fermi_gas_nograd_call():
    """IdealFermiGasThermodynamics nograd_call returns expected keys and physical values."""
    N = 13.0
    V = 100.0
    T = 1.0

    ideal = IdealFermiGasThermodynamics(N=N)
    result = ideal.nograd_call(V, T)

    # Check all expected keys are present
    assert set(result.keys()) == {'U', 'Cv', 'P', 'F', 'Z', 'S', 'mu'}

    # Cv is None for nograd_call
    assert result['Cv'] is None

    # All electrons are free in ideal Fermi gas
    assert jnp.isclose(result['Z'], N)

    # Physical checks: U, P should be positive
    assert result['U'] > 0
    assert result['P'] > 0


def test_ideal_fermi_gas_grad_call():
    """IdealFermiGasThermodynamics grad_call computes Cv."""
    N = 13.0
    V = 100.0
    T = 1.0

    ideal = IdealFermiGasThermodynamics(N=N)
    result = ideal.grad_call(V, T)

    # Cv should be computed (not None)
    assert result['Cv'] is not None

    # All electrons are free
    assert jnp.isclose(result['Z'], N)


def test_ideal_fermi_gas_calc_EoS_from_mu():
    """calc_EoS_from_mu returns consistent U_free, P_free, S_free."""
    N = 13.0
    V = 100.0
    T = 1.0

    ideal = IdealFermiGasThermodynamics(N=N)
    mu = find_free_chemical_potential(V, N, T)
    U_free, P_free, S_free = ideal.calc_EoS_from_mu(V, T, mu)

    # P = (2/3) U_free / V
    assert jnp.isclose(P_free, (2.0 / 3.0) * U_free / V, rtol=1e-10)

    # U_free and P_free should be positive
    assert U_free > 0
    assert P_free > 0


def test_ideal_fermi_gas_classical_limit():
    """At high T and low density, ideal Fermi gas approaches classical ideal gas.

    Classical ideal gas: PV = NT, U_total = (3/2)NT.
    Here U is energy density so U*V gives total energy.
    """
    N = 1.0
    V = 1e6  # Very large volume -> low density
    T = 1e3  # Very high temperature

    ideal = IdealFermiGasThermodynamics(N=N)
    result = ideal.nograd_call(V, T)

    # Classical ideal gas: PV/(NT) -> 1
    PV_over_NT = result['P'] * V / (N * T)
    assert jnp.isclose(PV_over_NT, 1.0, rtol=0.01), f"PV/(NT) = {PV_over_NT}, expected ~1.0"

    # Classical ideal gas: U*V = (3/2)*N*T
    UV_over_NT = result['U'] * V / (N * T)
    assert jnp.isclose(UV_over_NT, 1.5, rtol=0.01), f"U*V/(NT) = {UV_over_NT}, expected ~1.5"
