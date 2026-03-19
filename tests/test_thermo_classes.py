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
    """DFTThermodynamics inherits from Thermodynamics."""
    ideal = IdealFermiGasThermodynamics(N=13.0)
    dft = DFTThermodynamics(N=13.0, ideal_fermi_gas=ideal)
    assert isinstance(dft, Thermodynamics)


def test_dft_has_ideal_fermi_gas_field():
    """DFTThermodynamics has an ideal_fermi_gas field."""
    ideal = IdealFermiGasThermodynamics(N=13.0)
    dft = DFTThermodynamics(N=13.0, ideal_fermi_gas=ideal)
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
