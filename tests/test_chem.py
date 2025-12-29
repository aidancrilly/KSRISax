from KSRISax.chem import find_chemical_potential_w_freecontinuum
import jax.numpy as jnp

def test_find_chemical_potential_w_freecontinuum():
    energies = jnp.array([-1.0, -0.5])
    degeneracies = jnp.array([2, 2])
    N = 3
    T = 0.1
    V = 100.0

    mu, occ = find_chemical_potential_w_freecontinuum(energies, degeneracies, V, N, T)

    expected_mu = -0.5
    assert jnp.isclose(mu, expected_mu, atol=1e-2)

    N = 5

    mu, occ = find_chemical_potential_w_freecontinuum(energies, degeneracies, V, N, T)

    assert mu > 0.0