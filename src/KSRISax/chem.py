
import jax.numpy as jnp
import optimistix as opt
from KSRISax.quad import quad
from FDint_JAX import fermi_dirac_integral_half

_ESCALE = 0.1

def fermi_dirac_dist(energy, mu, T):
    occ = 1.0 / (1.0 + jnp.exp((energy - mu) / T))
    return occ

def bound_mask(E, E_scale = _ESCALE):
    return jnp.where(E < 0.0, 1.0, jnp.exp(-E/E_scale))

def fermi_energy(V, N):
    return 0.5 * (3 * jnp.pi**2 * (N/V))**(2/3)

def thermal_deBroglie_wavelength(T):
    return jnp.sqrt(2 * jnp.pi / T)

def ideal_gas_chemical_potential(V, N, T):
    return T * jnp.log(N / V * thermal_deBroglie_wavelength(T)**3)

def find_chemical_potential_w_freecontinuum(energies, degeneracies, V, N, T, tol=1e-6, max_iter=100):
    mu_lower = ideal_gas_chemical_potential(V, N, T) - 100.0 * T
    mu_upper = 1.1*fermi_energy(V, N) + 100.0 * T

    def root_func(mu, args):
        by_state_occ = fermi_dirac_dist(energies, mu, T) * degeneracies
        # Apply bound mask
        by_state_occ = by_state_occ * bound_mask(energies)
        bound_occ = jnp.sum(by_state_occ)

        # finding N for free electrons
        free_occ = ((jnp.sqrt(2) * V * T**(3/2)) / (jnp.pi**2)) * jnp.sqrt(jnp.pi) / 2 * fermi_dirac_integral_half(mu/T)

        occ = bound_occ + free_occ
        return occ - N, {'state_occ': by_state_occ, 'free_occ': free_occ}

    mu_guess = mu_lower + T
    op = opt.Bisection(rtol=tol, atol=tol)
    opt_result = opt.root_find(root_func, op, y0 = mu_guess, options={'lower': mu_lower, 'upper': mu_upper}, args=None, has_aux=True)

    return opt_result.value, opt_result.aux

def find_chemical_potential_w_KSstates(energies, degeneracies, V, N, T, tol=1e-6, max_iter=100):
    # Using KS states only - therefore must include positive energy states
    mu_lower = jnp.min(energies) - 10.0 * T
    mu_upper = jnp.max(energies) + 10.0 * T # May need adjusted

    def root_func(mu, args):
        by_state_occ = fermi_dirac_dist(energies, mu, T) * degeneracies
        occ = jnp.sum(by_state_occ)
        return occ - N, {'state_occ': by_state_occ, 'free_occ': 0.0}

    mu_guess = energies[jnp.argmin(jnp.abs(jnp.cumsum(degeneracies) - N))]
    op = opt.Bisection(rtol=tol, atol=tol)
    opt_result = opt.root_find(root_func, op, y0 = mu_guess, options={'lower': mu_lower, 'upper': mu_upper}, args=None, has_aux=True)

    return opt_result.value, opt_result.aux

def find_free_chemical_potential(V, N, T, tol=1e-6):
    mu_lower = ideal_gas_chemical_potential(V, N, T) - 100.0 * T
    mu_upper = 1.1*fermi_energy(V, N) + 100.0 * T

    def root_func(mu, args):
        free_occ = ((jnp.sqrt(2) * V * T**(3/2)) / (jnp.pi**2)) * jnp.sqrt(jnp.pi) / 2 * fermi_dirac_integral_half(mu/T)
        return free_occ - N

    mu_guess = mu_lower + T
    op = opt.Bisection(rtol=tol, atol=tol)
    opt_result = opt.root_find(root_func, op, y0 = mu_guess, options={'lower': mu_lower, 'upper': mu_upper}, args=None)

    return opt_result.value

def free_entropy_integral(mu,T):
    def _integrand(x,mu,T):
        E = jnp.tan(jnp.pi*x/4.0)
        f = fermi_dirac_dist(E,mu,T)
        return jnp.sqrt(E)*(f*jnp.log(f)+(1-f)*jnp.log(1-f))
    return quad(_integrand,0.0,0.99,(mu,T))

def bound_entropy_calc(energies, degeneracies, mu, T):
    by_state_f = fermi_dirac_dist(energies, mu, T)
    # Apply bound mask
    by_state_f = by_state_f * bound_mask(energies)
    return -jnp.sum(degeneracies*(by_state_f*jnp.log(by_state_f)+(1-by_state_f)*jnp.log(1-by_state_f)))

