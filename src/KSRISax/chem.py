
import jax.numpy as jnp
import optimistix as opt
import jax
from FDint_JAX import fermi_dirac_integral_half

def fermi_dirac_dist(energy, mu, T):
    occ = 1.0 / (1.0 + jnp.exp((energy - mu) / T))
    return occ

def bound_mask(E):
    return jnp.where(E < 0.0, 1.0, 0.0)

def find_chemical_potential_w_freecontinuum(energies, degeneracies, V, N, T, tol=1e-6, max_iter=100):
    mu_lower = jnp.min(energies) - 100.0 * T
    mu_upper = 10.0 * T # May need adjusted

    def root_func(mu, args):
        by_state_occ = fermi_dirac_dist(energies, mu, T) * degeneracies
        # Apply bound mask
        by_state_occ = by_state_occ * bound_mask(energies)
        bound_occ = jnp.sum(by_state_occ)

        # finding N for free electrons
        free_occ = ((jnp.sqrt(2) * V * T**(3/2)) / (jnp.pi**2)) * jnp.sqrt(jnp.pi) / 2 * fermi_dirac_integral_half(mu/T)

        occ = bound_occ + free_occ
        return occ - N, {'state_occ': by_state_occ, 'free_occ': free_occ}

    mu_guess = mu_lower + T # energies[jnp.argmin(jnp.abs(jnp.cumsum(degeneracies) - N))]
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