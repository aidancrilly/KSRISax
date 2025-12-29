
import jax.numpy as jnp
import optimistix as opt
import jax
from KSRISax.quad import quad

def fermi_dirac_dist(energy, mu, T):
    return 1.0 / (1.0 + jnp.exp((energy - mu) / T))

def find_chemical_potential_w_KSstates(energies, degeneracies, V, N, T, tol=1e-6, max_iter=100):
    mu_lower = jnp.min(energies) - 10.0 * T
    mu_upper = jnp.max(energies) + 10.0 * T # May need adjusted

    def root_func(mu, args):
        by_state_occ = fermi_dirac_dist(energies, mu, T) * degeneracies
        occ = jnp.sum(by_state_occ)
        return occ - N, by_state_occ

    mu_guess = energies[jnp.argmin(jnp.abs(jnp.cumsum(degeneracies) - N))]
    op = opt.Bisection(rtol=tol, atol=tol)
    opt_result = opt.root_find(root_func, op, y0 = mu_guess, options={'lower': mu_lower, 'upper': mu_upper}, args=None, has_aux=True)
    
    return opt_result.value, opt_result.aux

def fso_transform(x):
    return jnp.tan(jnp.pi * x / 2.0)

def free_state_occupation_integrand(x,mu,T,V):
    E = fso_transform(x)
    dEdx = jax.grad(fso_transform)(x)
    return (V / (jnp.sqrt(2) * jnp.pi**2)) * dEdx * jnp.sqrt(E) * fermi_dirac_dist(E, mu, T)

def find_chemical_potential_w_freecontinuum(energies, degeneracies, V, N, T, tol=1e-6, max_iter=100):
    mu_lower = jnp.min(energies) - 10.0 * T
    mu_upper = 10.0 * T # May need adjusted

    def root_func(mu, args):
        by_state_occ = fermi_dirac_dist(energies, mu, T) * degeneracies
        bound_occ = jnp.sum(by_state_occ)
        free_occ = quad(free_state_occupation_integrand, 0.0, 0.99, [mu, T, V])
        occ = bound_occ + free_occ
        return occ - N, by_state_occ

    mu_guess = energies[jnp.argmin(jnp.abs(jnp.cumsum(degeneracies) - N))]
    op = opt.Bisection(rtol=tol, atol=tol)
    opt_result = opt.root_find(root_func, op, y0 = mu_guess, options={'lower': mu_lower, 'upper': mu_upper}, args=None, has_aux=True)
    
    return opt_result.value, opt_result.aux