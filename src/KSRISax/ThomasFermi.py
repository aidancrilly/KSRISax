import equinox as eqx
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_bvp
import optimistix as optx
from FDint_JAX import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from .chem import thermal_deBroglie_wavelength

gamma3h = 0.5 * np.sqrt(np.pi)
gamma5h = 3 * np.sqrt(np.pi) / 4


def approx_fermi_dirac_integral_half(x):
    # Cheap version
    return jnp.where(x < 0, jnp.exp(x), 4 / (3 * jnp.sqrt(jnp.pi)) * x**1.5)

class ThomasFermiSolver(eqx.Module):
    method : str
    n_bvp_points : int = 200
    nw_integral : int = 400
    eps : float = 1e-8
    max_nodes : int = int(2e5)
    tol : float = 1e-3

    @staticmethod
    def I(x):
        return fermi_dirac_integral_half(x) * gamma3h

    def solve_beta_bvp(self, a, wb, w_integral):
        """
        Solve:
            d/dw ( (beta'(w))/w ) = (w^3 / 2) I( 2 beta(w) / w^2 )
        with BCs:
            beta(0) = a
            beta'(wb) = 2 beta(wb) / wb

        # ---- ODE as first-order system ----
            # Let y0 = beta, y1 = beta'
            #
            # Given: d/dw (y1 / w) = (w^3 / 2) I( 2 y0 / w^2 )
            # Expand: (y1/w)' = y1'/w - y1/w^2
            # So: y1'/w - y1/w^2 = (w^3 / 2) I(...)
            # => y1' = y1/w + (w^4 / 2) I(...)

        Uses solve_bvp on [eps, wb] to avoid division by zero at w=0.
        """
        if self.method == 'scipy':

            def fun(w, y):
                beta = y[0]
                dbeta = y[1]

                # avoid division issues
                w_safe = np.maximum(w, self.eps)

                arg = 2.0 * beta / (w_safe**2)
                rhs = (w_safe**4) * self.I(arg) / 2.0

                d_beta = dbeta
                d_dbeta = (dbeta / w_safe) + rhs
                return np.vstack((d_beta, d_dbeta))

            # ---- Boundary conditions ----
            # At w=0: beta(0)=a  -> enforce at w=eps: beta(eps)=a
            # At w=b: beta'(wb) - 2 beta(wb)/wb = 0
            def bc(ya, yb):
                return np.array([
                    ya[0] - a,
                    yb[1] - 2.0 * yb[0] / wb
                ])

            # ---- Mesh & initial guess ----
            w = np.linspace(self.eps, wb, self.n_bvp_points)

            # A reasonable guess that satisfies beta(eps)=a and roughly respects the slope condition:
            # try beta ~ a*(w/eps)^2 near 0 is too aggressive; instead use a gentle quadratic anchored at a.
            # We'll pick beta ~ a, dbeta ~ (2a/wb) w as a mild slope.
            beta_guess = a * np.ones_like(w)
            dbeta_guess = (2.0 * a / wb) * (w / wb)
            y_guess = np.vstack((beta_guess, dbeta_guess))

            sol = solve_bvp(fun, bc, w, y_guess, tol=self.tol, max_nodes=self.max_nodes)

            if not sol.success:
                raise RuntimeError(f"solve_bvp failed: {sol.message}")

            return sol.sol(w_integral)

        elif self.method == 'relaxation_JAX':
            w_jnp = jnp.array(w_integral)
            return eqx.filter_jit(self._relax_solve)(a, wb, w_jnp)

    def _relax_solve(self, a, wb, w_integral):

        def RHS(y, args):
            w_mid = 0.5 * (args['w'][:-1] + args['w'][1:])
            y_mid = 0.5 * (y['y'][:-1] + y['y'][1:])
            dydw_mid = 0.5 * (y['dydw'][:-1] + y['dydw'][1:])

            arg = 2.0 * y_mid / (w_mid**2)
            d_dbeta = dydw_mid / w_mid + (w_mid**4) * args['FD_onehalf'](arg) * gamma3h / 2.0

            rhs = {
                'y' : dydw_mid,
                'dydw' : d_dbeta
            }
            return rhs

        def error(y, args):
            g = RHS(y, args)

            e = {
                'y' : jnp.concatenate(
                    [
                    y['y'][:1] - a,
                    y['y'][1:] - y['y'][:-1] - (args['w'][1:]-args['w'][:-1])*g['y'],
                    jnp.zeros(1)
                    ]
                ),
                'dydw' : jnp.concatenate(
                    [
                    np.zeros(1),
                    y['dydw'][1:] - y['dydw'][:-1] - (args['w'][1:]-args['w'][:-1])*g['dydw'],
                    y['dydw'][-1:] - 2 * y['y'][-1:] / wb
                    ]
                )
            }
            return e

        solver = optx.Dogleg(rtol = self.tol, atol = self.tol, norm = optx.max_norm)

        y0 = {
            'y' : a * jnp.ones_like(w_integral),
            'dydw' : (2.0 * a / wb) * (w_integral / wb)
        }

        args= {
            'w' : w_integral,
            'FD_onehalf' : fermi_dirac_integral_half
        }

        sol = optx.least_squares(error, solver, y0, args)

        return sol.value['y'], sol.value['dydw']

    def c_FMT(self, T):
        return np.sqrt(np.sqrt(np.pi) * thermal_deBroglie_wavelength(T) / 8.0)

    def alpha_b_wb_FMT(self, N, V, T):
        a = (3 * V / (4 * np.pi))**(1/3)
        c = self.c_FMT(T)
        b = a / c
        return N / (T * c), b, np.sqrt(2*b)

    def __call__(self, N, V, T):
        alpha, b, wb = self.alpha_b_wb_FMT(N, V, T)
        w_integral = np.linspace(self.eps,wb,self.nw_integral)
        beta_sol = self.solve_beta_bvp(alpha, wb, w_integral)

        beta = beta_sol[0]

        # Chemical potential
        mu = T * (beta[-1] / b)

        # Potential energy: Epot = (1/2) E_nV + (1/2) EeN
        # where E_nV = -∫ 4πr² n(r) V(r) dr = 2Eee + EeN
        # and   EeN  = -∫ 4πr² n(r) (N/r) dr
        # Using s = w²/2, n(r) = 2/λ³ I(β/s), V(r) = Tβ/s - μ, N/r = 2N/(cw²)
        c = self.c_FMT(T)
        lam = thermal_deBroglie_wavelength(T)
        arg = 2.0 * beta / (w_integral**2)
        I_vals = self.I(arg) / gamma3h

        # E_nV = -(2πc³/λ³) ∫ w⁵ I(2β/w²)(T·2β/w² - μ) dw
        V_r = T * arg - mu
        integrand_nV = w_integral**5 * I_vals * V_r
        E_nV = - (2.0 * np.pi * c**3 / lam**3) * np.trapezoid(integrand_nV, w_integral)

        # EeN = -(4πNc²/λ³) ∫ w³ I(2β/w²) dw
        integrand_eN = w_integral**3 * I_vals
        EeN = - (4.0 * np.pi * N * c**2 / lam**3) * np.trapezoid(integrand_eN, w_integral)

        Epot = 0.5 * E_nV + 0.5 * EeN

        P = (2.0 / 9.0) * (N / V) * T * (b**3 / alpha) * fermi_dirac_integral_three_half(beta[-1] / b) * gamma5h
        U = 1.5 * P + Epot / V
        Z = V * (2 / lam**3) * I_vals[-1]

        return {
            'U' : U,
            'P' : P,
            'mu' : mu,
            'Z' : Z,
            'Epot' : Epot,
            'beta(0)' : beta[0]
            }

