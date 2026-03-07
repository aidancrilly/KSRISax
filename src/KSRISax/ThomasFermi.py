import equinox as eqx
import numpy as np
from scipy.integrate import solve_bvp
from FDint_JAX import fermi_dirac_integral_half, fermi_dirac_integral_three_half
from .chem import thermal_deBroglie_wavelength

gamma5h = 3 * np.sqrt(np.pi) / 4

class ThomasFermiSolver(eqx.Module):
    n_bvp_points : int = 200
    nw_integral : int = 400
    eps : float = 1e-8
    max_nodes : int = int(2e5)
    tol : float = 1e-6

    @staticmethod
    def I(x):
        return fermi_dirac_integral_half(x) * (0.5 * np.sqrt(np.pi))

    def solve_beta_bvp(self, a, wb):
        """
        Solve:
            d/dw ( (beta'(w))/w ) = (w^3 / 2) I( 2 beta(w) / w^2 )
        with BCs:
            beta(0) = a
            beta'(wb) = 2 beta(wb) / wb

        Uses solve_bvp on [eps, wb] to avoid division by zero at w=0.
        """

        # ---- ODE as first-order system ----
        # Let y0 = beta, y1 = beta'
        #
        # Given: d/dw (y1 / w) = (w^3 / 2) I( 2 y0 / w^2 )
        # Expand: (y1/w)' = y1'/w - y1/w^2
        # So: y1'/w - y1/w^2 = (w^3 / 2) I(...)
        # => y1' = y1/w + (w^4 / 2) I(...)
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

        return sol.sol

    def c_FMT(self, T):
        return np.sqrt(np.sqrt(np.pi) * thermal_deBroglie_wavelength(T) / 8.0)

    def alpha_b_wb_FMT(self, N, V, T):
        a = (3 * V / (4 * np.pi))**(1/3)
        c = self.c_FMT(T)
        b = a / c
        return N / (T * c), b, np.sqrt(2*b)

    def __call__(self, N, V, T):
        alpha, b, wb = self.alpha_b_wb_FMT(N, V, T)
        beta_sol = self.solve_beta_bvp(alpha, wb)

        w_integral = np.linspace(self.eps,wb,self.nw_integral)
        beta = beta_sol(w_integral)[0]

        # Chemical potential
        mu = T * (beta[-1] / b)

        # Potential energy: Epot = -(1/2) U_nV + (1/2) UeN
        # where U_nV = ∫ 4πr² n(r) V(r) dr = 2Uee + UeN
        # and   UeN  = ∫ 4πr² n(r) (N/r) dr
        # Using s = w²/2, n(r) = 2/λ³ I(β/s), V(r) = Tβ/s - μ, N/r = 2N/(cw²)
        c = self.c_FMT(T)
        lam = thermal_deBroglie_wavelength(T)
        arg = 2.0 * beta / (w_integral**2)
        I_vals = self.I(arg)

        # U_nV = (2πc³/λ³) ∫ w⁵ I(2β/w²)(T·2β/w² - μ) dw
        V_r = T * arg - mu
        integrand_nV = w_integral**5 * I_vals * V_r
        U_nV = (2.0 * np.pi * c**3 / lam**3) * np.trapezoid(integrand_nV, w_integral)

        # UeN = (4πNc²/λ³) ∫ w³ I(2β/w²) dw
        integrand_eN = w_integral**3 * I_vals
        UeN = (4.0 * np.pi * N * c**2 / lam**3) * np.trapezoid(integrand_eN, w_integral)

        Epot = -0.5 * U_nV + 0.5 * UeN

        P = (2.0 / 9.0) * (N / V) * T * (b**3 / alpha) * fermi_dirac_integral_three_half(beta[-1] / b) * gamma5h
        U = 1.5 * P + 0.5 * Epot / V
        return {'U' : U, 'P' : P, 'mu' : mu, 'beta(0)' : beta[0]}

