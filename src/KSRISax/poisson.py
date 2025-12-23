import jax
import jax.numpy as jnp
import lineax as lx
import equinox as eqx
from KSRISax.grid import Grid

class PoissonSolver(eqx.Module):
    grid: Grid

    @jax.jit
    def solve(self, n, V_gauge = 0.0):
        """Solve the Poisson equation for the Hartree potential given the electron density n."""
        # Right-hand side of Poisson equation: -4 * pi * n(r)
        rhs = - n * self.grid.vol

        # Construct the tridiagonal matrix
        diag = - (self.grid.xb[1:]**2 + self.grid.xb[:-1]**2) / self.grid.dx
        lower_diag = self.grid.xb[1:-1]**2 / self.grid.dx
        upper_diag = self.grid.xb[1:-1]**2 / self.grid.dx

        # BCs
        diag = diag.at[-1].set(1.0)
        lower_diag = lower_diag.at[-1].set(0.0)
        rhs = rhs.at[-1].set(V_gauge)

        Laplacian = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)

        # Solve for V_H using a linear solver
        V_H = lx.linear_solve(Laplacian, rhs).value

        return V_H