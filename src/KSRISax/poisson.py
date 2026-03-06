import jax
import jax.numpy as jnp
import lineax as lx
import equinox as eqx
from KSRISax.grid import Grid

class PoissonSolver(eqx.Module):
    grid: Grid
    V_gauge: float | None = None

    @jax.jit
    def solve(self, n):
        """Solve the Poisson equation for the Hartree potential given the electron density n."""
        # Right-hand side of Poisson equation: -4 * pi * n(r)
        rhs = - n * self.grid.vol

        # Construct the tridiagonal matrix
        if(self.grid.log):
            diag = - (self.grid.xb[1:]**2 / self.grid.dx[1:] + self.grid.xb[:-1]**2 / self.grid.dx[:-1])
            lower_diag = self.grid.xb[1:-1]**2 / self.grid.dx[1:-1]
            upper_diag = self.grid.xb[1:-1]**2 / self.grid.dx[1:-1]
            diag = diag.at[0].set(-self.grid.xb[1]**2 / self.grid.dx[1])
        else:
            diag = - (self.grid.xb[1:]**2 + self.grid.xb[:-1]**2) / self.grid.dx
            lower_diag = self.grid.xb[1:-1]**2 / self.grid.dx
            upper_diag = self.grid.xb[1:-1]**2 / self.grid.dx

        # BCs
        if self.V_gauge is not None:
            diag = diag.at[-1].set(1.0)
            lower_diag = lower_diag.at[-1].set(0.0)
            rhs = rhs.at[-1].set(self.V_gauge)

        Laplacian = lx.TridiagonalLinearOperator(diag, lower_diag, upper_diag)

        # Solve for V_H using a linear solver
        V_H = lx.linear_solve(Laplacian, rhs).value

        return V_H