import jax
import jax.numpy as jnp
from KSRISax.grid import Grid
import equinox as eqx

class KohnShamSolver(eqx.Module):
    grid: Grid

    @jax.jit
    def EigenSolve(self, l, V_ext, V_H, V_xc):

        V_centrifugal = jnp.where(self.grid.xc > 1e-10, l * (l + 1) / (2.0 * self.grid.xc**2), 0.0)

        Vdiag = V_ext + V_H + V_xc + V_centrifugal
        
        KE_diag = - 0.5 * jnp.full(self.grid.Nx, -2.0) / (self.grid.dx**2)
        KE_offdiag = - 0.5 * jnp.full(self.grid.Nx - 1, 1.0) / (self.grid.dx**2)

        # Impose boundary condition at r=0
        # Vdiag = Vdiag.at[0].set(0)  
        # KE_diag = KE_diag.at[0].set(1.0)
        # KE_offdiag = KE_offdiag.at[0].set(0.0)

        H = jnp.diag(KE_diag) + jnp.diag(KE_offdiag, 1) + jnp.diag(KE_offdiag, -1) + jnp.diag(Vdiag)
        eigvals, eigvecs = jnp.linalg.eigh(H, UPLO='U')

        # Normalise eigenvectors
        norm_factors = jnp.sqrt(jnp.sum((eigvecs**2) * self.grid.vol[:,jnp.newaxis], axis=0))
        eigvecs = eigvecs / norm_factors

        return eigvals, eigvecs
