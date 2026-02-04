import jax
import jax.scipy.linalg as jla
import jax.numpy as jnp
from KSRISax.grid import Grid
import equinox as eqx

class KohnShamSolver(eqx.Module):
    grid: Grid

    @staticmethod
    def cholesky_solve(A,B):
        """ 
        basically a generalized eigenvalue/vec solver, for the form Ax = lambda B x
        performing cholesky decomposition, uses the eigh function
        
        background:
        B = L L^T --> Ax = lambda L L^T x --> L^-1 A x = lambda L^T x = lambda y
        y defined to be y = L^T x 
        L^-1 A L^-T y = A_tilde y = lambda y --> can use eigh function from jax to obtain y
        obtain x through x = L^-T y

        inputs:
        A,B: the matrices mentioned above

        outputs:
        eigvals, eigenvecs of the original equation (in that order)
        
        """
        L = jla.cholesky(B, lower=True)

        # can use jla.inv but if a matrix is triangular, can use solve_triangular
        # to basically solve Lx = I --> x = L^-1 (more computationally efficient)
        L_inv = jla.solve_triangular(L,jnp.eye(L.shape[0]), lower = True)

        A_tilde = L_inv @ A @ L_inv.T

        eigvals, eigvecs_tilde = jla.eigh(A_tilde)

        eigvecs = L_inv.T @ eigvecs_tilde

        return eigvals, eigvecs

    @jax.jit
    def EigenSolve(self, l, V_ext, V_H, V_xc):

        V_centrifugal = jnp.where(self.grid.xc > 1e-10, l * (l + 1) / (2.0 * self.grid.xc**2), 0.0)

        Vdiag = V_ext + V_H + V_xc + V_centrifugal

        # Creating matrices for Numerov
        A_lower = jnp.ones(self.grid.Nx-1)
        A_mid = -2 * jnp.ones(self.grid.Nx)
        A_upper = jnp.ones(self.grid.Nx-1)

        B_lower = jnp.ones(self.grid.Nx-1)
        B_mid = 10 * jnp.ones(self.grid.Nx)
        B_upper = jnp.ones(self.grid.Nx-1)

        # Impose boundary condition at r=0
        # Ghost cell u_-1 = (-1)^(l+1) * u_1 --> modifies first row of A and B
        s = (-1)**(l+1)
        A_mid = A_mid.at[0].set(s - 2)
        B_mid = B_mid.at[0].set(10 + s)

        A = (jnp.diag(A_lower, k = -1) + jnp.diag(A_mid, k = 0) + jnp.diag(A_upper, k = 1)) / self.grid.dx ** 2
        B = (jnp.diag(B_lower, k = -1) + jnp.diag(B_mid, k = 0) + jnp.diag(B_upper, k = 1)) / 12

        # constructing Hamiltonian matrix
        H = -1/2 * A + B @ jnp.diag(Vdiag)

        # obtaining eigenvals / vecs from cholesky decomposition
        eigvals, eigvecs = self.cholesky_solve(H,B)

        # Normalise eigenvectors
        # norm_factors = jnp.sqrt(jnp.sum((eigvecs**2) * self.grid.vol[:,jnp.newaxis], axis=0))
        # eigvecs = eigvecs / norm_factors

        return eigvals, eigvecs
