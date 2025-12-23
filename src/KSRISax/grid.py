import jax
import jax.numpy as jnp
import equinox as eqx

class Grid(eqx.Module):
    x: jnp.ndarray
    Nx: int = eqx.field(static=True)
    dx: float

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int) -> "Grid":
        x = jnp.linspace(x_min, x_max, num_points)
        dx = x[1] - x[0]
        return Grid(x=x, Nx=num_points, dx=dx)
