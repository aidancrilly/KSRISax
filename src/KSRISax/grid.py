import jax
import jax.numpy as jnp
import equinox as eqx

class Grid(eqx.Module):
    xc: jnp.ndarray
    xb: jnp.ndarray
    vol: jnp.ndarray
    Nx: int = eqx.field(static=True)
    dx: float

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int) -> "Grid":
        xb = jnp.linspace(x_min, x_max, num_points + 1)
        xc = 0.5 * (xb[:-1] + xb[1:])
        vol = 4.0 / 3.0 * jnp.pi * (xb[1:]**3 - xb[:-1]**3)
        dx = xb[1] - xb[0]
        return Grid(xb=xb, xc=xc, vol=vol, Nx=num_points, dx=dx)
