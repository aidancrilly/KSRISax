import jax.numpy as jnp
import equinox as eqx

class Grid(eqx.Module):
    xc: jnp.ndarray
    xb: jnp.ndarray
    vol: jnp.ndarray
    Nx: int = eqx.field(static=True)
    dx: float
    log: bool = eqx.field(static=True)
    log_spacing: float
class LinearGrid(Grid):

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int) -> "LinearGrid":
        xb = jnp.linspace(x_min, x_max, num_points + 1)
        xc = 0.5 * (xb[:-1] + xb[1:])
        vol = 4.0 / 3.0 * jnp.pi * (xb[1:]**3 - xb[:-1]**3)
        dx = xc[1] - xc[0]
        return LinearGrid(xb=xb, xc=xc, vol=vol, Nx=num_points, dx=dx, log=False, log_spacing=jnp.nan)


class LogarithmicGrid(Grid):

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int) -> "LogarithmicGrid":
        log_spacing = jnp.log(x_max/x_min) / (num_points + 1)
        r0 = x_min*jnp.exp(0.5*log_spacing)
        xb = r0 * jnp.exp((jnp.arange(num_points+1)-0.5)*log_spacing)
        xc = r0 * jnp.exp(jnp.arange(num_points)*log_spacing)
        vol = 4.0 / 3.0 * jnp.pi * (xb[1:]**3 - xb[:-1]**3)
        dx = xc[1:]-xc[:-1]
        dx = jnp.concatenate([dx[:1],dx,dx[-1:]])
        return LogarithmicGrid(xb=xb, xc=xc, vol=vol, Nx=num_points, dx=dx, log=True, log_spacing=log_spacing)
