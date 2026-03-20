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
    Nextend: int = eqx.field(static=True, default=0)
    xc_ext: jnp.ndarray | None = None
    xb_ext: jnp.ndarray | None = None
    vol_ext: jnp.ndarray | None = None

class LinearGrid(Grid):

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int, Nextend: int = 0) -> "LinearGrid":
        xb = jnp.linspace(x_min, x_max, num_points + 1)
        xc = 0.5 * (xb[:-1] + xb[1:])
        vol = 4.0 / 3.0 * jnp.pi * (xb[1:]**3 - xb[:-1]**3)
        dx = xc[1] - xc[0]
        xc_ext = xb_ext = vol_ext = None
        if Nextend > 0:
            xb_ext = jnp.concatenate([xb, xb[-1] + jnp.arange(1, Nextend + 1) * dx])
            xc_ext = 0.5 * (xb_ext[:-1] + xb_ext[1:])
            vol_ext = 4.0 / 3.0 * jnp.pi * (xb_ext[1:]**3 - xb_ext[:-1]**3)
        return LinearGrid(xb=xb, xc=xc, vol=vol, Nx=num_points, dx=dx, log=False, log_spacing=jnp.nan,
                          Nextend=Nextend, xc_ext=xc_ext, xb_ext=xb_ext, vol_ext=vol_ext)


class LogarithmicGrid(Grid):

    @staticmethod
    def create(x_min: float, x_max: float, num_points: int, Nextend: int = 0) -> "LogarithmicGrid":
        log_spacing = jnp.log(x_max/x_min) / (num_points + 1)
        r0 = x_min*jnp.exp(0.5*log_spacing)
        xb = r0 * jnp.exp((jnp.arange(num_points+1)-0.5)*log_spacing)
        xc = r0 * jnp.exp(jnp.arange(num_points)*log_spacing)
        vol = 4.0 / 3.0 * jnp.pi * (xb[1:]**3 - xb[:-1]**3)
        dx = xc[1:]-xc[:-1]
        dx = jnp.concatenate([dx[:1],dx,dx[-1:]])
        xc_ext = xb_ext = vol_ext = None
        if Nextend > 0:
            xb_ext = r0 * jnp.exp((jnp.arange(num_points + Nextend + 1) - 0.5) * log_spacing)
            xc_ext = r0 * jnp.exp(jnp.arange(num_points + Nextend) * log_spacing)
            vol_ext = 4.0 / 3.0 * jnp.pi * (xb_ext[1:]**3 - xb_ext[:-1]**3)
        return LogarithmicGrid(xb=xb, xc=xc, vol=vol, Nx=num_points, dx=dx, log=True, log_spacing=log_spacing,
                               Nextend=Nextend, xc_ext=xc_ext, xb_ext=xb_ext, vol_ext=vol_ext)
