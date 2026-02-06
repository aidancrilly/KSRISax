# KSRISax

Kohn Sham Radial Ion Sphere in JAX

Work in progress, currently implemented:

- Matrix Numerov solver for KS equation
- Finite volume Poisson solver
- Chemical potential root finding with continuum states
- Self consistent field solver w/o exchange-correlation functional

All differentiable allowing gradient-based methods for iterative solve!

Many thanks to Hirofumi Muramoto for his contributions towards this project


### Installation notes

Perform local editable install:

```bash
pip install -e .[dev]
```

And edit the ```__init__.py``` file in the FDint_JAX package, turning _VJP_VERSION to False.