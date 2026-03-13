# KSRISax

Kohn Sham Radial Ion Sphere in JAX

Work in progress, currently implemented:

- Matrix Numerov solver for KS equation (linear and logarithmic grids)
- Finite volume Poisson solver
- Chemical potential root finding with continuum states
- Self consistent field solver with Dirac UEG exchange functional
- Temperature-dependent Thomas-Fermi solver ala Feynman, Metropolis and Teller for benchmarking
- Thermodynamics module in development

All differentiable allowing gradient-based methods for iterative solves and trainable models!

Many thanks to Hirofumi Muramoto for his contributions towards this project

### Installation notes

Perform local editable install:

```bash
pip install -e .[dev]
```

