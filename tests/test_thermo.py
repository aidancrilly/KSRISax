from KSRISax.thermo import Thermodynamics
import jax.numpy as jnp
import jax

therm = Thermodynamics(
    N=13.0,
    rmin=1e-5,
    Nr=200,
    SCF_max_iterations=200,
    SCF_L_max=3,
    SCF_convergence_threshold=1e-6,
    SCF_damping=0.1
    )

Ts = jnp.logspace(-1,0,20)

Rs = [30.0,10.0,5.0,2.0]

import matplotlib.pyplot as plt

n_guess = jnp.zeros(therm.Nr)

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

for R in Rs:
    V = 4 * jnp.pi * (R)**3 / 3

    Us, Ps, Zs, mus = [], [], [], []
    for i,T in enumerate(Ts):
        print(f'T = {T} Ha')
        therm_out = therm.nograd_call(V,T,n_guess)
        # n_guess = therm_out['n_SCF']

        Us.append(therm_out['U'])
        Ps.append(therm_out['P'])
        Zs.append(therm_out['Z'])
        mus.append(therm_out['mu'])

    ax1.semilogx(Ts,Us,label=f'R = {R:.1f}')
    ax2.semilogx(Ts,Ps)
    ax3.semilogx(Ts,Zs)
    ax4.semilogx(Ts,mus)

ax1.legend(frameon=False)

fig.suptitle('Al DFT + LDA X')

ax1.set_ylabel('U')
ax2.set_ylabel('P')
ax3.set_ylabel('Z')
ax4.set_ylabel('mu')


fig.tight_layout()

# Vs = 4 * jnp.pi / 3.0 * jnp.linspace(2.0,20.0,40)

# T = 1e2/(2*13.6)

# import matplotlib.pyplot as plt

# n_guess = jnp.zeros(therm.Nr)

# plt.figure(dpi=200)
# for i,V in enumerate(Vs):
#     print(f'T = {T} Ha')
#     therm_out = jax.jit(therm)(V,T,n_guess)
#     # n_guess = therm_out['n_SCF']
#     mask = therm_out['energies'] < 0.0
#     print(therm_out['energies'][mask])
#     print(therm_out['occupancies'][mask])
#     print(therm_out['mu'])

#     if i == 0:
#         plt.semilogx(V,therm_out['U_total'], 'rx',label='Utot')
#         plt.semilogx(V,therm_out['U_free'], 'bx',label='Uf')
#         plt.semilogx(V,therm_out['U_bound'], 'gx',label='Ub')
#     else:
#         plt.semilogx(V,therm_out['U_total'], 'rx')
#         plt.semilogx(V,therm_out['U_free'], 'bx')
#         plt.semilogx(V,therm_out['U_bound'], 'gx')

# plt.title('Al DFT + LDA X')
# plt.xlabel('Ion Sphere Volume (a0)^3')
# plt.ylabel('Internal energy per atom (Ha)')
# plt.legend(frameon=False)
# plt.tight_layout()

plt.show()
