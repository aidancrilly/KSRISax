#%%
from KSRISax.thermo import DFTThermodynamics
from KSRISax.ThomasFermi import ThomasFermiSolver
import jax.numpy as jnp
import jax
import time

jax.config.update('jax_enable_x64', True)

N = 13

therm = DFTThermodynamics(
    N=N,
    rmin=1e-5,
    Nr=200,
    SCF_max_iterations=200,
    SCF_L_max=3,
    SCF_convergence_threshold=1e-6,
    SCF_damping=0.1,
    verbose=False
    )

TF = ThomasFermiSolver(
    method = 'relaxation_JAX',
    tol = 1e-3,
    eps = 1e-6
    )



Ts = jnp.logspace(-1,2,50)

Rs = [30.0,10.0,5.0,2.0]

import matplotlib.pyplot as plt

n_guess = jnp.zeros(therm.Nr)

#%%

Us, Ps, Zs, mus = [], [], [], []
U_TFs, P_TFs, Z_TFs, mu_TFs = [], [], [], []

for R in Rs:
    V = 4 * jnp.pi * (R)**3 / 3

    for i,T in enumerate(Ts):
        print(f'T = {T} Ha')
        print('DFT calc')
        start = time.time()
        therm_out = therm.nograd_call(T,V,n_guess)
        print(f'Runtime: {time.time()-start}')

        Us.append(therm_out['U'])
        Ps.append(therm_out['P'])
        Zs.append(therm_out['Z'])
        mus.append(therm_out['mu'])

        print('TF calc')
        start = time.time()
        TF_out = TF(N, V, T)
        print(f'Runtime: {time.time()-start}')
        U_TFs.append(TF_out['U'])
        P_TFs.append(TF_out['P'])
        Z_TFs.append(TF_out['Z'])
        mu_TFs.append(TF_out['mu'])

#%%

fig = plt.figure(dpi=200)
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

Us, Ps, Zs, mus = jnp.array(Us).reshape(len(Rs),-1), jnp.array(Ps).reshape(len(Rs),-1), jnp.array(Zs).reshape(len(Rs),-1), jnp.array(mus).reshape(len(Rs),-1)
U_TFs, P_TFs, Z_TFs, mu_TFs = jnp.array(U_TFs).reshape(len(Rs),-1), jnp.array(P_TFs).reshape(len(Rs),-1), jnp.array(Z_TFs).reshape(len(Rs),-1), jnp.array(mu_TFs).reshape(len(Rs),-1)

for i,R in enumerate(Rs):
    V = 4 * jnp.pi * (R)**3 / 3

    ax1.loglog(Ts,(Us[i,:]-Us[i,0])*V/(N*Ts),label=f'R = {R:.1f}')
    ax2.loglog(Ts,Ps[i,:]*V/(N*Ts))
    ax3.semilogx(Ts,Zs[i,:])
    ax4.semilogx(Ts,mus[i,:])

    ax1.semilogx(Ts,(U_TFs[i,:]-U_TFs[i,0])*V/(N*Ts),ls='--',c='k')
    ax2.semilogx(Ts,P_TFs[i,:]*V/(N*Ts),ls='--',c='k')
    ax3.semilogx(Ts,Z_TFs[i,:],ls='--',c='k')
    ax4.semilogx(Ts,mu_TFs[i,:],ls='--',c='k')

ax1.legend(frameon=False)

fig.suptitle('Al DFT + LDA X')

ax1.set_ylabel('U V / (Z kT)')
ax2.set_ylabel('P V / (Z kT)')
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

# %%
