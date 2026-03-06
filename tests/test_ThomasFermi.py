from KSRISax.ThomasFermi import ThomasFermiSolver
import numpy as np
import scipy.constants as sc
import jax

jax.config.update('jax_enable_x64', True)

def test_ThomasFermiSolver():

    TF = ThomasFermiSolver()

    # Pressures, b and beta boundary values from table XI of Feynman, Metropolis and Teller
    # For 56Fe
    N = 26.0
    Vs_Angstrom_cubed  = [8.353 , 8.659 , 110.54, 124.91, 106.88, 3.774 , 1.816 , 23.893 , 96.30  , 331.04  , 189.52 ]
    Ts_keV             = [0.2231, 0.4926, 0.1476, 0.2381, 0.5297, 0.2366, 14.660, 0.2923 , 0.9892 , 0.3416  , 0.0326 ]
    PVs_over_kTZ       = [0.5227, 0.7028, 0.5210, 0.6470, 0.8283, 0.5266, 0.5936, 0.6239 , 0.9172 , 0.8037  , 0.2380 ]
    alphas             = [7.2021, 3.9774, 9.8160, 6.8602, 3.7659, 6.8918, 0.3121, 5.8808 , 2.3570 , 5.2326  , 30.3768]
    bs                 = [5.4   , 6.6612, 11.52 , 13.52 , 15.68 , 4.205 , 9.245 , 8.2012 , 17.7012, 20.48   , 9.4612 ]
    beta_boundary_vals = [-13.5 ,-22.8914,-51.62,-68.9520,-92.512,-7.4536,-61.2944,-31.0828,-117.3593,-131.072,-33.4928]

    for V_Angstrom_cubed, T_keV, PV_over_kTZ, alpha_ref, b_ref, beta_boundary_ref in zip(
        Vs_Angstrom_cubed, Ts_keV, PVs_over_kTZ, alphas, bs, beta_boundary_vals
    ):
        V = V_Angstrom_cubed / (0.148185)
        T = T_keV / (0.0272114)
        alpha, b, _ = TF.alpha_b_wb_FMT(N, V, T)

        # Checks on definitions
        assert np.isclose(TF.c_FMT(T) * sc.value('Bohr radius'), 1.602e-11 / (T_keV)**(0.25))
        assert np.isclose(alpha, 0.0899*N/(T_keV)**(0.75),atol=1e-3)
        # Check values from table XI
        assert np.isclose(alpha, alpha_ref, rtol=1e-3)
        assert np.isclose(b, b_ref, rtol=1e-3)
        res = TF(N, V, T)
        PV_over_kTZ_TF = res['P'] * V / (T * N)
        beta_boundary_TF = res['mu'] / T
        # Checks on solution
        assert np.isclose(beta_boundary_TF, beta_boundary_ref, rtol=1e-2)
        assert np.isclose(PV_over_kTZ_TF, PV_over_kTZ, atol=0.02)
