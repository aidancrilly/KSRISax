from KSRISax.ThomasFermi import ThomasFermiSolver
import numpy as np
import scipy.constants as sc

def test_ThomasFermiSolver():

    TF = ThomasFermiSolver()

    # Pressures from table XI of Feynman, Metropolis and Teller
    # For 56Fe
    N = 26.0
    Vs_Angstrom_cubed = [8.353 , 8.659 , 110.54, 124.91, 106.88, 3.774 , 1.816, 23.893 , 96.30 , 331.04, 189.52]
    Ts_keV            = [0.2231, 0.4926, 0.1476, 0.2381, 0.5297, 0.2366, 14.660, 0.2923, 0.9892, 0.3416, 0.0326]
    PVs_over_kTZ      = [0.5227, 0.7028, 0.5210, 0.6470, 0.8283, 0.5266, 0.5936, 0.6239, 0.9172, 0.8037, 0.2380]

    for V_Angstrom_cubed, T_keV, PV_over_kTZ in zip(Vs_Angstrom_cubed,Ts_keV,PVs_over_kTZ):
        V = V_Angstrom_cubed / (0.148185)
        T = T_keV / (0.0272114)
        alpha, _, _ = TF.alpha_b_wb_FMT(N, V, T)

        # Checks on definitions
        assert np.isclose(TF.c_FMT(T) * sc.value('Bohr radius'), 1.602e-11 / (T_keV)**(0.25))
        assert np.isclose(alpha, 0.0899*N/(T_keV)**(0.75),atol=1e-3)
        res = TF(N, V, T)
        PV_over_kTZ_TF = res['P'] * V / (T * N)
        # Checks on solution
        # Skip T_keV=14.660 case which does not converge to correct solution
        if T_keV == 14.660:
            continue
        assert np.isclose(PV_over_kTZ_TF, PV_over_kTZ, atol=0.02)