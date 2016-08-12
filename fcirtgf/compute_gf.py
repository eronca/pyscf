import numpy as np
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.fci

# Apply Hamiltonian on a vector
def apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, vec):

    nocc = ncore + ncas

    eri_cas = pyscf.fci.direct_spin1.absorb_h1e(h1eff[ncore:nocc,ncore:nocc].copy(), eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc], ncas, (nalpha, nbeta), 0.5)
    temp = pyscf.fci.direct_spin1.contract_2e(eri_cas, vec, ncas, (nalpha, nbeta) )

    ham_vec = temp - e_zero*vec

    return ham_vec

# Compute CASSCF Green's function matrix element
def gf_calculation(Re_psi_t, Im_psi_t, Re_phi_t_rem, Im_phi_t_rem, Re_phi_t_add, Im_phi_t_add, init_rem_norm, init_add_norm, delta_t, e_zero, p, q, casscf, h1eff, eri):

    # Get information about CASSCF
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncore + ncas

    #Propagate a_i|psi_0> for removal part
    nalpha = nelecas[0]-1
    nbeta = nelecas[1]

    Re_ket_k1 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_phi_t_rem)
    Im_ket_k1 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_phi_t_rem)

    Re_temp = Re_phi_t_rem + 0.5*Re_ket_k1
    Im_temp = Im_phi_t_rem + 0.5*Im_ket_k1

    Re_ket_k2 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)
    Im_ket_k2 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_temp = Re_phi_t_rem + 0.5*Re_ket_k2
    Im_temp = Im_phi_t_rem + 0.5*Im_ket_k2

    Re_ket_k3 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)        
    Im_ket_k3 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_temp = Re_phi_t_rem + Re_ket_k3
    Im_temp = Im_phi_t_rem + Im_ket_k3

    Re_ket_k4 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)              
    Im_ket_k4 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_phi_tdt_rem = Re_phi_t_rem + Re_ket_k1/6.0 + Re_ket_k2/3.0 + Re_ket_k3/3.0 + Re_ket_k4/6.0
    Im_phi_tdt_rem = Im_phi_t_rem + Im_ket_k1/6.0 + Im_ket_k2/3.0 + Im_ket_k3/3.0 + Im_ket_k4/6.0

    norm_Re_phi_tdt_rem = np.linalg.norm(Re_phi_tdt_rem)
    norm_Im_phi_tdt_rem = np.linalg.norm(Im_phi_tdt_rem)
    norm_phi_tdt_rem = np.sqrt(pow(norm_Re_phi_tdt_rem,2.0)+pow(norm_Im_phi_tdt_rem,2.0))

    Re_phi_tdt_rem *= init_rem_norm/norm_phi_tdt_rem
    Im_phi_tdt_rem *= init_rem_norm/norm_phi_tdt_rem

    #Build the removal part of the GF
    p_Re_psi_tdt_rem = pyscf.fci.addons.des_a(Re_psi_t, ncas, nelecas, p)
    p_Im_psi_tdt_rem = pyscf.fci.addons.des_a(Im_psi_t, ncas, nelecas, p)

    g_imag_rem = 2.0*(np.dot(np.ravel(Im_phi_tdt_rem), np.ravel(p_Re_psi_tdt_rem)) + np.dot(np.ravel(Re_phi_tdt_rem), np.ravel(p_Im_psi_tdt_rem)))
    g_real_rem = 2.0*(np.dot(np.ravel(Re_phi_tdt_rem), np.ravel(p_Re_psi_tdt_rem)) - np.dot(np.ravel(Im_phi_tdt_rem), np.ravel(p_Im_psi_tdt_rem)))

    #Propagate a_i^+|psi_0> for addition part
    nalpha = nelecas[0]+1
    nbeta = nelecas[1]

    Re_ket_k1 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_phi_t_add)
    Im_ket_k1 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_phi_t_add)

    Re_temp = Re_phi_t_add + 0.5*Re_ket_k1
    Im_temp = Im_phi_t_add + 0.5*Im_ket_k1

    Re_ket_k2 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)
    Im_ket_k2 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_temp = Re_phi_t_add + 0.5*Re_ket_k2
    Im_temp = Im_phi_t_add + 0.5*Im_ket_k2

    Re_ket_k3 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)
    Im_ket_k3 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_temp = Re_phi_t_add + Re_ket_k3
    Im_temp = Im_phi_t_add + Im_ket_k3

    Re_ket_k4 = delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Im_temp)
    Im_ket_k4 = -delta_t*apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, e_zero, Re_temp)

    Re_phi_tdt_add = Re_phi_t_add + Re_ket_k1/6.0 + Re_ket_k2/3.0 + Re_ket_k3/3.0 + Re_ket_k4/6.0
    Im_phi_tdt_add = Im_phi_t_add + Im_ket_k1/6.0 + Im_ket_k2/3.0 + Im_ket_k3/3.0 + Im_ket_k4/6.0

    norm_Re_phi_tdt_add = np.linalg.norm(Re_phi_tdt_add)
    norm_Im_phi_tdt_add = np.linalg.norm(Im_phi_tdt_add)
    norm_phi_tdt_add = np.sqrt(pow(norm_Re_phi_tdt_add,2.0)+pow(norm_Im_phi_tdt_add,2.0))

    Re_phi_tdt_add *= init_add_norm/norm_phi_tdt_add
    Im_phi_tdt_add *= init_add_norm/norm_phi_tdt_add

    #Build the addition part of the GF
    p_Re_psi_tdt_add = pyscf.fci.addons.cre_a(Re_psi_t, ncas, nelecas, p)
    p_Im_psi_tdt_add = pyscf.fci.addons.cre_a(Im_psi_t, ncas, nelecas, p)

    g_imag_add = 2.0*(np.dot(np.ravel(Im_phi_tdt_add), np.ravel(p_Re_psi_tdt_add)) + np.dot(np.ravel(Re_phi_tdt_add), np.ravel(p_Im_psi_tdt_add)))
    g_real_add = 2.0*(np.dot(np.ravel(Re_phi_tdt_add), np.ravel(p_Re_psi_tdt_add)) - np.dot(np.ravel(Im_phi_tdt_add), np.ravel(p_Im_psi_tdt_add)))

    return Re_phi_tdt_rem, Im_phi_tdt_rem,  Re_phi_tdt_add, Im_phi_tdt_add, g_real_rem, g_imag_rem, g_real_add, g_imag_add
