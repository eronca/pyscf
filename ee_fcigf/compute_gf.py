import numpy as np
import scipy.linalg
import pyscf.lib.logger as logger
import pyscf.fci

# Make natural orbitals from CASSCF 1-RDM
def make_natorb(casscf):
    fcivec = casscf.ci
    mo = casscf.mo_coeff
    ncore = casscf.ncore
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    nocc = ncore + ncas

    casdm1 = casscf.fcisolver.make_rdm1(fcivec, ncas, nelecas)

    occ, ucas = scipy.linalg.eigh(casdm1)
    logger.debug(casscf, 'Natural occs')
    logger.debug(casscf, str(occ))
    natocc = np.zeros(mo.shape[1])
    natocc[:ncore] = 1
    natocc[ncore:nocc] = occ[::-1] * .5

    natorb_in_cas = np.dot(mo[:,ncore:nocc], ucas[:,::-1])
    natorb_on_ao = np.hstack((mo[:,:ncore], natorb_in_cas, mo[:,nocc:]))

    return natorb_on_ao, natocc


# Apply Hamiltonian on a vector
def apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, vec):

    nocc = ncore + ncas

    eri_cas = pyscf.fci.direct_spin1.absorb_h1e(h1eff[ncore:nocc,ncore:nocc].copy(), eri[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc], ncas, (nalpha, nbeta), 0.5)
    ham_vec = pyscf.fci.direct_spin1.contract_2e(eri_cas, vec, ncas, (nalpha, nbeta) )
    
    return ham_vec


# Ax = ([H - E_0] * [H - E_0] + [iomega + delta] * [iomega + delta]) * |I>
def compute_Ax(h1eff, eri, nalpha, nbeta, ncore, ncas, romega, iomega, delta, e_zero, vec):

    ham_vec = apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, vec)

    temp = ham_vec + (romega - e_zero) * vec

    print  "NORM Ri :", np.dot(np.ravel(ham_vec), np.ravel(ham_vec))

    ham_temp_vec = apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, temp)

    ham_squared = ham_temp_vec + (romega - e_zero) * temp

    Ax = (iomega + delta) * (iomega + delta) * vec + ham_squared
    
    return Ax


# Solve conjugate gradients for Ax = b, where Ax = ([H - E_0] * [H - E_0] + [iomega + delta] * [iomega + delta]) * |x>
def solve_conjugate_gradients(h1eff, eri, nalpha, nbeta, ncore, ncas, romega, iomega, delta, e_zero, vec, b, tol, maxiter):

    print  "NORM Xi :", np.dot(np.ravel(vec), np.ravel(vec))

    # Ax = ([H - E_0] * [H - E_0] + [iomega + delta] * [iomega + delta]) * |x>
    Ax = compute_Ax(h1eff, eri, nalpha, nbeta, ncore, ncas, romega, iomega, delta, e_zero, vec)

    # Compute residual
    r = b - Ax

    rms = np.linalg.norm(r)/np.sqrt(r.size)

    if rms < 1e-8:
        return vec

    d = r

    delta_new = np.dot(np.ravel(r), np.ravel(r))

    conv = False

    for imacro in range(maxiter):

        # Ad = ([H - E_0] * [H - E_0] + [iomega + delta] * [iomega + delta]) * d
        Ad = compute_Ax(h1eff, eri, nalpha, nbeta, ncore, ncas, romega, iomega, delta, e_zero, d)

        # q = A * d
        q = Ad

        # alpha = delta_new / d . q
        alpha = delta_new / (np.dot(np.ravel(d), np.ravel(q)))

        # x = x + alpha * d
        vec = vec + alpha * d

        r = r - alpha * q 

        delta_old = delta_new
        delta_new = np.dot(np.ravel(r), np.ravel(r))

        beta = delta_new/delta_old

        d = r + beta * d

        # Compute RMS of the residual
        rms = np.linalg.norm(r)/np.sqrt(r.size)
        
        print "Iteration ", imacro, ": RMS = %8.4e" % rms

        if abs(rms) < tol:
            conv = True
            break

    if conv:
        print "Iterations converged"
    else:
        raise Exception("Iterations did not converge")

    return vec


# Compute CASSCF Green's function matrix element for electron detachment
# G_pq(i iomega) = <|a_p^dag 1.0/(i iomega + H - E_0 + i delta) a_q|>
# Takes full set of one-electron integrals in the MO basis
# Takes full set of two-electron integtals in the MO basis in Chemist's notation
def gf_removal(romega, iomega, delta, e_zero, p, q, casscf, h1eff, eri, tol_gf, maxiter):

    # Get information about CASSCF
    fcivec = casscf.ci
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncore + ncas

    # Compute Dipole Integrals
    mo_cas_coeff = casscf.mo_coeff[:,casscf.ncore:casscf.ncore+casscf.ncas]
    AOdipInt = casscf.mol.intor('cint1e_r_sph', comp=3)

    MOdipInt = np.zeros([3,mo_cas_coeff.shape[1],mo_cas_coeff.shape[1]])

    for i in range(AOdipInt.shape[0]):
        MOdipInt[i] = reduce(np.dot, (mo_cas_coeff.T, AOdipInt[i], mo_cas_coeff))

    # Compute sum_pq(mu_pq a_p^(dagger)a_q|>)

    dip_wf = np.zeros((len(fcivec),len(fcivec)))

    mat = np.ones((ncas,ncas))

    dip_wf = pyscf.fci.direct_spin1.contract_1e(mat, fcivec, ncas, nelecas)

    print "NORM BEGIN : ", np.dot(np.ravel(dip_wf), np.ravel(dip_wf))

    nelecas = casscf.nelecas
    nalpha = nelecas[0]
    nbeta = nelecas[1]

    # Iterate imaginary part of the Green's function
    print "\nIterating imaginary part of the Green's function :"

    ci_norm = np.sqrt(np.linalg.norm(dip_wf))
    tol = tol_gf * ci_norm
    print "Convergence threshold: ", tol

    conv = False

    # Generate guess for |I>
    # |I> = - (iomega + delta) * a_q|>
    ivec = - (iomega + delta) * dip_wf

    # b = - (iomega + delta) * a_q|>
    b = - (iomega + delta) * dip_wf

    # Solve for imaginary part
    # ([H - E_0] * [H - E_0] + [iomega + delta] * [iomega + delta]) * |I> = - (iomega + delta) * a_q|>
    ivec = solve_conjugate_gradients(h1eff, eri, nalpha, nbeta, ncore, ncas, -romega, iomega, delta, e_zero, ivec, b, tol, maxiter)
 
    # Iterate real part of the Green's function
    #print "\nIterating real part of the Green's function (removal):"

    # Generate guess for |R>
    # |R> = [-1 / (iomega + delta)] * (H - E_0) * |I>
    #ham_i = apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, ivec)
    #rvec = - (1/(iomega + delta)) * (ham_i + (romega - e_zero) * ivec)

    #b = (H - E_0) * a_q|> 
    #b = apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, dip_wf) + (romega - e_zero) * dip_wf

    # Solve for real part
    #rvec = solve_conjugate_gradients(h1eff, eri, nalpha, nbeta, ncore, ncas, romega, iomega, delta, e_zero, rvec, b, tol, maxiter)

    print "NORM IVEC :", np.dot(np.ravel(ivec), np.ravel(ivec))

    g_imag = np.dot(np.ravel(ivec), np.ravel(dip_wf))
    #g_real = np.dot(np.ravel(rvec), np.ravel(dip_wf))
    g_real = g_imag

    return g_real, g_imag

    
# Compute CASSCF Green's function matrix element for electron attachment
# G_pq(i iomega) = <|a_p 1.0/(i iomega - H + E_0 + i delta) a_q^dag |>
# Takes full set of one-electron integrals in the MO basis
# Takes full set of two-electron integtals in the MO basis in Chemist's notation
def gf_addition(romega, iomega, delta, e_zero, p, q, casscf, h1eff, eri, tol_gf, maxiter):

    # Get information about CASSCF
    fcivec = casscf.ci
    ncas = casscf.ncas
    nelecas = casscf.nelecas
    ncore = casscf.ncore
    nocc = ncore + ncas

    # Compute N+1 electron CI space a_q^dag |> 
    q_cas = pyscf.fci.addons.cre_a(fcivec, ncas, nelecas, q)
    nalpha = nelecas[0] + 1
    nbeta = nelecas[1]

    # Iterate imaginary part of the Green's function
    print "\nIterating imaginary part of the Green's function (addition):"

    ci_norm = np.sqrt(np.linalg.norm(q_cas))
    tol = tol_gf * ci_norm
    print "Convergence threshold: ", tol

    conv = False

    # Generate guess for |I>
    # |I> = - (iomega + delta) * a_q^dag |>
    ivec = - (iomega + delta) * q_cas

    # b = - (iomega + delta) * a_q^dag |>
    b = - (iomega + delta) * q_cas

    # Solve for imaginary part
    # ([H - E_0 - romega] * [H - E_0 - romega] + [iomega + delta] * [iomega + delta]) * |I> = - (iomega + delta) * a_q^dag |>
    ivec = solve_conjugate_gradients(h1eff, eri, nalpha, nbeta, ncore, ncas, -romega, iomega, delta, e_zero, ivec, b, tol, maxiter)

    # Iterate real part of the Green's function
    print "\nIterating real part of the Green's function (addition):"

    # Generate guess for |R>
    # |R> = [-1 / (iomega + delta)] * (romega - H + E_0) * |I>
    ham_i = apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, ivec)
    rvec = - (1/(iomega + delta)) * (-ham_i + (romega + e_zero) * ivec)

    #==DEBUG==============================================
    print "NORM ivec : ", np.linalg.norm(ivec)
    print "NORM rvec : ", np.linalg.norm(rvec)
    rrvec = (-ham_i + (e_zero) * ivec)
    print "NORM rvec no omega-1/eta : ", np.linalg.norm(rrvec)
    rrvec = (-ham_i + (romega + e_zero) * ivec)
    print "NORM rvec no -1/eta : ", np.linalg.norm(rrvec)
    #==DEBUG==============================================

    # b = (-H + romega + E_0) * a_q^dag |> 
    b = -apply_H(h1eff, eri, nalpha, nbeta, ncore, ncas, q_cas) + (romega + e_zero) * q_cas

    #==DEBUG==============================================
    print "NORM b : ", np.linalg.norm(b)
    #==DEBUG==============================================

    # Solve for real part
    rvec = solve_conjugate_gradients(h1eff, eri, nalpha, nbeta, ncore, ncas, -romega, iomega, delta, e_zero, rvec, b, tol, maxiter)

    #==DEBUG==============================================
    print "NORM rvec final : ", np.linalg.norm(rvec)
    #==DEBUG==============================================

    # Compute N+1 electron CI space <|a_p
    p_cas = pyscf.fci.addons.cre_a(fcivec, ncas, nelecas, p)

    g_imag = 2 * np.dot(np.ravel(ivec), np.ravel(p_cas))
    g_real = 2 * np.dot(np.ravel(rvec), np.ravel(p_cas))

    return g_real, g_imag


# Returns mean-field Green's function in the AO basis
def gf_mean_field(romega, iomega, delta, mf):

    mo_energy = mf.mo_energy
    mo = mf.mo_coeff
    nmo = mo.shape[1]
    ovlp = mf.get_ovlp()

    gf_mf = 2/(romega - mo_energy + 1j * (iomega + delta))

    gf_mf_ = np.zeros((nmo, nmo), 'complex')

    for p in range(nmo):
        gf_mf_[p,p] = gf_mf[p]

    # g(omega)_AO = S C g(omega) C^T S
    mo_inv = np.dot(mo.T, ovlp)

    gf_mf_ao = reduce(np.dot, (mo_inv.T, gf_mf_, mo_inv))

    return gf_mf_ao


