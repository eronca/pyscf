import numpy as np
import scipy.linalg
import pyscf.gto
import pyscf.scf
import pyscf.mcscf
import pyscf.ao2mo
import pyscf.lib.logger as logger
import math
import sys
import timeit
from math import exp
import cmath
from pyscf.fcigf import compute_gf
from compute_gf import *
from scipy import fft, arange
import matplotlib.pyplot as plt

def run_gf_casscf(mc, mf, mol, emc, time, time_real, eta, npoints):

    print "\nStarting RT-GF code..."
    print "(c) Enrico Ronca, 2015\n"

    start = timeit.default_timer()
    
    np.set_printoptions(linewidth=150, edgeitems=10, suppress=True)

    # Collect general information
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    mo = mc.mo_coeff
    nmo = mo.shape[1]
    nvir = nmo - nocc
    nelec = mol.nelectron
    nelecas = mc.nelecas
    ovlp = mf.get_ovlp()
    
    print "Ground state CASSCF energy: ", emc
    print "Number of electrons:        ", nelec
    print "Number of basis functions:  ", nmo
    print "Number of core orbitals:    ", ncore
    print "Number of external orbitals:", nvir
    print "Number of active orbitals:  ", ncas
    print "Number of active electrons: ", nelecas
    
    print "SCF orbital energies:"
    print mf.mo_energy
    
    # Transform integrals
    h1e_ao = mc.get_hcore()
    h1e_mo = reduce(np.dot, (mo.T, h1e_ao, mo))
    
    v2e = pyscf.ao2mo.incore.full(mf._eri, mo) # v2e has 4-fold symmetry now
    v2e = pyscf.ao2mo.restore(1, v2e, nmo) # to remove 4-fold symmetry, turn v2e to n**4 array
    
    # Compute core Hamiltonian with frozen-core two-electron component
    h1eff = h1e_mo + 2 * np.einsum('pqrr->pq', v2e[:,:,:ncore,:ncore]) - np.einsum('prqr->pq', v2e[:,:ncore,:,:ncore])
    
    # Test CASSCF energy
    fcivec = mc.ci
    eri_cas = pyscf.fci.direct_spin1.absorb_h1e(h1eff[ncore:nocc,ncore:nocc], v2e[ncore:nocc,ncore:nocc,ncore:nocc,ncore:nocc], ncas, nelecas, .5)
    ham_vec = pyscf.fci.direct_spin1.contract_2e(eri_cas, fcivec, ncas, nelecas )
    
    e_cas = np.dot(np.ravel(fcivec), np.ravel(ham_vec))
    e_fc = 2.0 * np.einsum('pp', h1e_mo[:ncore,:ncore])
    print v2e[:ncore,:ncore,:ncore,:ncore].shape
    v_ppqq = np.einsum('prqq->pr', v2e[:ncore,:ncore,:ncore,:ncore])
    e_ppqq = np.einsum('pp', v_ppqq)
    v_pqpq = np.einsum('pqrq->pr', v2e[:ncore,:ncore,:ncore,:ncore])
    e_pqpq = np.einsum('pp', v_pqpq)
    e_fc += 2 * e_ppqq - e_pqpq
    
    print "Energy(FC)                   %15.10f" % e_fc
    print "Energy(CAS)                  %15.10f" % e_cas 
    print "Total Electronic Energy      %15.10f" % (e_cas + e_fc) 
    
    # Compute Green's function matrix elements
    # Range
    time_range = time[1] - time[0]
    delta_t = time_range / npoints
    
    print "\nTime range: ", time
    print "Number of points: ", npoints
    print "Step size: ", delta_t
    
    time_array = np.arange(time[0], time[1], delta_t)

    density_of_states_rem = np.zeros((npoints))
    real_part_rem = np.zeros((npoints))
    density_of_states_add = np.zeros((npoints))
    real_part_add = np.zeros((npoints))
    fftinp = np.zeros((npoints))

    for orb in range(ncas):
        print "Calculating Green's Function for orbital : ", orb
        Re_psi_t = mc.ci
        Im_psi_t = np.zeros(Re_psi_t.shape)
        Re_phi_t_rem = pyscf.fci.addons.des_a(mc.ci, ncas, nelecas, orb)
        Im_phi_t_rem = np.zeros(Re_phi_t_rem.shape)
        Re_phi_t_add = pyscf.fci.addons.cre_a(mc.ci, ncas, nelecas, orb)
        Im_phi_t_add = np.zeros(Re_phi_t_add.shape)

        gf_real_rem = np.zeros((npoints))
        gf_imag_rem = np.zeros((npoints))
        gf_real_add = np.zeros((npoints))
        gf_imag_add = np.zeros((npoints))

        init_rem_norm = np.linalg.norm(Re_phi_t_rem)
        init_add_norm = np.linalg.norm(Re_phi_t_add)

        gf_real_rem[0] = 2.0*np.dot(np.ravel(Re_phi_t_rem), np.ravel(Re_phi_t_rem))
        gf_real_add[0] = 2.0*np.dot(np.ravel(Re_phi_t_add), np.ravel(Re_phi_t_add))

        for j in range(1, npoints, 1): 
            print "Calculating Removal Part"
            print "Propagation at time step : ", time_array[j]
            psi_t = Re_psi_t+1j*Im_psi_t
            phi_t_rem = Re_phi_t_rem+1j*Im_phi_t_rem
            phi_t_add = Re_phi_t_add+1j*Im_phi_t_add
            print "Norm Psi(t) : ", np.linalg.norm(psi_t)
            print "Norm Phi(t) Removal : ", np.linalg.norm(phi_t_rem)
            print "Norm Phi(t) Addition : ", np.linalg.norm(phi_t_add)
            Re_phi_t_rem, Im_phi_t_rem, Re_phi_t_add, Im_phi_t_add, \
            gf_real_rem[j], gf_imag_rem[j], gf_real_add[j], gf_imag_add[j] = gf_calculation(Re_psi_t, Im_psi_t, Re_phi_t_rem, Im_phi_t_rem,
                                                                                            Re_phi_t_add, Im_phi_t_add, init_rem_norm, init_add_norm, delta_t, e_cas, orb, orb, mc, h1eff, v2e)


        density_of_states_rem += gf_imag_rem
        real_part_rem += gf_real_rem
        density_of_states_add += gf_imag_add
        real_part_add += gf_real_add

    frq = np.fft.fftfreq(npoints,delta_t)
    frq = np.fft.fftshift(frq)*2.0*np.pi

    fftinp_rem = 1j*(real_part_rem + 1j*density_of_states_rem)
    fftinp_add = 1j*(real_part_add - 1j*density_of_states_add)

    # Plot the results
    with open('time_real_part_rem.txt', 'w') as fout:
        fout.write('#     Time          A(Time)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (time_array[i], real_part_rem[i]))
 
    with open('time_density_of_states_rem.txt', 'w') as fout:
        fout.write('#     Time          A(Time)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (time_array[i], density_of_states_rem[i]))

    with open('time_real_part_add.txt', 'w') as fout:
        fout.write('#     Time          A(Time)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (time_array[i], real_part_add[i]))

    with open('time_density_of_states_add.txt', 'w') as fout:
        fout.write('#     Time          A(Time)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (time_array[i], density_of_states_add[i]))
    #==================================================================

    for i in range(npoints):
        fftinp_rem[i] = fftinp_rem[i]*np.exp(-eta*time_array[i])
        fftinp_add[i] = fftinp_add[i]*np.exp(-eta*time_array[i])

    Y_rem = fft(fftinp_rem)
    Y_rem = np.fft.fftshift(Y_rem)
    Y_add = fft(fftinp_add)
    Y_add = np.fft.fftshift(Y_add)


    Y_real = Y_rem.real + Y_add.real
    Y_real = (Y_real*time[1]/npoints)
    Y_imag = Y_rem.imag + Y_add.imag
    Y_imag = (Y_imag*time[1]/npoints)/np.pi
    

    # Plot the results
    with open('density_of_states.txt', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (frq[i], Y_imag[i]))

    with open('real_part.txt', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (frq[i], Y_real[i]))

#    import scipy.integrate
#    Y_integral = [x for i, x in enumerate(Y_imag) if (frq[i] >= -1.5 and frq[i] <= 1.5)]
#    frq_int = [x for x in frq if (x >= -1.5 and x <= 1.5)]
#    print "Integral of the Density of States: ", scipy.integrate.simps(np.array(Y_integral), frq_int) 

    #Non-interating real-time Green's Function
    #Implemented following Mattuck, A Guide to Feynman Diagrams in the Many Body Problem, Chap. 3, 4 and 9

    occ = range(nelecas[0]//2)
    virt = np.delete(range(ncas), occ)
    orb_energies = mf.mo_energy[ncore:]

    fftinp_real = np.zeros(npoints)
    fftinp_imag = np.zeros(npoints)
    for i in range(npoints):
       if i == 0:
         fftinp_real[i] = 1.0
         fftinp_imag[i] = 0.0
       else:
         for orb in occ:
             fftinp_real[i] = fftinp_real[i] + math.cos(time_array[i]*orb_energies[orb])
             fftinp_imag[i] = fftinp_imag[i] + math.sin(time_array[i]*orb_energies[orb])
         for orb in virt:
             fftinp_real[i] = fftinp_real[i] + math.cos(time_array[i]*orb_energies[orb])
             fftinp_imag[i] = fftinp_imag[i] + math.sin(time_array[i]*orb_energies[orb])

    fftinp = -1j*(fftinp_real + 1j*fftinp_imag)
    for i in range(npoints):
        fftinp[i] = fftinp[i]*np.exp(-eta*time_array[i])

    Y = fft(fftinp)
    Y = np.fft.fftshift(Y)

    Y = -2.0*(Y*time[1]/npoints)/np.pi


    # Plot the results
    with open('density_of_states_mf.txt', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (frq[i], Y[i].imag))

    with open('real_part_mf.txt', 'w') as fout:
        fout.write('#     Omega          A(Omega)\n')
        for i in range(npoints):
            fout.write('%6.3f  %8.4f\n' % (frq[i], Y[i].real))
 
    stop = timeit.default_timer()

    print ("--- %s seconds ---" % (stop - start))
